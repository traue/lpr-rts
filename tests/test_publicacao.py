"""Politica de publicacao: revisao, decisao na saida e o que o banco e a avaliacao veem."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from lprts.analytics.store import SCHEMA_VERSION, ResultStore
from lprts.config import AppConfig, ConfigError
from lprts.core.pipeline import LPRPipeline
from lprts.core.types import Detection, Frame, PlateLayout, PlateRead, TrackState
from lprts.detect.base import Detector
from lprts.fusion.publication import PublicationPolicy
from lprts.fusion.voting import PlateVoter
from lprts.ocr import plates_br
from lprts.ocr.base import OCREngine
from lprts.optimize.refusion import replay_decisions, simulate_online

IMG = np.zeros((720, 1280, 3), dtype=np.uint8)
CAIXA = (100.0, 50.0, 220.0, 90.0)
# A trilha comeca na placa errada (tipico de troca de identidade sob borrao) e
# depois le a certa por mais tempo.
ROTEIRO = ["ABC1D23"] * 3 + ["XYZ9K88"] * 8


class PlacaFixa(Detector):
    name = "placa_fixa"

    def detect(self, image):
        return [Detection(CAIXA, 0.9, "plate", 0)]


class OCRRoteiro(OCREngine):
    name = "roteiro"

    def __init__(self, textos: list[str]) -> None:
        self.textos, self.i = textos, 0

    def read(self, plate_crop):
        texto = self.textos[min(self.i, len(self.textos) - 1)]
        self.i += 1
        c = plates_br.correct(texto)
        return PlateRead(text=c.text, confidence=0.9, layout=c.layout, valid_syntax=c.valid,
                         engine=self.name, raw_text=texto, n_corrections=c.corrections)


def _leituras(textos: list[str]) -> list[PlateRead]:
    return [PlateRead(text=t, confidence=0.9, valid_syntax=True, layout=PlateLayout.MERCOSUL,
                      frame_index=i) for i, t in enumerate(textos)]


def _rodar(tmp_path, textos, quadros, nome="p.db", **fusao):
    db = tmp_path / nome
    cfg = AppConfig()
    cfg.analytics.save_crops = False
    cfg.ocr.min_plate_width = 10
    cfg.ocr.ocr_every_n = 1
    cfg.ocr.max_reads_published = 0
    cfg.override(**{"detector.detect_vehicles": False, "fusion.min_reads": 3,
                    "fusion.max_reads_per_track": 0,
                    **{f"fusion.{k}": v for k, v in fusao.items()}})
    with ResultStore(db) as store:
        store.start_session("s")
        pipe = LPRPipeline(cfg, None, PlacaFixa(), OCRRoteiro(textos), store).start()
        for i in range(quadros):
            pipe.process_frame(Frame(i, IMG, timestamp=0.0, source_pts=i / 30))
        pipe.finalize()
        pipe.stop()
    return db


class TestPolitica:
    def _trilha(self):
        return TrackState(track_id=1, bbox=CAIXA, score=0.9)

    def test_politica_invalida_e_erro(self):
        with pytest.raises(ConfigError):
            PublicationPolicy(PlateVoter(), "depois")
        with pytest.raises(ConfigError):
            PublicationPolicy(PlateVoter(), "revise", exit_min_reads=-1)

    def test_first_stable_nao_revisa(self):
        voter = PlateVoter(min_reads=3)
        pol = PublicationPolicy(voter, "first_stable")
        t = self._trilha()
        primeira = pol.on_update(t, voter.vote(_leituras(["ABC1D23"] * 3)))
        pol.apply(t, primeira, 3, 0.1)
        assert pol.on_update(t, voter.vote(_leituras(ROTEIRO))) is None
        assert pol.on_exit(t, voter.vote(_leituras(ROTEIRO))) is None

    def test_revise_substitui_e_guarda_a_primeira(self):
        voter = PlateVoter(min_reads=3)
        pol = PublicationPolicy(voter, "revise")
        t = self._trilha()
        pol.apply(t, pol.on_update(t, voter.vote(_leituras(["ABC1D23"] * 3))), 3, 0.1)
        rev = pol.on_update(t, voter.vote(_leituras(ROTEIRO)))
        assert (rev.kind, rev.revision, rev.text) == ("revision", 1, "XYZ9K88")
        pol.apply(t, rev, 11, 0.4)
        assert t.published_text == "XYZ9K88" and t.first_published_text == "ABC1D23"
        assert t.n_revisions == 1 and t.published_frame == 3     # instante da 1a decisao

    def test_on_exit_so_decide_no_fim(self):
        voter = PlateVoter(min_reads=3)
        pol = PublicationPolicy(voter, "on_exit")
        t = self._trilha()
        assert pol.on_update(t, voter.vote(_leituras(ROTEIRO))) is None
        assert pol.on_exit(t, voter.vote(_leituras(ROTEIRO))).text == "XYZ9K88"

    def test_saida_relaxada_so_para_trilha_sem_decisao(self):
        voter = PlateVoter(min_reads=3)
        uma = voter.vote(_leituras(["ABC1D23"]))
        assert PublicationPolicy(voter).on_exit(self._trilha(), uma) is None
        d = PublicationPolicy(voter, exit_min_reads=1).on_exit(self._trilha(), uma)
        assert (d.kind, d.text) == ("exit_relaxed", "ABC1D23")
        # Os demais criterios continuam valendo: sintaxe invalida nao publica.
        ruim = voter.vote([PlateRead(text="AB12", confidence=0.9, valid_syntax=False)])
        assert PublicationPolicy(voter, exit_min_reads=1).on_exit(self._trilha(), ruim) is None


class TestReplica:
    @pytest.mark.parametrize("politica,esperado", [
        ("first_stable", "ABC1D23"), ("revise", "XYZ9K88"), ("on_exit", "XYZ9K88"),
    ])
    def test_replica_aplica_a_politica(self, politica, esperado):
        texto, _ = simulate_online(_leituras(ROTEIRO), PlateVoter(min_reads=3), politica)
        assert texto == esperado

    def test_revisao_registra_primeira_decisao(self):
        r = replay_decisions(_leituras(ROTEIRO), PlateVoter(min_reads=3), "revise")
        assert (r.first_text, r.final_text, r.n_revisions, r.n_reads_first) == \
            ("ABC1D23", "XYZ9K88", 1, 3)

    @pytest.mark.parametrize("politica,saida", [
        ("first_stable", 0), ("revise", 0), ("on_exit", 0), ("first_stable", 1),
    ])
    def test_pipeline_e_replica_concordam(self, tmp_path, politica, saida):
        """A regra existe num lugar so: o pipeline e o tune decidem igual."""
        textos = ROTEIRO if not saida else ["ABC1D23"]
        db = _rodar(tmp_path, textos, 16 if not saida else 4, publish_policy=politica,
                    exit_min_reads=saida)
        con = sqlite3.connect(db)
        (publicado,) = con.execute("SELECT published_text FROM tracks").fetchone()
        reads = [PlateRead(text=t, confidence=c, valid_syntax=bool(v), frame_index=f,
                           layout=PlateLayout.MERCOSUL)
                 for t, c, v, f in con.execute(
                     "SELECT text, confidence, valid_syntax, frame_index FROM reads "
                     "WHERE text <> '' ORDER BY frame_index")]
        r = replay_decisions(reads, PlateVoter(min_reads=3, max_reads_per_track=0), politica, saida)
        assert publicado == (r.final_text or "")


class TestBanco:
    def test_revisao_gravada_e_decisao_vigente(self, tmp_path):
        con = sqlite3.connect(_rodar(tmp_path, ROTEIRO, 16, publish_policy="revise"))
        eventos = con.execute(
            "SELECT plate_text, decision, revision FROM events ORDER BY revision").fetchall()
        assert eventos == [("ABC1D23", "stable", 0), ("XYZ9K88", "revision", 1)]
        assert con.execute("SELECT plate_text FROM v_decisions").fetchall() == [("XYZ9K88",)]
        assert con.execute(
            "SELECT published_text, first_published_text, n_revisions, published_decision "
            "FROM tracks").fetchone() == ("XYZ9K88", "ABC1D23", 1, "revision")
        assert con.execute("SELECT n_events FROM v_unique_plates").fetchall() == [(1,)]

    def test_padrao_continua_sem_revisao(self, tmp_path):
        con = sqlite3.connect(_rodar(tmp_path, ROTEIRO, 16))
        assert con.execute("SELECT plate_text, decision, revision FROM events").fetchall() == \
            [("ABC1D23", "stable", 0)]

    def test_banco_da_versao_2_e_migrado_sem_perda(self, tmp_path):
        """A 0.2 so conhecia a 1a decisao estavel: decision='stable', revision=0 sao exatos."""
        db = _rodar(tmp_path, ROTEIRO, 16)
        con = sqlite3.connect(db)
        # Reconstroi o esquema 2: sem as colunas, a visao e o indice novos.
        for sql in ("DROP VIEW v_unique_plates", "DROP VIEW v_decisions",
                    "DROP INDEX idx_events_track",
                    "ALTER TABLE events DROP COLUMN decision",
                    "ALTER TABLE events DROP COLUMN revision",
                    "ALTER TABLE events DROP COLUMN decision_elapsed_ms",
                    "ALTER TABLE tracks DROP COLUMN first_published_text",
                    "ALTER TABLE tracks DROP COLUMN n_revisions",
                    "ALTER TABLE tracks DROP COLUMN published_decision",
                    "PRAGMA user_version = 2"):
            con.execute(sql)
        con.commit()
        con.close()

        from lprts.analytics.store import legacy_schema_version

        assert legacy_schema_version(db) is None        # migravel: nao vai para legado/
        with ResultStore(db) as store:
            store.start_session("nova")
        con = sqlite3.connect(db)
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert con.execute("SELECT plate_text, decision, revision FROM events").fetchall() == \
            [("ABC1D23", "stable", 0)]
        assert con.execute(
            "SELECT first_published_text, published_decision FROM tracks").fetchone() == \
            ("ABC1D23", "stable")
        assert con.execute("SELECT COUNT(*) FROM v_decisions").fetchone()[0] == 1


class TestAvaliacao:
    def test_metricas_usam_a_decisao_vigente(self, tmp_path):
        from lprts.eval.groundtruth import GroundTruth, GTAnnotation
        from lprts.eval.session import evaluate_session

        db = _rodar(tmp_path, ROTEIRO, 16, publish_policy="revise")
        gt = GroundTruth(annotations=[GTAnnotation(i, CAIXA, "XYZ9K88", track_id=1)
                                      for i in range(16)])
        v = evaluate_session(db, None, gt, n_boot=200).vehicle_level
        assert v["recall"] == 1.0 and v["precision"] == 1.0     # vigente: XYZ9K88, certa
        assert v["first_decision_recall"] == 0.0                 # 1a decisao: ABC1D23, errada
        assert v["precision_emitted"] == 0.5 and v["n_revisions"] == 1
        assert v["n_duplicate_events"] == 0                      # revisao nao e duplicata
