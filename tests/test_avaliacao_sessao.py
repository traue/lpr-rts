"""Avaliacao a partir do banco: pareamento por posicao, metricas por veiculo e inferencia."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lprts.analytics.store import ResultStore
from lprts.core.types import Detection, Frame, FrameResult, PlateRead, TrackState
from lprts.eval.groundtruth import GroundTruth, GTAnnotation, load_ground_truth
from lprts.eval.metrics import evaluate_detection
from lprts.eval.session import associate_tracks, evaluate_session, tracking_metrics
from lprts.eval.stats import (
    bootstrap_ci,
    mcnemar_exact,
    paired_comparison,
    ratio_bootstrap_ci,
    wilson_ci,
)

RAIZ = Path(__file__).resolve().parent.parent


class TestEstatistica:
    def test_bootstrap_contem_a_media(self):
        est, lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], seed=3)
        assert est == pytest.approx(0.7)
        assert lo <= est <= hi and lo >= 0.0 and hi <= 1.0

    def test_bootstrap_e_reproduzivel(self):
        assert bootstrap_ci([0, 1] * 20, seed=7) == bootstrap_ci([0, 1] * 20, seed=7)

    def test_razao_por_agrupamento(self):
        est, lo, hi = ratio_bootstrap_ci([9, 1, 5], [10, 10, 10])
        assert est == pytest.approx(15 / 30)
        assert lo < est < hi

    def test_mcnemar_exato(self):
        assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
        assert mcnemar_exact(3, 3) == 1.0
        assert mcnemar_exact(0, 0) == 1.0

    def test_comparacao_pareada(self):
        r = paired_comparison([True, False, False, True], [True, True, True, True])
        assert r["only_b_correct"] == 2 and r["only_a_correct"] == 0
        assert r["diff_b_minus_a"] == pytest.approx(0.5)

    def test_wilson(self):
        lo, hi = wilson_ci(8, 10)
        assert lo < 0.8 < hi


class TestDificeis:
    def test_anotacao_dificil_nao_e_erro_nem_acerto(self):
        """Protocolo VOC: predicao sobre placa 'difficult' nao e FP, e ela nao e FN."""
        preds = {0: [((10, 10, 50, 50), 0.9), ((100, 100, 140, 140), 0.8)]}
        gts = {0: [(10, 10, 50, 50)]}
        ign = {0: [(100, 100, 140, 140)]}
        m = evaluate_detection(preds, gts, 0.5, ign)
        assert (m.tp, m.fp, m.fn) == (1, 0, 0)

    def test_sem_ignorar_seria_falso_positivo(self):
        preds = {0: [((10, 10, 50, 50), 0.9), ((100, 100, 140, 140), 0.8)]}
        m = evaluate_detection(preds, {0: [(10, 10, 50, 50)]}, 0.5)
        assert m.fp == 1

    def test_restricao_aos_quadros_processados(self):
        m = evaluate_detection({}, {0: [(0, 0, 10, 10)], 1: [(0, 0, 10, 10)]}, 0.5, frames={0})
        assert m.fn == 1


class TestAssociacao:
    def test_trilha_vai_para_o_veiculo_que_ela_cobre(self):
        gt = {"id:1": dict.fromkeys(range(10), (100, 100, 200, 140)),
              "id:2": dict.fromkeys(range(10), (500, 100, 600, 140))}
        trilhas = {7: dict.fromkeys(range(10), (101, 101, 199, 139)),
                   8: dict.fromkeys(range(10), (900, 500, 950, 520))}
        a = associate_tracks(trilhas, gt)
        assert a[7]["vehicle"] == "id:1" and a[7]["purity"] == 1.0
        assert 8 not in a


class TestRastreamento:
    def _gt(self):
        return {"id:1": {f: (10 * f, 0, 10 * f + 50, 20) for f in range(10)}}

    def test_rastreio_perfeito(self):
        r = tracking_metrics({1: self._gt()["id:1"]}, self._gt(), set(range(10)))
        assert r["mota"] == pytest.approx(1.0) and r["idf1"] == pytest.approx(1.0)
        assert r["id_switches"] == 0

    def test_troca_de_identidade(self):
        caixas = self._gt()["id:1"]
        sis = {1: {f: b for f, b in caixas.items() if f < 5},
               2: {f: b for f, b in caixas.items() if f >= 5}}
        r = tracking_metrics(sis, self._gt(), set(range(10)))
        assert r["id_switches"] == 1
        assert r["idf1"] == pytest.approx(0.5)


class TestGroundTruth:
    def test_tipo_inferido_e_veiculos(self):
        gt = GroundTruth(annotations=[
            GTAnnotation(1, (0, 0, 10, 5), "ABC1D23", track_id=3),
            GTAnnotation(1, (0, 0, 40, 30), "", track_id=3),
            GTAnnotation(2, (2, 0, 12, 5), "ABC1D23", track_id=3, difficult=True),
        ])
        assert [a.kind for a in gt.annotations] == ["plate", "object", "plate"]
        v = gt.vehicles()["id:3"]
        assert v.n_frames == 2 and v.difficult_frames == {2} and not v.all_difficult
        assert gt.has_track_ids()


PLACA = (0.0, 0.0, 100.0, 20.0)


def _gt_um_veiculo(objetos: bool = True) -> GroundTruth:
    """Veiculo id:1 anotado do quadro 0 ao 30 (placa e, opcionalmente, objeto)."""
    anotacoes = [GTAnnotation(f, PLACA, "ABC1D23", track_id=1) for f in range(31)]
    if objetos:
        anotacoes += [GTAnnotation(f, (0, 0, 120, 80), "", track_id=1, kind="object")
                      for f in range(31)]
    return GroundTruth(annotations=anotacoes, fps=30.0)


def _sessao_com_decisao(db, elapsed_ms, realtime=False):
    """Trilha vista pela 1a vez em 0,5 s; decisao correta no quadro 30 (1,0 s)."""
    trilha = TrackState(track_id=1, bbox=PLACA, score=.9, plate_text="ABC1D23",
                        first_seen=0.5)
    with ResultStore(db, async_writes=False) as store:
        store.start_session("teste", source_fps=30, realtime=realtime,
                            config={"source": {"realtime": realtime}})
        store.add_read(1, PlateRead(text="ABC1D23", confidence=.9, bbox=PLACA,
                                    frame_index=15))
        store.add_event(trilha, 30, 1.0, decision_elapsed_ms=elapsed_ms)
        return store.session_id


class TestDoisRelogios:
    """Latencia de decisao em tempo de video; o relogio do host em campos proprios."""

    def test_latencia_em_tempo_de_video_e_relogio_separado(self, tmp_path):
        db = tmp_path / "relogios.db"
        r = evaluate_session(db, _sessao_com_decisao(db, 250.0), _gt_um_veiculo(), n_boot=10)
        evento, veiculo, nivel = r.events[0], r.vehicles[0], r.vehicle_level
        # Tempo de video: da trilha (0,5 s) e do 1o quadro anotado (0 s) ate 1,0 s.
        assert evento["decision_delay_s"] == pytest.approx(0.5)
        assert veiculo["time_to_correct_s"] == pytest.approx(1.0)
        assert nivel["decision_delay_median_s"] == pytest.approx(1.0)
        assert nivel["decision_delay_n"] == 1
        # Relogio do host: medido no pipeline, nunca derivado do video.
        assert evento["decision_elapsed_s"] == pytest.approx(0.25)
        assert veiculo["elapsed_to_correct_s"] == pytest.approx(0.25)
        assert nivel["decision_elapsed_median_s"] == pytest.approx(0.25)
        assert nivel["decision_elapsed_n"] == 1

    def test_prazo_de_quadro_nao_vira_metrica_de_decisao(self, tmp_path):
        # Uma decisao com min_reads leituras nunca cabe num periodo de quadro: a
        # razao "dentro do prazo" seria 1 offline e 0 em tempo real por construcao.
        db = tmp_path / "prazo.db"
        r = evaluate_session(db, _sessao_com_decisao(db, 250.0), _gt_um_veiculo(), n_boot=10)
        assert not {"recall_within_deadline", "decision_deadline_ms"} & set(r.vehicle_level)
        assert "correct_within_deadline" not in r.vehicles[0]

    def test_banco_sem_relogio_nao_vira_tempo_zero(self, tmp_path):
        db = tmp_path / "antigo.db"
        r = evaluate_session(db, _sessao_com_decisao(db, None), _gt_um_veiculo(), n_boot=10)
        assert r.vehicle_level["decision_delay_median_s"] == pytest.approx(1.0)
        assert r.vehicle_level["decision_elapsed_n"] == 0
        assert r.vehicle_level["decision_elapsed_median_s"] is None
        assert r.vehicles[0]["elapsed_to_correct_s"] is None
        assert any("sem decision_elapsed_ms" in a for a in r.warnings)

    @pytest.mark.parametrize("realtime", [False, True])
    def test_relogio_so_aparece_no_texto_em_tempo_real(self, tmp_path, realtime):
        db = tmp_path / "texto.db"
        r = evaluate_session(db, _sessao_com_decisao(db, 250.0, realtime), _gt_um_veiculo(),
                             n_boot=10)
        texto = r.to_text()
        assert "Latencia de decisao" in texto
        assert ("no relogio" in texto) is realtime


def _sessao_com_quadro(db, detect_vehicles=True, fallback=False, objeto=False):
    """Um quadro detectado; opcionalmente uma deteccao de objeto gravada."""
    config = {"analytics": {"store_detections": True},
              "detector": {"detect_vehicles": detect_vehicles}}
    with ResultStore(db, async_writes=False) as store:
        store.start_session("teste", source_fps=30, config=config)
        store.add_frame_telemetry(FrameResult(
            frame=Frame(0, np.zeros((20, 20, 3), dtype=np.uint8), 0), detection_ran=True,
        ), 30)
        if objeto:
            store.add_detection(0, 0.0, "object", Detection((0, 0, 120, 80), .9, "car", 2))
        sid = store.session_id
        store.finish_session("fim_da_fonte", stats={"fallback_active": fallback})
    return sid


class TestDeteccaoDeObjetos:
    """So se mede o que foi procurado e anotado."""

    def test_zero_deteccao_com_detector_ligado_e_erro_medido(self, tmp_path):
        db = tmp_path / "ligado.db"
        r = evaluate_session(db, _sessao_com_quadro(db), _gt_um_veiculo(), n_boot=10)
        assert r.object_detection is not None and r.object_detection.fn == 1
        assert r.object_detection.recall == 0
        assert r.detection is not None and r.detection.fn == 1

    def test_detector_desligado_nao_e_medido(self, tmp_path):
        db = tmp_path / "desligado.db"
        r = evaluate_session(db, _sessao_com_quadro(db, detect_vehicles=False),
                             _gt_um_veiculo(), n_boot=10)
        assert r.object_detection is None
        assert "DETECCAO DE OBJETOS" not in r.to_text()
        assert any("detector de objetos estava desligado" in a for a in r.warnings)
        assert r.detection is not None            # as placas continuam medidas

    def test_fallback_no_meio_da_sessao_nao_e_medido(self, tmp_path):
        db = tmp_path / "fallback.db"
        r = evaluate_session(db, _sessao_com_quadro(db, fallback=True), _gt_um_veiculo(),
                             n_boot=10)
        assert r.object_detection is None
        assert any("fallback" in a for a in r.warnings)

    def test_objeto_nao_anotado_nao_vira_falso_positivo(self, tmp_path):
        db = tmp_path / "sem_objetos.db"
        r = evaluate_session(db, _sessao_com_quadro(db, objeto=True),
                             _gt_um_veiculo(objetos=False), n_boot=10)
        assert r.object_detection is None
        assert any("nao anota objetos" in a for a in r.warnings)


@pytest.fixture(scope="module")
def video_avaliado(tmp_path_factory):
    """Video sintetico curto, processado e avaliado uma vez para o modulo."""
    from lprts.cli import _evaluate
    from lprts.config import AppConfig

    pasta = tmp_path_factory.mktemp("aval")
    video = pasta / "v.mp4"
    subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "make_synthetic_dataset.py"), "--out", str(video),
         "--frames", "240", "--vehicles", "3", "--pedestrians", "1", "--seed", "4"],
        check=True, capture_output=True,
    )
    cfg = AppConfig()
    cfg.source.uri = str(video)
    cfg.source.resize_width = None
    cfg.detector.detect_vehicles = False
    cfg.detector.plate_detector = "classic"
    cfg.ocr.engine = "template"
    cfg.ocr.min_plate_width = 55
    cfg.ocr.ocr_every_n = 1
    cfg.ocr.max_reads_published = 0
    cfg.fusion.max_reads_per_track = 0
    cfg.analytics.save_crops = False
    cfg.analytics.db_path = str(pasta / "a.db")
    gt = video.with_name("v_gt.csv")
    report = _evaluate(cfg, gt, pasta / "r.json", quiet=True)
    return report, pasta, load_ground_truth(gt)


class TestAvaliacaoPontaAPonta:
    def test_veiculos_avaliados_sao_os_anotados(self, video_avaliado):
        report, _, gt = video_avaliado
        elegiveis = [k for k, v in gt.vehicles().items() if not v.all_difficult]
        assert report.vehicle_level["n_vehicles"] == len(elegiveis)
        assert len(report.vehicles) == len(gt.vehicles())

    def test_metricas_sao_proporcoes_validas_com_ic(self, video_avaliado):
        report, _, _ = video_avaliado
        v = report.vehicle_level
        for chave in ("recall", "precision"):
            est, (lo, hi) = v[chave], v[f"{chave}_ci"]
            assert 0.0 <= lo <= est <= hi <= 1.0

    def test_cada_evento_e_rotulado_pela_posicao(self, video_avaliado):
        report, _, _ = video_avaliado
        assert report.events
        for e in report.events:
            assert e["correct"] in (0, 1)
            if e["correct"]:
                assert e["plate_text"] == e["gt_plate"] and e["gt_vehicle"]

    def test_duplicatas_contadas(self, video_avaliado):
        report, _, _ = video_avaliado
        esperado = sum(max(0, r["n_events"] - 1)
                       for r in report.vehicles if not r["excluded_difficult"])
        assert report.vehicle_level["n_duplicate_events"] == esperado

    def test_comparacao_da_fusao_na_mesma_populacao(self, video_avaliado):
        report, _, _ = video_avaliado
        f = report.fusion_comparison
        com_leitura = [r for r in report.vehicles
                       if r["n_reads_text"] > 0 and not r["excluded_difficult"]]
        assert f["n"] == len(com_leitura)
        assert 0.0 <= f["fused"] <= 1.0 and "mcnemar_p" in f["fused_vs_best"]

    def test_tentativas_sem_texto_entram_no_denominador(self, video_avaliado):
        report, _, _ = video_avaliado
        m, r = report.ocr_frame_level, report.frame_level
        assert m.n == r["n_attempts"]
        assert 0.0 <= r["rejection_rate"] <= 1.0

    def test_tabelas_para_modelagem(self, video_avaliado):
        report, pasta, _ = video_avaliado
        with (pasta / "r_leituras.csv").open() as fh:
            linhas = list(csv.DictReader(fh))
        assert len(linhas) == len(report.reads)
        assert {"correct", "plate_w", "sharpness", "confidence", "n_corrections"} <= set(linhas[0])
        assert (pasta / "r_veiculos.csv").exists() and (pasta / "r_eventos.csv").exists()

    def test_reavaliacao_a_partir_do_banco_e_identica(self, video_avaliado):
        """Avaliar de novo a mesma sessao (eval-db) reproduz o relatorio."""
        from lprts.eval.session import evaluate_session

        report, pasta, gt = video_avaliado
        de_novo = evaluate_session(pasta / "a.db", report.session_id, gt)
        assert de_novo.vehicle_level == report.vehicle_level
        assert de_novo.fusion_comparison == report.fusion_comparison

    def test_desempenho_vem_da_telemetria(self, video_avaliado):
        report, _, _ = video_avaliado
        p = report.performance
        assert p["n_frames"] == 240 and p["dropped_frames"] == 0
        assert p["latency_p95_ms"] >= p["service_p50_ms"] > 0


class TestSessaoParalela:
    """A acuracia de uma sessao paralela e a do serial; o tempo por quadro, nao."""

    @pytest.mark.parametrize("realtime", [False, True])
    def test_aviso_de_latencia_offline_em_paralelo(self, tmp_path, realtime):
        db = tmp_path / "paralela.db"
        with ResultStore(db, async_writes=False) as store:
            store.start_session("teste", source_fps=30, realtime=realtime)
            sid = store.session_id
            store.finish_session("fim_da_fonte", stats={"pipeline_workers": 4})
        r = evaluate_session(db, sid, _gt_um_veiculo(), n_boot=10)
        avisou = any("workers em paralelo" in a for a in r.warnings)
        assert avisou is (not realtime)


# Um quadro com quatro placas: a do veiculo 1 (100 px, anotada), a do veiculo 2 (30 px,
# anotada: longe demais), uma de fundo de 30 px sem anotacao e uma de 100 px sem anotacao.
LONGE = (300.0, 0.0, 330.0, 10.0)
FUNDO = (500.0, 0.0, 530.0, 10.0)
ESQUECIDA = (700.0, 0.0, 800.0, 20.0)


def _gt_perto_e_longe() -> GroundTruth:
    return GroundTruth(annotations=[
        GTAnnotation(0, PLACA, "ABC1D23", track_id=1),
        GTAnnotation(0, LONGE, "XYZ9876", track_id=2),
    ], fps=30.0)


def _sessao_com_fundo(db):
    """O sistema detecta, le e publica as quatro placas; so a do veiculo 1 com o texto certo."""
    placas = [(1, PLACA, "ABC1D23"), (2, LONGE, "XYZ9B76"), (3, FUNDO, "AAA0000"),
              (4, ESQUECIDA, "BBB1111")]
    with ResultStore(db, async_writes=False) as store:
        store.start_session("teste", source_fps=30,
                            config={"analytics": {"store_detections": True}})
        store.add_frame_telemetry(FrameResult(
            frame=Frame(0, np.zeros((20, 20, 3), dtype=np.uint8), 0), detection_ran=True,
        ), 30)
        for tid, caixa, texto in placas:
            store.add_detection(0, 0.0, "plate", Detection(caixa, .9, "plate", 0))
            store.add_read(tid, PlateRead(text=texto, confidence=.9, bbox=caixa, frame_index=0))
            store.add_event(TrackState(track_id=tid, bbox=caixa, score=.9, plate_text=texto,
                                       first_seen=0.0), 0, 0.0)
        sid = store.session_id
        store.finish_session("fim_da_fonte")
    return sid


class TestEscopoPorLargura:
    """Placa longe demais para ler: fora da avaliacao, anotada ou nao (como o tamanho minimo do KITTI)."""

    def test_sem_escopo_tudo_conta(self, tmp_path):
        db = tmp_path / "tudo.db"
        r = evaluate_session(db, _sessao_com_fundo(db), _gt_perto_e_longe(), n_boot=10)
        v = r.vehicle_level
        assert v["n_vehicles"] == 2 and v["n_excluded_difficult"] == 0
        assert v["n_events"] == 4 and v["n_events_unmatched"] == 2
        assert v["precision"] == pytest.approx(1 / 4) and v["recall"] == pytest.approx(1 / 2)
        assert (r.detection.tp, r.detection.fp, r.detection.fn) == (2, 2, 0)
        assert r.frame_level["n_unmatched_reads"] == 2
        # Sem a opcao o relatorio nao muda: nenhuma chave nem coluna nova.
        assert "scope_min_width_px" not in v and "out_of_scope" not in r.events[0]
        assert not any("Escopo" in a for a in r.warnings)

    def test_com_escopo_o_que_e_estreito_sai_e_o_resto_continua_contando(self, tmp_path):
        db = tmp_path / "escopo.db"
        r = evaluate_session(db, _sessao_com_fundo(db), _gt_perto_e_longe(), n_boot=10,
                             scope_min_width=50)
        v = r.vehicle_level
        # O veiculo 2 (30 px) vira difficult: sai da revocacao, e a leitura errada dele
        # sai da precisao.
        assert v["n_vehicles"] == 1 and v["n_excluded_difficult"] == 1
        assert v["recall"] == pytest.approx(1.0)
        # A placa de fundo (30 px, sem anotacao) e ignorada; a de 100 px sem anotacao
        # continua erro: o gabarito tem de estar completo para o que da para ler.
        assert v["n_events"] == 2 and v["n_events_unmatched"] == 1
        assert v["n_events_out_of_scope"] == 1 and v["precision"] == pytest.approx(1 / 2)
        assert (r.detection.tp, r.detection.fp, r.detection.fn) == (1, 1, 0)
        assert r.frame_level["n_unmatched_reads"] == 1
        assert r.frame_level["n_reads_out_of_scope"] == 1
        assert r.coverage["detections_out_of_scope"] == 1
        assert r.coverage["gt_boxes_below_scope"] == 1
        fora = {e["plate_text"]: e["out_of_scope"] for e in r.events}
        assert fora == {"ABC1D23": 0, "XYZ9B76": 0, "AAA0000": 1, "BBB1111": 0}
        assert any("Escopo" in a and "50 px" in a for a in r.warnings)

    def test_escopo_nao_altera_o_ground_truth_recebido(self, tmp_path):
        db = tmp_path / "copia.db"
        gt = _gt_perto_e_longe()
        evaluate_session(db, _sessao_com_fundo(db), gt, n_boot=10, scope_min_width=50)
        assert [a.difficult for a in gt.annotations] == [False, False]

    def test_escopo_negativo_e_recusado(self, tmp_path):
        db = tmp_path / "negativo.db"
        sid = _sessao_com_fundo(db)
        with pytest.raises(ValueError, match="scope_min_width"):
            evaluate_session(db, sid, _gt_perto_e_longe(), n_boot=10, scope_min_width=-1)

    def test_eval_db_aceita_a_opcao(self, tmp_path):
        from typer.testing import CliRunner

        from lprts.cli import app
        from lprts.eval.groundtruth import save_ground_truth

        db = tmp_path / "cli.db"
        _sessao_com_fundo(db)
        gt = save_ground_truth(_gt_perto_e_longe(), tmp_path / "gt.csv")
        saida = tmp_path / "r.json"
        res = CliRunner().invoke(app, ["eval-db", "--db", str(db), "--gt", str(gt),
                                       "--out", str(saida), "--scope-min-width", "50"])
        assert res.exit_code == 0, res.output
        import json
        dados = json.loads(saida.read_text(encoding="utf-8"))
        assert dados["vehicle_level"]["scope_min_width_px"] == 50
        assert dados["vehicle_level"]["n_events_out_of_scope"] == 1
