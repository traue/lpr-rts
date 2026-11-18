"""Regressoes da auditoria de integridade dos dados (2026-10).

Cada teste reproduz um defeito que corrompia os dados gravados para analise e
que a suite anterior nao cobria. O docstring de cada um diz o que acontecia.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from lprts.analytics.store import ResultStore, SchemaVersionError
from lprts.capture.ringbuffer import DropOldestQueue
from lprts.config import AppConfig, ConfigError
from lprts.core.pipeline import LPRPipeline
from lprts.core.types import Detection, Frame, PlateLayout, PlateRead
from lprts.detect.base import Detector
from lprts.ocr import plates_br
from lprts.ocr.base import OCREngine
from lprts.track.bytetrack import ByteTracker

IMG = np.zeros((720, 1280, 3), dtype=np.uint8)


class CarroFalso(Detector):
    """Carro andando ``vel`` px por quadro; o teste informa o quadro corrente."""

    name = "carro_falso"

    def __init__(self, vel: float = 10.0, largura: float = 200.0) -> None:
        self.vel, self.largura, self.frame, self.chamadas = vel, largura, 0, 0

    def detect(self, image):
        self.chamadas += 1
        x = 50 + self.vel * self.frame
        return [Detection((x, 300.0, x + self.largura, 450.0), 0.9, "car", 2)]


class PlacaFalsa(Detector):
    """Uma placa parada; caixa em coordenadas da imagem processada."""

    name = "placa_falsa"

    def __init__(self, caixa=(100.0, 50.0, 220.0, 90.0)) -> None:
        self.caixa = caixa

    def detect(self, image):
        return [Detection(self.caixa, 0.9, "plate", 0)]


class OCRRoteiro(OCREngine):
    """Devolve textos de um roteiro, em ordem (o ultimo se repete)."""

    name = "roteiro"

    def __init__(self, textos: list[str], atraso: float = 0.0) -> None:
        self.textos, self.i, self.atraso = textos, 0, atraso

    def read(self, plate_crop):
        if self.atraso:
            time.sleep(self.atraso)
        texto = self.textos[min(self.i, len(self.textos) - 1)]
        self.i += 1
        c = plates_br.correct(texto)
        return PlateRead(text=c.text, confidence=0.9, layout=c.layout,
                         valid_syntax=c.valid, engine=self.name, raw_text=texto,
                         n_corrections=c.corrections)


def _cfg(**kw) -> AppConfig:
    cfg = AppConfig()
    cfg.analytics.save_crops = False
    cfg.ocr.min_plate_width = 10
    cfg.ocr.ocr_every_n = 1
    for k, v in kw.items():
        cfg.override(**{k: v})
    return cfg


class TestBaseDeTempo:
    def _rodar(self, relogio: list[float]) -> tuple[float, float]:
        cfg = _cfg()
        cfg.analytics.calibration = {"meters_per_pixel": 0.05}
        det = CarroFalso(vel=10.0)
        pipe = LPRPipeline(cfg, det, None, None, None).start()
        for i, ts in enumerate(relogio):
            det.frame = i
            pipe.process_frame(Frame(i, IMG, timestamp=ts, source_pts=i / 30.0))
        t = pipe.tracker.tracks[0]
        pipe.stop()
        return t.speed_kmh, t.dwell_s

    def test_velocidade_e_permanencia_nao_dependem_da_maquina(self):
        """Antes: o mesmo veiculo saia a 219 km/h numa maquina rapida e 17 km/h numa lenta.

        A base fisica era time.monotonic() (relogio de processamento); agora e o
        tempo de midia do video.
        """
        rapida = self._rodar([i * 0.003 for i in range(40)])     # ~333 quadros/s
        lenta = self._rodar([i * 0.100 for i in range(40)])      # 10 quadros/s
        assert rapida[0] == pytest.approx(lenta[0])
        assert rapida[1] == pytest.approx(lenta[1])
        # 10 px/quadro x 0,05 m/px x 30 quadros/s = 15 m/s = 54 km/h
        assert rapida[0] == pytest.approx(54.0, rel=0.05)
        assert rapida[1] == pytest.approx(39 / 30.0, abs=1e-6)


class TestPersistenciaDasTrilhas:
    def test_objetos_sem_placa_que_saem_de_cena_sao_gravados(self, tmp_path):
        """Antes: so as trilhas vivas no ultimo quadro chegavam ao banco.

        6 pedestres atravessavam a cena; a tabela tracks ficava com 0 ou 1 linha
        e as visoes de composicao do trafego saiam vazias.
        """
        entradas = [0, 40, 80, 120, 160, 250]

        class Pedestres(Detector):
            name = "pedestres"
            frame = 0

            def detect(self, image):
                dets = []
                for k, t0 in enumerate(entradas):
                    dt = self.frame - t0
                    if 0 <= dt < 60:
                        x = 50 + 3 * dt
                        y = 100 + 80 * k
                        dets.append(Detection((x, y, x + 40, y + 70), 0.9, "person", 0))
                return dets

        det = Pedestres()
        db = tmp_path / "c.db"
        with ResultStore(db) as store:
            store.start_session("sintetico")
            pipe = LPRPipeline(_cfg(), det, None, None, store).start()
            for i in range(300):
                det.frame = i
                pipe.process_frame(Frame(i, IMG, timestamp=i / 30, source_pts=i / 30))
            pipe.finalize()
            pipe.stop()
        con = sqlite3.connect(db)
        linhas = con.execute(
            "SELECT n_frames, dwell_s, end_reason, object_class FROM tracks"
        ).fetchall()
        assert len(linhas) == 6
        assert all(classe == "pessoa" for *_, classe in linhas)
        # Permanencia em tempo de video: 60 quadros a 30 fps (o ultimo entra no
        # quadro 250 de 300 e fica 50).
        assert sorted(n for n, *_ in linhas) == [50, 60, 60, 60, 60, 60]
        assert all(d == pytest.approx((n - 1) / 30) for n, d, _, _ in linhas)
        assert {r for _, _, r, _ in linhas} == {"lost", "end_of_stream"}
        assert con.execute("SELECT n_tracks FROM v_object_counts").fetchone()[0] == 6

    def test_trilha_publicada_e_gravada_com_o_estado_final(self, tmp_path):
        """Antes: a linha era escrita na publicacao e nunca atualizada (5-9 quadros)."""
        db = tmp_path / "p.db"
        cfg = _cfg(**{"detector.detect_vehicles": False})
        with ResultStore(db) as store:
            store.start_session("s")
            pipe = LPRPipeline(cfg, None, PlacaFalsa(), OCRRoteiro(["ABC1D23"]), store).start()
            for i in range(50):
                pipe.process_frame(Frame(i, IMG, timestamp=0.0, source_pts=i / 30))
            pipe.finalize()
            pipe.stop()
        con = sqlite3.connect(db)
        n_frames, last, publicado, texto, fim = con.execute(
            "SELECT n_frames, last_frame, published, published_text, end_reason FROM tracks"
        ).fetchone()
        assert (n_frames, last, publicado, texto, fim) == (50, 49, 1, "ABC1D23", "end_of_stream")
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


class TestEventosNoFimDoVideo:
    def _rodar(self, tmp_path, textos, quadros):
        db = tmp_path / "f.db"
        cfg = _cfg(**{"detector.detect_vehicles": False, "fusion.min_reads": 3})
        with ResultStore(db) as store:
            store.start_session("s")
            pipe = LPRPipeline(cfg, None, PlacaFalsa(), OCRRoteiro(textos), store).start()
            for i in range(quadros):
                pipe.process_frame(Frame(i, IMG, timestamp=0.0, source_pts=10.0 + i / 30))
            pipe.finalize()
            pipe.stop()
        return sqlite3.connect(db)

    def test_sem_criterio_de_estabilidade_nao_ha_evento(self, tmp_path):
        """Antes: o finalize publicava com 1-2 leituras, ignorando min_reads."""
        # 3 quadros para confirmar a trilha + 2 leituras: abaixo de min_reads=3.
        con = self._rodar(tmp_path, ["ABC1D23"], 4)
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        texto, estavel, publicado = con.execute(
            "SELECT plate_text, plate_stable, published FROM tracks"
        ).fetchone()
        assert (texto, estavel, publicado) == ("ABC1D23", 0, 0)

    def test_texto_provisorio_acompanha_a_fusao(self, tmp_path):
        """Antes: o texto ficava congelado no 1o fragmento ("III", "1", "CI74I")."""
        con = self._rodar(tmp_path, ["III", "ABC1D23", "ABC1D23"], 6)
        (texto,) = con.execute("SELECT plate_text FROM tracks").fetchone()
        assert texto == "ABC1D23"

    def test_evento_tem_o_tempo_de_video_correto(self, tmp_path):
        """Antes: eventos do fim do video eram gravados com video_time_s = 0."""
        con = self._rodar(tmp_path, ["ABC1D23"], 20)
        tempos = [r[0] for r in con.execute("SELECT video_time_s FROM events")]
        assert tempos and all(t >= 10.0 for t in tempos)


class TestDescartes:
    def test_descartes_chegam_ao_banco(self, tmp_path):
        """Antes: frames.dropped_frames era sempre 0 (atribuido depois de gravado)."""

        class FonteComDescarte:
            fps = 30.0
            dropped = 0

            def frames(self):
                for i in range(0, 60, 3):            # 2 de cada 3 quadros descartados
                    self.dropped = i - i // 3
                    yield Frame(i, IMG, timestamp=time.monotonic(), source_pts=i / 30)

        db = tmp_path / "d.db"
        fonte = FonteComDescarte()
        with ResultStore(db) as store:
            store.start_session("s")
            pipe = LPRPipeline(_cfg(), CarroFalso(), None, None, store).start()
            for _ in pipe.run(fonte):
                pass
            m = pipe.metrics.snapshot()
            pipe.stop()
        con = sqlite3.connect(db)
        maximo, pulados = con.execute(
            "SELECT MAX(dropped_frames), SUM(frames_skipped) FROM frames"
        ).fetchone()
        assert maximo == 38 and pulados == 38
        assert m.frames_dropped == 38


class TestRastreador:
    def test_carro_rapido_e_confirmado(self):
        """Antes: IoU >= 0,8 na 1a passada - um carro a ~60 km/h nunca era confirmado.

        No ByteTrack de referencia, match_thresh=0.8 e custo (IoU >= 0,2).
        """
        tracker = ByteTracker()                       # padroes do projeto
        ids = set()
        for f in range(60):
            x = 50 + 30 * f                           # 30 px/quadro, caixa de 200 px
            for t in tracker.update([Detection((x, 300, x + 200, 450), 0.9, "car")], f, f / 30):
                ids.add(t.track_id)
        assert ids == {1}

    def test_quadros_pulados_nao_quebram_a_trilha(self):
        """Com descarte, o Kalman preve o numero real de quadros decorridos."""
        tracker = ByteTracker()
        ids = set()
        for f in range(0, 90, 3):                     # so 1 a cada 3 quadros chega
            x = 50 + 20 * f
            for t in tracker.update([Detection((x, 300, x + 200, 450), 0.9, "car")], f, f / 30):
                ids.add(t.track_id)
        assert len(ids) == 1

    def test_ids_recomecam_a_cada_rastreador(self):
        """Antes: contador global - o mesmo video gerava IDs diferentes no mesmo processo."""
        for _ in range(2):
            t = ByteTracker(min_hits=1)
            assert t.update([Detection((0, 0, 50, 50), 0.9, "car")], 0)[0].track_id == 1

    def test_detect_every_n_nao_reenvia_deteccao_velha(self):
        """Antes: os quadros sem deteccao reenviavam as caixas do quadro anterior.

        A velocidade do Kalman ia a zero e, com n=3, um veiculo virava 20 trilhas.
        """
        det = CarroFalso(vel=12.0)
        pipe = LPRPipeline(_cfg(**{"detector.detect_every_n": 3}), det, None, None, None).start()
        ids = set()
        for i in range(60):
            det.frame = i
            r = pipe.process_frame(Frame(i, IMG, timestamp=0.0, source_pts=i / 30))
            ids.update(t.track_id for t in r.tracks)
        vx = pipe.tracker._tracks[0].kf.velocity[0]
        pipe.stop()
        assert ids == {1}
        assert vx == pytest.approx(12.0, rel=0.1)
        assert det.chamadas == 20                     # 1 deteccao a cada 3 quadros


class TestCoordenadas:
    def test_banco_grava_pixels_da_imagem_original(self, tmp_path):
        """Antes: caixas na escala reduzida e ground truth na original (P=R=0 na avaliacao)."""
        db = tmp_path / "e.db"
        cfg = _cfg(**{"detector.detect_vehicles": False})
        img = np.zeros((360, 640, 3), np.uint8)       # imagem processada = metade da original
        with ResultStore(db) as store:
            store.start_session("s")
            pipe = LPRPipeline(cfg, None, PlacaFalsa((100.0, 50.0, 220.0, 90.0)),
                               OCRRoteiro(["ABC1D23"]), store).start()
            for i in range(10):
                pipe.process_frame(Frame(i, img, timestamp=0.0, source_pts=i / 30, scale=0.5))
            pipe.finalize()
            pipe.stop()
        con = sqlite3.connect(db)
        assert con.execute("SELECT x1, y1, x2, y2 FROM detections LIMIT 1").fetchone() == (
            200.0, 100.0, 440.0, 180.0
        )
        x1, x2, w = con.execute("SELECT bbox_x1, bbox_x2, plate_w FROM reads LIMIT 1").fetchone()
        assert x1 > 150 and x2 > 400 and w > 200


class TestOCRAssincrono:
    def test_leitura_gravada_no_quadro_do_recorte(self, tmp_path):
        """Antes: gravada no quadro em que o resultado chegou (2-3 quadros depois)."""
        db = tmp_path / "a.db"
        cfg = _cfg(**{"detector.detect_vehicles": False, "source.realtime": True})
        with ResultStore(db) as store:
            store.start_session("s", realtime=True)
            pipe = LPRPipeline(cfg, None, PlacaFalsa(), OCRRoteiro(["ABC1D23"], atraso=0.03),
                               store).start()
            for i in range(30):
                pipe.process_frame(Frame(i, IMG, timestamp=time.monotonic(), source_pts=i / 30))
                time.sleep(0.01)
            pipe.finalize()
            pipe.stop()
        con = sqlite3.connect(db)
        linhas = con.execute("SELECT frame_index, collected_frame FROM reads").fetchall()
        assert len(linhas) > 3
        assert all(fi <= cf for fi, cf in linhas)
        # So a leitura colhida no encerramento pode coincidir com o ultimo quadro.
        assert sum(1 for fi, cf in linhas if fi < cf) >= len(linhas) - 1


class TestParaleloEmTempoReal:
    def test_reordenacao_nao_trava_com_descarte(self, tmp_path):
        """Antes: cada quadro descartado deixava um buraco que a reordenacao esperava.

        Com deteccao mais lenta que a fonte, a latencia chegava a ~1,6 s enquanto o
        banco registrava ~90 ms; e descartes eram contados em dobro.
        """
        from lprts.capture.source import VideoSource
        from lprts.core.parallel import ParallelPipeline
        from lprts.detect.classic_plate import ClassicPlateDetector

        caminho = tmp_path / "rt.mp4"
        w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
        for _ in range(90):
            w.write(np.full((360, 640, 3), 95, dtype=np.uint8))
        w.release()

        class Lento(ClassicPlateDetector):
            def detect(self, image):
                time.sleep(0.08)
                return super().detect(image)

        cfg = _cfg(**{"detector.detect_vehicles": False, "source.realtime": True})
        src = VideoSource(caminho, realtime=True).open()
        pipe = ParallelPipeline(cfg, detector_factory=lambda: (None, Lento()), n_workers=2,
                                plate_detector=Lento(), ocr_engine=None).start()
        idades = []
        n = 0
        ultimo = None
        for r in pipe.run(src):
            n += 1
            idades.append(r.timing.latency_ms)
            ultimo = r
        pipe.stop()
        src.close()
        assert n > 0
        assert float(np.percentile(idades, 95)) < 800.0
        # Cada quadro lido da fonte foi processado ou descartado - nunca os dois.
        assert n + ultimo.dropped_frames <= 90


class TestGramatica:
    @pytest.mark.parametrize("lixo", ["1231231", "0000000", "8888888"])
    def test_nao_inventa_placa(self, lixo):
        """Antes (3 correcoes): "1231231" virava "IZB1231" valida - o README dizia o contrario."""
        assert not plates_br.correct(lixo).valid

    def test_correcoes_sao_contadas(self):
        assert plates_br.correct("4BC1D23").corrections == 1

    def test_raw_text_do_gabarito_e_a_decodificacao_livre(self):
        """Antes: raw_text == text em 100% das leituras do OCR por gabaritos.

        A ablacao "quanto a gramatica contribui" era impossivel nesse motor.
        """
        from lprts.ocr.template_ocr import ALPHABET, _decode_free, _decode_with_masks

        scores = [dict.fromkeys(ALPHABET, 0.1) for _ in range(7)]
        for pos, ch in enumerate("0BC1D23"):          # posicao 0 prefere o DIGITO 0
            scores[pos][ch] = 0.9
        scores[0]["O"] = 0.8
        livre, _ = _decode_free(scores)
        restrito, *_ = _decode_with_masks(scores)
        assert livre == "0BC1D23" and restrito == "OBC1D23"

    def test_template_conta_trocas_da_gramatica(self, plate_image):
        from lprts.ocr.template_ocr import TemplateOCR

        r = TemplateOCR().read(plate_image("ABC1D23"))
        trocas = sum(1 for a, b in zip(r.raw_text, r.text, strict=False) if a != b)
        assert r.n_corrections == trocas


class TestConfiguracao:
    def test_chave_desconhecida_e_erro(self):
        """Antes: um erro de digitacao no YAML era ignorado em silencio."""
        with pytest.raises(ConfigError, match="min_reads"):
            AppConfig.from_dict({"fusion": {"min_read": 5}})

    def test_chave_renomeada_explica_a_mudanca(self):
        with pytest.raises(ConfigError, match="match_iou"):
            AppConfig.from_dict({"tracker": {"match_thresh": 0.8}})

    def test_combinacao_de_yamls(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("ocr:\n  engine: template\n  ocr_every_n: 1\n", encoding="utf-8")
        b.write_text("ocr:\n  ocr_every_n: 3\n", encoding="utf-8")
        cfg = AppConfig.load_many([a, b])
        assert cfg.ocr.engine == "template" and cfg.ocr.ocr_every_n == 3


class TestMotorDeOCR:
    def test_motor_explicito_ausente_e_erro(self):
        """Antes: --ocr paddle sem paddle rodava o baseline e o relatorio dizia 'paddle'."""
        import importlib.util

        from lprts.ocr.engines import EngineUnavailableError, build_ocr_engine

        if importlib.util.find_spec("pytesseract") is not None:
            pytest.skip("pytesseract instalado")
        with pytest.raises(EngineUnavailableError):
            build_ocr_engine("tesseract")


class TestBanco:
    def test_banco_de_versao_anterior_e_recusado(self, tmp_path):
        db = tmp_path / "velho.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        con.commit()
        con.close()
        with pytest.raises(SchemaVersionError):
            ResultStore(db).open()

    def test_fila_cheia_descarta_so_telemetria_e_conta(self, tmp_path, monkeypatch):
        """Antes: qualquer linha, inclusive eventos, era descartada em silencio."""
        from lprts.core.types import FrameResult, StageTiming, TrackState

        store = ResultStore(tmp_path / "q.db", batch_size=1, queue_size=2)
        original = store._flush

        def flush_lento(batch):
            time.sleep(0.02)
            original(batch)

        monkeypatch.setattr(store, "_flush", flush_lento)
        store.open()
        store.start_session("s")
        for i in range(40):
            store.add_frame_telemetry(
                FrameResult(frame=Frame(i, IMG, 0.0, i / 30), timing=StageTiming()), 30.0
            )
        trilha = TrackState(track_id=1, bbox=(0, 0, 10, 10), score=0.9, plate_text="ABC1D23",
                            plate_layout=PlateLayout.MERCOSUL)
        for i in range(5):
            store.add_event(trilha, i, i / 30)
        store.finish_session()
        n_eventos = store.query("SELECT COUNT(*) AS n FROM events")[0]["n"]
        perdidas = store.rows_lost
        stats = store.query("SELECT stats_json FROM sessions")[0]["stats_json"]
        store.close()
        assert n_eventos == 5
        assert perdidas.get("frames", 0) > 0 and "events" not in perdidas
        assert '"frames"' in stats                   # a perda fica registrada na sessao


def test_fila_drop_oldest_conta_descartes_sob_concorrencia():
    q = DropOldestQueue(maxsize=2)
    produzidos = 1000

    def produzir():
        for i in range(produzidos):
            q.put(i)
        q.close()

    t = threading.Thread(target=produzir)
    t.start()
    consumidos = 0
    while True:
        v = q.get(timeout=0.5)
        if v is None and q.closed:
            break
        if v is not None:
            consumidos += 1
    t.join()
    assert consumidos + q.dropped == produzidos


def test_flags_da_cli_nao_sobrescrevem_o_yaml_quando_ausentes():
    """Antes: --realtime/--offline tinha padrao False e anulava `realtime: true` do perfil.

    O mesmo valia para --vehicle-detector e --ui: a configuracao gravada na sessao
    nao era a do YAML que o pesquisador passou.
    """
    import inspect

    from lprts import cli

    for comando in (cli.run, cli.eval_cmd, cli.diagnose, cli.bench):
        params = inspect.signature(comando).parameters
        for nome in ("realtime", "vehicle_detector", "ui"):
            if nome in params:
                assert params[nome].default.default is None, f"{comando.__name__}.{nome}"


class TestBancosAntigos:
    def _velho(self, caminho):
        con = sqlite3.connect(caminho)
        con.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, "
                    "started_at TEXT, source_uri TEXT)")
        con.execute("CREATE TABLE events (session_id TEXT, track_id INTEGER, plate_text TEXT)")
        con.execute("CREATE TABLE reads (session_id TEXT, track_id INTEGER, text TEXT)")
        con.commit()
        con.close()

    def test_arquivamento_move_sem_apagar(self, tmp_path):
        """O ./start.sh arquiva bancos antigos em vez de falhar na abertura."""
        from lprts.analytics.store import archive_legacy_databases

        self._velho(tmp_path / "lprts.db")
        (tmp_path / "lprts.db-wal").write_bytes(b"x")
        with ResultStore(tmp_path / "novo.db") as store:      # esquema atual: fica
            store.start_session("s")
        movidos = archive_legacy_databases(tmp_path)
        assert [o.name for o, _ in movidos] == ["lprts.db"]
        destino = movidos[0][1]
        assert destino.exists() and Path(str(destino) + "-wal").exists()
        assert not (tmp_path / "lprts.db").exists()
        assert (tmp_path / "novo.db").exists()
        ResultStore(tmp_path / "lprts.db").open().close()    # o caminho padrao volta a funcionar

    def test_erro_esperado_nao_vira_traceback(self):
        """Banco antigo, motor ausente etc. sao mensagens, nao 'falha no pipeline'."""
        pytest.importorskip("PySide6")
        from lprts.ui.worker import _erro_de_configuracao

        assert _erro_de_configuracao(SchemaVersionError("x"), iniciando=False)
        assert _erro_de_configuracao(ValueError("calibracao"), iniciando=True)
        assert not _erro_de_configuracao(ValueError("bug"), iniciando=False)
        assert not _erro_de_configuracao(KeyError("bug"), iniciando=True)


class TestDetectorCoreML:
    """Antes: com .mlpackage e dispositivo "mps", TODA deteccao falhava.

    O ultralytics usava "mps" no pos-processamento e o NMS convertia para
    float64, que o MPS nao suporta ("Cannot convert a MPS Tensor to float64").
    O CoreML escolhe as proprias unidades; o pos-processamento vai para a CPU.
    """

    class _Modelo:
        def __init__(self):
            self.dispositivos: list[str] = []

        def predict(self, image, **kw):
            self.dispositivos.append(kw["device"])
            return []

    def _detector(self, tmp_path, nome, monkeypatch):
        from lprts.detect.backends import DeviceInfo
        from lprts.detect.yolo_detector import YOLODetector

        monkeypatch.setattr(YOLODetector, "_load", lambda self: None)
        mps = DeviceInfo(kind="mps", torch_device="mps", onnx_providers=(), description="teste")
        # O caminho do ultralytics (referencia); o padrao para .mlpackage e o CoreML direto.
        det = YOLODetector(tmp_path / nome, device=mps, runtime="ultralytics")
        det._model = self._Modelo()
        return det

    @pytest.mark.parametrize("nome,esperado", [
        ("yolo11n.mlpackage", "cpu"), ("yolo11n.mlmodel", "cpu"), ("yolo11n.pt", "mps"),
    ])
    def test_dispositivo_do_pos_processamento(self, tmp_path, monkeypatch, nome, esperado):
        det = self._detector(tmp_path, nome, monkeypatch)
        assert det.detect(IMG) == []
        assert det._model.dispositivos == [esperado]

    def test_describe_registra_coreml(self, tmp_path, monkeypatch):
        det = self._detector(tmp_path, "yolo11n.mlpackage", monkeypatch)
        info = det.describe()
        assert info["runtime"] == "ultralytics/coreml" and info["device"] == "coreml"

    def test_nms_embutido_registra_parametros_efetivos(self, tmp_path, monkeypatch):
        """Antes: .mlpackage com NMS embutido ignorava iou/imgsz e o banco gravava os pedidos."""
        import lprts.detect.yolo_detector as yd

        monkeypatch.setattr(yd, "coreml_export_info", lambda w: {
            "imgsz": [640, 640], "nms_embedded": True, "iou_embedded": 0.7,
            "conf_embedded": 0.25, "precision": "fp16"})
        det = self._detector(tmp_path, "placa.mlpackage", monkeypatch)
        det.imgsz, det.iou = 320, 0.45
        det._check_coreml_export()
        info = det.describe()
        assert info["imgsz"] == 640 and info["iou"] == 0.7
        assert info["coreml_export"]["nms_embedded"] is True


class TestPreferenciaCoreML:
    """Com device auto/coreml, um .pt rodava em MPS mesmo com o .mlpackage ao lado."""

    def _pesos(self, tmp_path, monkeypatch, info):
        import lprts.detect.yolo_detector as yd

        (tmp_path / "yolo11n.pt").write_bytes(b"x")
        (tmp_path / "yolo11n.mlpackage").mkdir()
        monkeypatch.setattr(yd, "coreml_export_info", lambda w: info)
        return tmp_path / "yolo11n.pt"

    @staticmethod
    def _disp(kind):
        from lprts.detect.backends import DeviceInfo

        return DeviceInfo(kind=kind, torch_device="mps", onnx_providers=(), description="t")

    COMPATIVEL = {"imgsz": [640, 640], "nms_embedded": False, "iou_embedded": None,
                  "conf_embedded": None, "precision": "fp16"}

    def test_usa_mlpackage_compativel(self, tmp_path, monkeypatch):
        from lprts.factory import coreml_weights

        pt = self._pesos(tmp_path, monkeypatch, self.COMPATIVEL)
        assert coreml_weights(pt, self._disp("coreml"), 640).suffix == ".mlpackage"

    @pytest.mark.parametrize("kind,imgsz,mudanca", [
        ("mps", 640, {}),                         # PyTorch pedido explicitamente
        ("coreml", 320, {}),                      # entrada fixa diferente da configurada
        ("coreml", 640, {"nms_embedded": True}),  # ignoraria o iou configurado
    ])
    def test_mantem_pt_quando_incompativel(self, tmp_path, monkeypatch, kind, imgsz, mudanca):
        from lprts.factory import coreml_weights

        pt = self._pesos(tmp_path, monkeypatch, {**self.COMPATIVEL, **mudanca})
        assert coreml_weights(pt, self._disp(kind), imgsz) == pt

    def test_sem_mlpackage_mantem_pt(self, tmp_path):
        from lprts.factory import coreml_weights

        (tmp_path / "yolo11n.pt").write_bytes(b"x")
        pt = tmp_path / "yolo11n.pt"
        assert coreml_weights(pt, self._disp("coreml"), 640) == pt


class TestCoreMLDireto:
    """O .mlpackage roda pelo coremltools, com pre/pos-processamento proprios.

    Pelo ultralytics, o preparo custava 3-4x a inferencia no Neural Engine. Em
    quadros reais (2 videos 4K), as caixas dos dois caminhos coincidem: 1923 de
    1923 objetos e 911 de 912 placas com IoU 1,000.
    """

    def _det(self, tmp_path, monkeypatch, nome="m.mlpackage", runtime="auto", **kw):
        from lprts.detect.yolo_detector import YOLODetector

        monkeypatch.setattr(YOLODetector, "_load", lambda self: None)
        return YOLODetector(tmp_path / nome, runtime=runtime, **kw)

    @pytest.mark.parametrize("nome,runtime,esperado", [
        ("m.mlpackage", "auto", "coreml"), ("m.onnx", "auto", "onnx"), ("m.pt", "auto", "ultralytics"),
        ("m.mlpackage", "ultralytics", "ultralytics"), ("m.pt", "coreml", "ultralytics"),
    ])
    def test_escolha_do_runtime(self, tmp_path, monkeypatch, nome, runtime, esperado):
        assert self._det(tmp_path, monkeypatch, nome, runtime).runtime == esperado

    def test_runtime_desconhecido_e_erro(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="Runtime de deteccao desconhecido"):
            self._det(tmp_path, monkeypatch, runtime="tensorrt")

    def test_decodificacao_igual_ao_ultralytics(self, tmp_path, monkeypatch):
        """Classe = maior score entre TODAS; filtro de classe depois; NMS por classe."""
        det = self._det(tmp_path, monkeypatch, conf=0.3, iou=0.45,
                        classes={2: "car", 0: "person"})
        raw = np.zeros((1, 4 + 80, 2100), dtype=np.float32)   # (1, 4+nc, ancoras)
        caixas = [(100, 300, 40, 20), (102, 300, 40, 20), (300, 400, 50, 50),
                  (500, 500, 10, 10), (50, 250, 20, 20)]
        for i, (cx, cy, w, h) in enumerate(caixas):
            raw[0, :4, i] = (cx, cy, w, h)
        raw[0, 4 + 2, 0], raw[0, 4 + 2, 1] = 0.9, 0.8   # dois "car" sobrepostos: NMS
        raw[0, 4 + 0, 2] = 0.7                           # "person"
        raw[0, 4 + 16, 3], raw[0, 4 + 2, 3] = 0.6, 0.5  # melhor e "dog": sai, nao vira "car"
        raw[0, 4 + 2, 4] = 0.2                           # abaixo do limiar
        # letterbox de 1280x640 para 640x640: escala 0,5 e faixa de 160 px em cima.
        dets = det._decode(raw, 0.5, (0, 160), (640, 1280))
        assert [(d.label, round(d.score, 2)) for d in dets] == [("car", 0.9), ("person", 0.7)]
        assert dets[0].bbox == pytest.approx((160.0, 260.0, 240.0, 300.0))

    def test_describe_registra_coremltools(self, tmp_path, monkeypatch):
        info = self._det(tmp_path, monkeypatch).describe()
        assert info["runtime"] == "coremltools" and info["device"] == "coreml"
