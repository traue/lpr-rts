"""Paralelismo: reordenacao, politica de filas e equivalencia com o serial."""

from __future__ import annotations

import random
import threading

import cv2
import numpy as np
import pytest

from lprts.capture.ringbuffer import BlockingQueue, DropOldestQueue
from lprts.capture.source import VideoSource
from lprts.config import AppConfig
from lprts.core.parallel import (
    ParallelPipeline,
    ReorderBuffer,
    choose_worker_count,
    resolve_workers,
    tune_opencv_threads,
)
from lprts.core.pipeline import LPRPipeline
from lprts.core.types import Detection
from lprts.detect.classic_plate import ClassicPlateDetector
from lprts.ocr.engines import build_ocr_engine


@pytest.fixture
def video_multi(tmp_path, plate_image):
    """Video com uma placa que cresce - suficiente para produzir leituras."""
    caminho = tmp_path / "p.mp4"
    w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    assert w.isOpened()
    for i in range(80):
        cena = np.full((360, 640, 3), 95, dtype=np.uint8)
        largura = 110 + i * 3
        altura = int(largura * 0.33)
        x1 = 300 - largura // 2
        cena[150 : 150 + altura, x1 : x1 + largura] = cv2.resize(
            plate_image("ABC1D23"), (largura, altura)
        )
        w.write(cena)
    w.release()
    return caminho


class VeiculoFixo:
    """Detector de objetos falso: um carro que contem a placa do video em todo quadro."""

    name = "veiculo_fixo"

    def detect(self, image):
        return [Detection((60.0, 80.0, 580.0, 330.0), 0.9, "car", 2)]

    def warmup(self):
        pass

    def close(self):
        pass

    def describe(self):
        return {"name": self.name, "runtime": "teste", "device": "cpu"}


class TestPreparoNosWorkers:
    """Offline, a leitura so decodifica; reduzir e girar rodam em qualquer thread."""

    def _quadros(self, caminho, crus: bool, rotacao=None):
        src = VideoSource(str(caminho), resize_width=320, keep_original=True).open()
        if rotacao is not None:
            src._rotacao = rotacao          # simula o video de celular na vertical
        try:
            if crus:
                return [src.prepare(c) for c in src.raw_frames()]
            return list(src.frames())
        finally:
            src.close()

    @pytest.mark.parametrize("rotacao", [None, cv2.ROTATE_90_CLOCKWISE])
    def test_preparo_adiado_e_o_mesmo_quadro(self, video_multi, rotacao):
        diretos = self._quadros(video_multi, crus=False, rotacao=rotacao)
        adiados = self._quadros(video_multi, crus=True, rotacao=rotacao)
        assert len(diretos) == len(adiados) == 80
        for a, b in zip(diretos, adiados, strict=True):
            assert (a.index, a.source_pts, a.scale, a.orig_width, a.orig_height) == (
                b.index, b.source_pts, b.scale, b.orig_width, b.orig_height)
            assert a.full_rotation == b.full_rotation
            assert np.array_equal(a.image, b.image)
            assert np.array_equal(a.full_image, b.full_image)

    def test_quadros_crus_so_offline(self, video_multi):
        src = VideoSource(str(video_multi), realtime=True)
        with pytest.raises(RuntimeError, match="offline"):
            next(iter(src.raw_frames()))


class TestReorderBuffer:
    def test_restaura_a_ordem(self):
        rb = ReorderBuffer()
        indices = list(range(30))
        random.Random(7).shuffle(indices)
        saida = []
        for i in indices:
            saida.extend(rb.push(i, i))
        saida.extend(rb.flush())
        assert saida == list(range(30))

    def test_ja_ordenado_passa_direto(self):
        rb = ReorderBuffer()
        assert [rb.push(i, i) for i in range(5)] == [[0], [1], [2], [3], [4]]

    def test_buraco_nao_trava_com_limite_baixo(self):
        """Modo tempo real: avancar vale mais que esperar um worker travado."""
        rb = ReorderBuffer(max_pending=3)
        saida = []
        for i in range(1, 8):        # o indice 0 nunca chega
            saida.extend(rb.push(i, i))
        assert saida and rb.skipped >= 1

    def test_limite_alto_segura_a_sequencia(self):
        """Modo offline: nada pode ser pulado enquanto couber na espera."""
        rb = ReorderBuffer(max_pending=10_000)
        saida = []
        for i in range(1, 8):
            saida.extend(rb.push(i, i))
        assert saida == [] and rb.skipped == 0

    def test_chegada_atrasada_e_ignorada(self):
        rb = ReorderBuffer()
        rb.push(0, "a")
        rb.push(1, "b")
        assert rb.push(0, "duplicado") == []


class TestPoliticaDeFilas:
    def test_blocking_nao_descarta(self):
        """Regressao: em offline, descartar quadros invalidaria a avaliacao."""
        q = BlockingQueue(maxsize=2)
        recebidos = []

        def consumir():
            while len(recebidos) < 10:
                v = q.get(timeout=0.3)
                if v is not None:
                    recebidos.append(v)

        t = threading.Thread(target=consumir)
        t.start()
        for i in range(10):
            q.put(i)
        t.join(timeout=5)
        assert recebidos == list(range(10))
        assert q.dropped == 0

    def test_drop_oldest_descarta(self):
        q = DropOldestQueue(maxsize=2)
        for i in range(10):
            q.put(i)
        assert q.dropped == 8

    def test_interface_identica(self):
        """As duas filas sao intercambiaveis pelo pipeline."""
        for q in (BlockingQueue(maxsize=4), DropOldestQueue(maxsize=4)):
            q.put(1)
            assert q.get(timeout=0.1) == 1
            assert isinstance(q.dropped, int)
            q.close()
            assert q.closed


class TestConfiguracao:
    def test_contagem_de_workers(self):
        assert choose_worker_count("latencia") <= choose_worker_count("vazao")
        assert choose_worker_count("auto") >= 1

    @pytest.mark.parametrize(("nucleos", "padrao", "com_ocr"), [(15, 4, 7), (8, 4, 4), (4, 3, 2), (1, 1, 1)])
    def test_automatico_sobe_com_ocr_antecipado(self, monkeypatch, nucleos, padrao, com_ocr):
        monkeypatch.setattr("os.cpu_count", lambda: nucleos)
        assert choose_worker_count("auto") == padrao
        assert choose_worker_count("auto", ocr_antecipado=True) == com_ocr

    def test_workers_efetivos(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 15)
        cfg = AppConfig()
        assert cfg.detector.workers == -1
        assert resolve_workers(cfg) == 4                 # offline, perfil padrao
        cfg.ocr.ocr_every_n, cfg.ocr.max_reads_published = 1, 0
        assert resolve_workers(cfg) == 7                 # le toda placa: OCR nos workers
        cfg.source.realtime = True
        assert resolve_workers(cfg) == 0                 # tempo real: serial
        cfg.detector.workers = 3
        assert resolve_workers(cfg) == 3                 # explicito vale sempre
        with pytest.raises(ValueError, match="workers"):
            AppConfig.from_dict({"detector": {"workers": -2}})

    def test_fabrica_segue_a_configuracao(self, monkeypatch):
        from lprts import factory

        monkeypatch.setattr(factory, "build_pipeline", lambda cfg, store=None: ("serial", store))
        monkeypatch.setattr(factory, "build_parallel_pipeline",
                            lambda cfg, n, store=None: ("paralelo", n))
        cfg = AppConfig()
        cfg.detector.workers = 0
        assert factory.build_configured_pipeline(cfg) == ("serial", None)
        cfg.detector.workers = 2
        assert factory.build_configured_pipeline(cfg) == ("paralelo", 2)

    def test_threads_do_opencv_repartidas(self):
        """Evita oversubscription: N workers x todos os nucleos cada."""
        anterior = cv2.getNumThreads()
        try:
            assert tune_opencv_threads(4) >= 1
            assert tune_opencv_threads(1) >= tune_opencv_threads(8)
        finally:
            cv2.setNumThreads(anterior)


class TestEquivalencia:
    def _config(self, tmp_path) -> AppConfig:
        cfg = AppConfig()
        cfg.detector.detect_vehicles = False
        cfg.source.realtime = False
        cfg.ocr.min_plate_width = 55
        cfg.ocr.ocr_every_n = 1
        cfg.ocr.max_ocr_per_frame = 4
        cfg.analytics.save_crops = False
        return cfg

    def _rodar(self, pipe, caminho):
        with VideoSource(caminho) as src:
            n = 0
            placas = set()
            for r in pipe.run(src):
                n += 1
                for t in r.tracks:
                    if t.plate_stable and t.plate_text:
                        placas.add(t.plate_text)
            pipe.finalize()
            pipe.stop()
        return n, placas

    @pytest.mark.parametrize("workers", [2, 3])
    def test_ocr_antecipado_da_o_mesmo_resultado_do_serial(self, tmp_path, video_multi, workers):
        """Com veiculos e o agendamento que le tudo, os workers leem - e nada muda."""
        serial = LPRPipeline(
            self._config_veiculos(), VeiculoFixo(), ClassicPlateDetector(min_height=12),
            build_ocr_engine("template"), None,
        ).start()
        leituras_s = self._leituras(serial, video_multi)
        paralelo = ParallelPipeline(
            self._config_veiculos(),
            detector_factory=lambda: (VeiculoFixo(), ClassicPlateDetector(min_height=12)),
            n_workers=workers,
            ocr_factory=lambda: build_ocr_engine("template"),
            vehicle_detector=VeiculoFixo(), plate_detector=ClassicPlateDetector(min_height=12),
            ocr_engine=build_ocr_engine("template"),
        ).start()
        assert paralelo.prefetch_enabled
        leituras_p = self._leituras(paralelo, video_multi)
        assert leituras_s and leituras_p == leituras_s
        stats = paralelo.parallel_stats()
        assert stats["ocr_antecipado"]
        # Toda leitura da etapa serial veio pronta do worker.
        assert stats["leituras_antecipadas_usadas"] == len(leituras_s)
        assert stats["leituras_antecipadas"] >= len(leituras_s)

    def test_agendamento_seletivo_nao_antecipa(self, tmp_path, video_multi):
        cfg = self._config_veiculos()
        cfg.ocr.ocr_every_n = 2          # a maior parte da leitura antecipada seria jogada fora
        paralelo = ParallelPipeline(
            cfg, detector_factory=lambda: (VeiculoFixo(), ClassicPlateDetector(min_height=12)),
            n_workers=2, ocr_factory=lambda: build_ocr_engine("template"),
            vehicle_detector=VeiculoFixo(), plate_detector=ClassicPlateDetector(min_height=12),
            ocr_engine=build_ocr_engine("template"),
        ).start()
        assert not paralelo.prefetch_enabled
        assert self._leituras(paralelo, video_multi)
        assert paralelo.parallel_stats()["leituras_antecipadas"] == 0

    def test_modo_so_placa_nao_antecipa(self, tmp_path, video_multi):
        """No so-placa o recorte vem da caixa da trilha: so a etapa serial o conhece."""
        cfg = self._config(tmp_path)
        cfg.ocr.max_reads_published = 0
        paralelo = ParallelPipeline(
            cfg, detector_factory=lambda: (None, ClassicPlateDetector(min_height=12)),
            n_workers=2, ocr_factory=lambda: build_ocr_engine("template"),
            plate_detector=ClassicPlateDetector(min_height=12),
            ocr_engine=build_ocr_engine("template"),
        ).start()
        assert not paralelo.prefetch_enabled
        assert self._leituras(paralelo, video_multi)
        assert paralelo.parallel_stats()["leituras_antecipadas"] == 0

    def _config_veiculos(self) -> AppConfig:
        cfg = AppConfig()
        cfg.source.realtime = False
        cfg.detector.detect_vehicles = True
        cfg.ocr.min_plate_width = 55
        cfg.ocr.ocr_every_n = 1
        cfg.ocr.max_reads_published = 0
        cfg.ocr.max_ocr_per_frame = 4
        cfg.analytics.save_crops = False
        return cfg

    def _leituras(self, pipe, caminho) -> list[tuple]:
        with VideoSource(caminho) as src:
            leituras = [(r.frame.index, lida.frame_index, lida.text, lida.confidence,
                         tuple(lida.char_confidences))
                        for r in pipe.run(src) for lida in r.reads]
            pipe.finalize()
            pipe.stop()
        return leituras

    def test_paralelo_processa_todos_os_quadros(self, tmp_path, video_multi):
        """Regressao: a fila drop-oldest fazia o modo offline perder quadros."""
        cfg = self._config(tmp_path)
        pipe = ParallelPipeline(
            cfg,
            detector_factory=lambda: (None, ClassicPlateDetector(min_height=12)),
            n_workers=3,
            plate_detector=ClassicPlateDetector(min_height=12),
            ocr_engine=build_ocr_engine("template"),
        ).start()
        n, _ = self._rodar(pipe, video_multi)
        assert n == 80
        assert pipe.parallel_stats()["quadros_descartados_entrada"] == 0
        assert pipe.parallel_stats()["quadros_pulados_reordenacao"] == 0

    @pytest.mark.parametrize("workers", [2, 3])
    def test_mesmo_resultado_do_serial(self, tmp_path, video_multi, workers):
        """Paralelismo que muda o resultado nao serve para avaliacao cientifica."""
        cfg = self._config(tmp_path)
        serial = LPRPipeline(
            cfg, None, ClassicPlateDetector(min_height=12), build_ocr_engine("template"), None
        ).start()
        n_s, placas_s = self._rodar(serial, video_multi)

        paralelo = ParallelPipeline(
            self._config(tmp_path),
            detector_factory=lambda: (None, ClassicPlateDetector(min_height=12)),
            n_workers=workers,
            plate_detector=ClassicPlateDetector(min_height=12),
            ocr_engine=build_ocr_engine("template"),
        ).start()
        n_p, placas_p = self._rodar(paralelo, video_multi)

        assert n_p == n_s
        assert placas_p == placas_s
