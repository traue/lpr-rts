"""Diagnostico de pipeline: identificar em que estagio o fluxo para."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lprts.config import AppConfig
from lprts.diagnose import Diagnosis, StageCounts, _analisar, relatorio, run_diagnosis


def _cfg(**kwargs) -> AppConfig:
    c = AppConfig()
    for k, v in kwargs.items():
        alvo = c
        partes = k.split(".")
        for parte in partes[:-1]:
            alvo = getattr(alvo, parte)
        setattr(alvo, partes[-1], v)
    return c


class TestVeredito:
    def test_sucesso_quando_ha_eventos(self):
        c = StageCounts(frames=100, objetos=50, objetos_com_placa=50, placas=40,
                        trilhas=30, leituras=20, eventos=5)
        assert _analisar(c, _cfg()).ok

    def test_sucesso_mesmo_com_estagio_zerado_pelo_fallback(self):
        """Regressao: o diagnostico acusava falha numa execucao bem-sucedida.

        Com o fallback automatico, a deteccao de objetos fica em zero mas o
        rastreio de placas assume e o fluxo chega ao fim. Apontar falha ali
        mandaria o usuario investigar um problema inexistente.
        """
        c = StageCounts(frames=400, objetos=0, objetos_com_placa=0, placas=384,
                        trilhas=362, leituras=77, eventos=7, fallback_ativo=True)
        d = _analisar(c, _cfg())
        assert d.ok
        assert "fallback" in relatorio(d).lower() or "so-placa" in relatorio(d)

    def test_captura_vazia(self):
        d = _analisar(StageCounts(frames=0), _cfg())
        assert d.estagio_quebrado == "captura"

    def test_sem_objetos(self):
        c = StageCounts(frames=100, objetos=0, modo_so_placa=False)
        d = _analisar(c, _cfg())
        assert d.estagio_quebrado == "deteccao de objetos"
        assert any("--no-vehicle-detector" in s for s in d.solucao)

    def test_objetos_sem_placa(self):
        """O caso real: o detector so ve pedestres, que nao portam placa."""
        from collections import Counter

        c = StageCounts(frames=100, objetos=80, objetos_com_placa=0,
                        classes_vistas=Counter({"person": 80}), modo_so_placa=False)
        d = _analisar(c, _cfg())
        assert d.estagio_quebrado == "objetos que podem portar placa"
        assert "person" in d.causa

    def test_sem_placas_detectadas(self):
        c = StageCounts(frames=100, objetos=50, objetos_com_placa=50, placas=0)
        assert _analisar(c, _cfg()).estagio_quebrado == "deteccao de placas"

    def test_sem_trilhas(self):
        c = StageCounts(frames=100, objetos=50, objetos_com_placa=50, placas=40, trilhas=0)
        assert _analisar(c, _cfg()).estagio_quebrado == "rastreamento"

    def test_placa_pequena_demais_e_diagnosticada(self):
        """A causa mais comum de silencio: placa abaixo do limiar de OCR."""
        c = StageCounts(frames=100, objetos=50, objetos_com_placa=50, placas=40,
                        trilhas=30, leituras=0, larguras_de_placa=[30.0] * 40)
        d = _analisar(c, _cfg(**{"ocr.min_plate_width": 60}))
        assert "pequena" in d.estagio_quebrado
        assert "30" in d.causa and "60" in d.causa

    def test_ocr_sem_leitura_com_placa_grande(self):
        c = StageCounts(frames=100, objetos=50, objetos_com_placa=50, placas=40,
                        trilhas=30, leituras=0, larguras_de_placa=[120.0] * 40)
        d = _analisar(c, _cfg(**{"ocr.min_plate_width": 60}))
        # O estagio nomeia o motor: saber QUAL falhou e o que orienta a acao.
        assert d.estagio_quebrado.startswith("OCR")
        assert any("--ocr template" in s for s in d.solucao)

    def test_fusao_nao_publica(self):
        c = StageCounts(frames=100, objetos=50, objetos_com_placa=50, placas=40,
                        trilhas=30, leituras=20, leituras_validas=18, eventos=0,
                        larguras_de_placa=[120.0] * 40)
        d = _analisar(c, _cfg())
        assert d.estagio_quebrado == "fusao (publicacao)"
        assert any("min_reads" in s for s in d.solucao)

    def test_sintaxe_baixa_sugere_motor_profundo(self):
        c = StageCounts(frames=100, objetos=50, objetos_com_placa=50, placas=40,
                        trilhas=30, leituras=20, leituras_validas=2, eventos=0,
                        larguras_de_placa=[120.0] * 40)
        d = _analisar(c, _cfg())
        assert any("ocr" in s and "pip install" in s for s in d.solucao)


class TestRelatorio:
    def test_modo_so_placa_nao_marca_falha_em_objetos(self):
        """Sem objetos no modo so-placa e o esperado, nao um defeito."""
        c = StageCounts(frames=100, objetos=0, placas=50, trilhas=40,
                        leituras=10, eventos=3, modo_so_placa=True)
        texto = relatorio(_analisar(c, _cfg()))
        assert "nao se aplica no modo so-placa" in texto
        assert " X objetos detectados" not in texto

    def test_relatorio_lista_as_solucoes(self):
        d = _analisar(StageCounts(frames=0), _cfg())
        texto = relatorio(d, "video.mp4")
        assert "O FLUXO PARA EM" in texto and "O que fazer" in texto


@pytest.fixture
def video(tmp_path, plate_image):
    caminho = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    for i in range(90):
        cena = np.full((360, 640, 3), 95, dtype=np.uint8)
        largura = 130 + i
        altura = int(largura * 0.33)
        cena[150 : 150 + altura, 240 : 240 + largura] = cv2.resize(
            plate_image("ABC1D23"), (largura, altura)
        )
        w.write(cena)
    w.release()
    return caminho


class TestExecucao:
    def test_diagnostico_completo_num_video(self, video):
        cfg = _cfg(**{
            "source.uri": str(video),
            "detector.detect_vehicles": False,
            "detector.plate_detector": "classic",
            "ocr.engine": "template",
            "ocr.min_plate_width": 55,
            "ocr.ocr_every_n": 1,
        })
        d = run_diagnosis(cfg, max_frames=90)
        assert d.counts.frames == 90
        assert d.counts.placas > 0
        assert d.ok

    def test_nao_altera_a_configuracao_original(self, video):
        """O diagnostico limita quadros e desliga recortes - na copia, nao no original."""
        cfg = _cfg(**{
            "source.uri": str(video),
            "detector.detect_vehicles": False,
            "detector.plate_detector": "classic",
            "ocr.engine": "template",
        })
        cfg.source.max_frames = 0
        cfg.analytics.save_crops = True
        run_diagnosis(cfg, max_frames=20)
        assert cfg.source.max_frames == 0
        assert cfg.analytics.save_crops is True

    def test_placa_pequena_e_detectada_no_diagnostico(self, tmp_path, plate_image):
        """Video em que a placa nunca alcanca o limiar de OCR."""
        caminho = tmp_path / "pequena.mp4"
        w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
        for _ in range(60):
            cena = np.full((360, 640, 3), 95, dtype=np.uint8)
            cena[170:184, 300:340] = cv2.resize(plate_image("ABC1D23"), (40, 14))
            w.write(cena)
        w.release()
        cfg = _cfg(**{
            "source.uri": str(caminho),
            "detector.detect_vehicles": False,
            "detector.plate_detector": "classic",
            "ocr.engine": "template",
            "ocr.min_plate_width": 100,
        })
        d = run_diagnosis(cfg, max_frames=60)
        assert not d.ok
        assert isinstance(d, Diagnosis)


class TestVereditoComFallback:
    """Regressao: o diagnostico apontou o estagio errado numa maquina real.

    Com o fallback acionado, `modo_so_placa` continua False (foi capturado no
    inicio da execucao) e `objetos` fica em zero. A logica antiga concluia
    "o fluxo para em deteccao de objetos" quando o bloqueio real estava tres
    estagios adiante, no OCR - e mandava o usuario mexer no lugar errado.
    """

    def _contagens_reais(self, **kwargs) -> StageCounts:
        # Numeros observados numa execucao real: 150 quadros, 31 placas,
        # 29 trilhas confirmadas, zero leituras de OCR.
        base = {
            "frames": 150, "objetos": 0, "objetos_com_placa": 0, "placas": 31,
            "trilhas": 29, "trilhas_confirmadas": 29, "leituras": 0, "eventos": 0,
            "fallback_ativo": True, "modo_so_placa": False, "motor_ocr": "fastplate",
            "larguras_de_placa": [76.0] * 31,
        }
        base.update(kwargs)
        return StageCounts(**base)

    def test_aponta_o_ocr_e_nao_a_deteccao_de_objetos(self):
        d = _analisar(self._contagens_reais(), _cfg(**{"ocr.min_plate_width": 60}))
        assert "OCR" in d.estagio_quebrado
        assert "fastplate" in d.estagio_quebrado
        assert "deteccao de objetos" not in d.estagio_quebrado

    def test_menciona_o_motor_e_sugere_trocar(self):
        d = _analisar(self._contagens_reais(), _cfg(**{"ocr.min_plate_width": 60}))
        assert any("--ocr template" in s for s in d.solucao)
        assert any("DEBUG" in s for s in d.solucao)

    def test_erro_do_motor_aparece_na_causa(self):
        c = self._contagens_reais(erros_ocr={"InvalidArgument: dimensoes invalidas"})
        d = _analisar(c, _cfg(**{"ocr.min_plate_width": 60}))
        assert "InvalidArgument" in d.causa

    def test_estagios_de_objeto_marcados_como_inaplicaveis(self):
        texto = relatorio(_analisar(self._contagens_reais(), _cfg()))
        assert "nao se aplica no modo so-placa" in texto

    def test_placa_pequena_ainda_tem_prioridade_sobre_o_motor(self):
        """Quando a placa e pequena demais, a causa e o tamanho, nao o motor."""
        c = self._contagens_reais(larguras_de_placa=[30.0] * 31)
        d = _analisar(c, _cfg(**{"ocr.min_plate_width": 60}))
        assert "pequena" in d.estagio_quebrado
