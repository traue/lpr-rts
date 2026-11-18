"""Contagens da figura do conjunto de dados do artigo (scripts/artigo/figuras.py)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("figuras_artigo", RAIZ / "scripts" / "artigo" / "figuras.py")
figuras = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = figuras
_spec.loader.exec_module(figuras)

CAMPOS = ["frame_index", "x1", "y1", "x2", "y2", "plate_text", "object_class", "track_id", "kind", "difficult"]


def _gabarito(caminho: Path, linhas: list[tuple]) -> Path:
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CAMPOS)
        for quadro, x1, x2, trilha, dificil, tipo in linhas:
            w.writerow([quadro, x1, 100, x2, 130, "ABC1D23", tipo, trilha, "plate", dificil])
    return caminho


def test_veiculo_e_legivel_se_algum_quadro_nao_e_dificil(tmp_path: Path):
    gt = _gabarito(tmp_path / "gt.csv", [
        (0, 0, 60, "1", "True", "carro"),      # trilha 1: difícil no começo...
        (1, 0, 150, "1", "False", "carro"),    # ...e legível quando cresce
        (0, 0, 40, "2", "True", "motocicleta"),
        (1, 0, 70, "2", "True", "motocicleta"),  # trilha 2: difícil o tempo todo
    ])
    vs = figuras.veiculos_do_gabarito(gt)
    assert [(v.trilha, v.largura_max, v.legivel, v.tipo) for v in vs] == [
        ("1", 150.0, True, "carro"), ("2", 70.0, False, "motocicleta")]


def test_limiar_do_ocr_acompanha_a_largura_do_video():
    assert figuras.ocr_min_original(3840) == pytest.approx(135.0)
    assert figuras.ocr_min_original(2160) == pytest.approx(75.94, abs=0.01)


def test_resumo_conta_legiveis_que_chegam_ao_minimo_do_ocr(tmp_path: Path):
    gt = _gabarito(tmp_path / "gt.csv", [
        (0, 0, 200, "1", "False", "carro"),    # chega aos 135 px
        (0, 0, 100, "2", "False", "carro"),    # legível, mas não chega
        (0, 0, 300, "3", "True", "carro"),     # difícil: não entra na fração
    ])
    r = figuras.resumo(figuras.veiculos_do_gabarito(gt), 3840)
    assert (r["veiculos"], r["legiveis"], r["dificeis"], r["legiveis_ate_ocr"]) == (3, 2, 1, 1)
    assert r["frac_legiveis_ate_ocr"] == pytest.approx(0.5)


def test_figura_e_gravada(tmp_path: Path, monkeypatch):
    for _, _, arquivo, _ in figuras.VIDEOS:
        _gabarito(tmp_path / arquivo, [(0, 0, 120, "1", "False", "carro"), (0, 0, 60, "2", "True", "carro")])
    saida = tmp_path / "fig" / "dados.pdf"
    resumos = figuras.figura_dados(tmp_path, saida)
    assert saida.exists() and saida.stat().st_size > 0
    assert [r["video"] for r in resumos] == ["V1", "V2", "V3", "V4"]


def test_limiar_do_ocr_por_resolucao_de_processamento():
    # 45 px no quadro reduzido: 180, 135 e 90 px no 4K horizontal; 76 px no vertical a 1280.
    assert [figuras.limiar_ocr(3840, r) for r in figuras.RESOLUCOES] == [180.0, 135.0, 90.0]
    assert figuras.limiar_ocr(2160, 1280) == pytest.approx(75.94, abs=0.01)


def test_janela_e_a_integral_do_tempo_acima_do_limiar():
    # 30 quadros a 140 px e 30 a 100 px: só o primeiro segundo está acima de 135 px.
    larguras = [(q, 140.0) for q in range(30)] + [(q, 100.0) for q in range(30, 60)]
    assert figuras.janela_s(larguras, 135.0) == pytest.approx(1.0)
    assert figuras.janela_s(larguras, 90.0) == pytest.approx(2.0)
    assert figuras.janela_s(larguras, 200.0) == 0.0


def test_teto_exige_quadros_para_tres_tentativas():
    # Com OCR a cada 2 quadros e 3 leituras, são precisos 6 quadros acima do limiar.
    veiculos = {"a": [(q, 150.0) for q in range(6)],      # cabe
                "b": [(q, 150.0) for q in range(5)],      # não cabe
                "c": [(q, 80.0) for q in range(60)]}      # nunca chega ao limiar
    assert figuras.QUADROS_MIN == 6
    assert figuras.teto(veiculos, 135.0) == pytest.approx(1 / 3)
    assert figuras.teto(veiculos, 70.0) == pytest.approx(2 / 3)      # "b" segue com 5 quadros
