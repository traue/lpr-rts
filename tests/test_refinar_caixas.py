"""Refino das caixas interpoladas de uma exportacao do CVAT (scripts/refinar_caixas.py)."""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("refinar_caixas", RAIZ / "scripts" / "refinar_caixas.py")
refino = importlib.util.module_from_spec(_spec)
# Registrado antes de executar: os dataclasses do script procuram o proprio modulo.
sys.modules[_spec.name] = refino
_spec.loader.exec_module(refino)

LARGURA, ALTURA = 60, 24          # a "placa"
TOPO = 100


def _x(t: int) -> float:
    """Canto esquerdo da placa no quadro t: desacelera, como um veiculo que se afasta."""
    return 100 + 40 * t - t * t


def _cena(n: int = 21, x=_x) -> list[np.ndarray]:
    """Quadros com um fundo texturizado fixo e uma placa texturizada que anda sobre ele."""
    rng = np.random.default_rng(7)
    fundo = cv2.GaussianBlur(rng.integers(0, 255, (240, 720), dtype=np.uint8), (5, 5), 0)
    placa = cv2.GaussianBlur(rng.integers(0, 255, (ALTURA, LARGURA), dtype=np.uint8), (3, 3), 0)
    quadros = []
    for t in range(n):
        img = fundo.copy()
        esquerda = int(round(x(t)))
        img[TOPO:TOPO + ALTURA, esquerda:esquerda + LARGURA] = placa
        quadros.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return quadros


def _box(quadro: int, x1: float, *, chave: int = 1, fora: int = 0) -> str:
    return (f'<box frame="{quadro}" keyframe="{chave}" outside="{fora}" occluded="0" xtl="{x1:.2f}" '
            f'ytl="{TOPO}" xbr="{x1 + LARGURA:.2f}" ybr="{TOPO + ALTURA}" z_order="0">'
            '<attribute name="texto">ABC1D23</attribute><attribute name="tipo">carro</attribute>'
            '<attribute name="dificil">false</attribute></box>')


def _exportacao(chaves: dict[int, float], *, so_chaves: bool = False, ultimo: int = 20) -> ET.Element:
    """Uma trilha como o CVAT a exporta: quadros-chave e, entre eles, a reta."""
    quadros = sorted(chaves)
    caixas = []
    for a, b in zip(quadros, quadros[1:], strict=False):
        caixas.append(_box(a, chaves[a]))
        if not so_chaves:
            caixas += [_box(q, chaves[a] + (chaves[b] - chaves[a]) * (q - a) / (b - a), chave=0)
                       for q in range(a + 1, b)]
    caixas.append(_box(quadros[-1], chaves[quadros[-1]]))
    caixas.append(_box(ultimo + 1, chaves[quadros[-1]], fora=1))
    return ET.fromstring(
        "<annotations><version>1.1</version><meta><job><size>30</size></job></meta>"
        f'<track id="3" label="placa" source="manual">{"".join(caixas)}</track></annotations>')


def _esquerdas(raiz: ET.Element) -> dict[int, float]:
    return {int(b.get("frame")): float(b.get("xtl")) for b in raiz.iter("box") if b.get("outside") == "0"}


@pytest.mark.parametrize("so_chaves", [False, True])
def test_caixa_interpolada_volta_para_cima_da_placa(so_chaves: bool):
    # Quadros-chave certos em 0, 10 e 20; no meio de cada vao a reta erra a placa por 25 px.
    raiz = _exportacao({0: _x(0), 10: _x(10), 20: _x(20)}, so_chaves=so_chaves)
    res = refino.refinar(raiz, enumerate(_cena()))
    assert res.vaos == 2 and res.aceitos["ambos"] == 2 and res.recusados == []
    assert res.caixas == 18 and res.corrigidos == 2
    achado = _esquerdas(raiz)
    assert sorted(achado) == list(range(21))             # as caixas que faltavam foram criadas
    for t in range(21):
        assert achado[t] == pytest.approx(_x(t), abs=1.0), t
    # Os quadros-chave do anotador nao mudam; os refinados passam a ser quadros-chave,
    # senao o CVAT os interpolaria de novo ao receber o arquivo.
    assert all(b.get("keyframe") == "1" for b in raiz.iter("box"))
    assert [int(b.get("frame")) for b in raiz.iter("box")] == list(range(22))
    assert {b.find("attribute[@name='texto']").text for b in raiz.iter("box")} == {"ABC1D23"}


def test_quadro_chave_fora_da_placa_nao_e_aceito():
    # O anotador pos a caixa do quadro 10 a duas larguras da placa: saindo de 0 o rastreio
    # chega na placa, nao nessa caixa. O vao nao fecha e fica como estava.
    errado = _x(10) + 2 * LARGURA
    raiz = _exportacao({0: _x(0), 10: errado})
    antes = _esquerdas(raiz)
    res = refino.refinar(raiz, enumerate(_cena()))
    assert not res.aceitos and res.caixas == 0
    (trilha, a, b, erro_frente, _), = res.recusados
    assert (trilha, a, b) == (3, 0, 10) and erro_frente == pytest.approx(2.0, abs=0.1)
    assert _esquerdas(raiz) == antes
    assert [b.get("keyframe") for b in raiz.iter("box")].count("1") == 3     # 0, 10 e a saida


def test_sem_textura_nao_ha_rastreio_e_nada_muda():
    # Quadros lisos: nao ha o que casar. A reta "fecharia" por construcao; nao pode valer.
    raiz = _exportacao({0: _x(0), 10: _x(10)})
    antes = _esquerdas(raiz)
    lisos = [(i, np.full((240, 720, 3), 90, dtype=np.uint8)) for i in range(21)]
    res = refino.refinar(raiz, lisos)
    assert not res.aceitos and len(res.recusados) == 1 and _esquerdas(raiz) == antes


def test_vao_depois_do_fim_do_video_fica_como_estava():
    raiz = _exportacao({0: _x(0), 10: _x(10), 20: _x(20)})
    antes = _esquerdas(raiz)
    res = refino.refinar(raiz, enumerate(_cena()[:15]))     # o video acaba no quadro 14
    assert res.aceitos["ambos"] == 1 and res.sem_video == 1
    depois = _esquerdas(raiz)
    assert all(depois[t] == antes[t] for t in range(11, 21))
    assert depois[5] == pytest.approx(_x(5), abs=1.0)


def test_fora_de_cena_separa_os_vaos():
    # Entre a saida e a volta nao ha placa para seguir: nao e um vao.
    raiz = ET.fromstring(
        "<annotations><version>1.1</version><meta/>"
        '<track id="1" label="placa" source="manual">'
        + _box(0, 100) + _box(4, 140) + _box(5, 140, fora=1) + _box(12, 400) + _box(16, 440)
        + _box(17, 440, fora=1) + "</track>"
        '<track id="2" label="outro" source="manual">' + _box(0, 100) + _box(9, 200) + "</track>"
        "</annotations>")
    assert [(v.trilha, v.a, v.b) for v in refino.listar_vaos(raiz, "placa")] == [(1, 0, 4), (1, 12, 16)]
    assert len(refino.listar_vaos(raiz)) == 3


def test_linha_de_comando_le_zip_e_grava_zip(tmp_path: Path):
    video = tmp_path / "v.mp4"
    escritor = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (720, 240))
    for quadro in _cena():
        escritor.write(quadro)
    escritor.release()
    entrada = tmp_path / "exportacao.zip"
    with zipfile.ZipFile(entrada, "w") as z:
        z.writestr("annotations.xml", ET.tostring(_exportacao({0: _x(0), 10: _x(10), 20: _x(20)})))
    saida = tmp_path / "refinado.zip"
    assert refino.main([str(entrada), "--video", str(video), "--out", str(saida)]) == 0
    achado = _esquerdas(refino.ler_exportacao(saida).getroot())
    # A compressao do mp4 mexe nos pixels: tolerancia maior que com os quadros exatos.
    assert achado[5] == pytest.approx(_x(5), abs=3.0) and achado[15] == pytest.approx(_x(15), abs=3.0)
