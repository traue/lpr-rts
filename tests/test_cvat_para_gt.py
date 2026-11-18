"""Conversao da exportacao do CVAT (CVAT for video 1.1) no ground truth do LPR-RTS."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lprts.eval.groundtruth import load_ground_truth

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cvat_para_gt", RAIZ / "scripts" / "cvat_para_gt.py")
cvat = importlib.util.module_from_spec(_spec)
# Registrado antes de executar: os dataclasses do script procuram o proprio modulo.
sys.modules[_spec.name] = cvat
_spec.loader.exec_module(cvat)


def _box(quadro: int, x1: float, *, fora: int = 0, chave: int = 1, texto: str = "ABC1D23",
         tipo: str = "carro", dificil: str = "false") -> str:
    return (
        f'<box frame="{quadro}" keyframe="{chave}" outside="{fora}" occluded="0" '
        f'xtl="{x1}" ytl="50" xbr="{x1 + 60}" ybr="70" z_order="0">'
        f'<attribute name="texto">{texto}</attribute>'
        f'<attribute name="tipo">{tipo}</attribute>'
        f'<attribute name="dificil">{dificil}</attribute></box>'
    )


def _xml(*trilhas: str, quadros: int = 100, largura: int = 1920, altura: int = 1080) -> str:
    return (
        "<annotations><version>1.1</version><meta><task><size>"
        f"{quadros}</size><mode>interpolation</mode><original_size><width>{largura}</width>"
        f"<height>{altura}</height></original_size></task></meta>{''.join(trilhas)}</annotations>"
    )


def _trilha(tid: int, *caixas: str, rotulo: str = "placa") -> str:
    return f'<track id="{tid}" label="{rotulo}" source="manual">{"".join(caixas)}</track>'


def _et(xml: str):
    import xml.etree.ElementTree as ET

    return ET.fromstring(xml)


def _converter(xml: str):
    return cvat.converter(_et(xml))


def test_interpola_entre_quadros_chave_e_para_na_saida():
    res = _converter(_xml(_trilha(
        3, _box(10, 100, texto="abc-1d23", tipo="moto"), _box(20, 200), _box(30, 200, fora=1),
    )))
    assert res.erros == [] and res.avisos == []
    por_quadro = {a.frame_index: a for a in res.anotacoes}
    assert sorted(por_quadro) == list(range(10, 30))   # 30 esta fora de cena
    assert por_quadro[15].bbox[0] == pytest.approx(150.0)
    # Entre uma caixa visivel e uma marcada fora, a caixa fica parada (como no CVAT).
    assert por_quadro[25].bbox[0] == pytest.approx(200.0)
    assert {a.plate_text for a in res.anotacoes} == {"ABC1D23"}
    assert {a.object_class for a in res.anotacoes} == {"motocicleta"}
    assert {a.track_id for a in res.anotacoes} == {3}
    assert res.trilhas == 1 and res.interpoladas == 18


def test_trilha_sem_saida_nao_e_estendida_ate_o_fim():
    # Sem a marca de saida, o CVAT exporta a caixa parada nos quadros seguintes.
    res = _converter(_xml(_trilha(
        1, _box(5, 100), _box(8, 130), _box(9, 130, chave=0), _box(10, 130, chave=0),
    )))
    assert max(a.frame_index for a in res.anotacoes) == 8
    assert any("sem a marca de saida" in a for a in res.avisos)


def test_trilha_sem_texto_so_passa_se_dificil():
    sem_texto = _converter(_xml(_trilha(1, _box(1, 10, texto=""), _box(3, 10, texto="", fora=1))))
    assert sem_texto.erros and "sem texto" in sem_texto.erros[0]
    assert sem_texto.anotacoes == []
    dificil = _converter(_xml(_trilha(
        1, _box(1, 10, texto="", dificil="true"), _box(3, 10, texto="", dificil="true", fora=1),
    )))
    assert dificil.erros == [] and dificil.dificeis == 1
    assert all(a.difficult and a.plate_text == "" for a in dificil.anotacoes)


def test_texto_que_muda_na_trilha_e_erro_e_padrao_estranho_e_aviso():
    muda = _converter(_xml(_trilha(1, _box(1, 10, texto="ABC1D23"), _box(2, 10, texto="ABC1D28"))))
    assert muda.erros and "muda" in muda.erros[0]
    estranho = _converter(_xml(_trilha(1, _box(1, 10, texto="AB12"), _box(2, 10, texto="AB12"))))
    assert estranho.erros == []
    assert any("padrao" in a for a in estranho.avisos)


def test_mesma_placa_em_duas_trilhas_e_outros_rotulos_geram_aviso():
    res = _converter(_xml(
        _trilha(1, _box(1, 10), _box(2, 10, fora=1)),
        _trilha(2, _box(50, 10), _box(51, 10, fora=1)),
        _trilha(3, _box(1, 300), rotulo="veiculo"),
    ))
    assert any("2 trilhas" in a for a in res.avisos)
    assert any("'veiculo'" in a for a in res.avisos)
    assert res.trilhas == 2


def test_texto_com_todos_os_quadros_dificeis_gera_aviso():
    # Quem digitou a placa a leu: o veiculo nao deveria sair da metrica.
    res = _converter(_xml(_trilha(
        4, _box(1, 10, dificil="true"), _box(5, 10, dificil="true"), _box(6, 10, fora=1),
    )))
    assert res.erros == [] and res.dificeis == 1
    assert any("trilha 4" in a and "'dificil'" in a for a in res.avisos)
    # Dificil so em parte dos quadros e o uso previsto: sem aviso.
    parcial = _converter(_xml(_trilha(
        4, _box(1, 10), _box(5, 10, dificil="true"), _box(6, 10, fora=1),
    )))
    assert parcial.avisos == [] and parcial.dificeis == 0


def test_interpolacao_que_escapa_da_placa_gera_aviso():
    # A placa desacelera: em 10 quadros anda 300 px, nos 10 seguintes, 100. A reta
    # 0->20 erraria o quadro 10 por 100 px, e no meio de cada vao sobram 25 px de
    # desvio numa caixa de 60 px (0,42 largura; o IoU cai abaixo de 0,5).
    curva = _converter(_xml(_trilha(
        7, _box(0, 100), _box(10, 400), _box(20, 500), _box(21, 500, fora=1),
    )))
    aviso = next(a for a in curva.avisos if "interpolada" in a)
    assert "trilha 7" in aviso and "0-10" in aviso and "10-20" in aviso
    assert cvat.desvios_da_interpolacao(cvat.ler_trilha(_et(_trilha(
        7, _box(0, 100), _box(10, 400), _box(20, 500),
    ))))[0][2] == pytest.approx(25 / 60)
    # Velocidade constante: a reta acerta, por mais rapido que a placa ande.
    reta = _converter(_xml(_trilha(
        7, _box(0, 100), _box(10, 1100), _box(20, 2100), _box(21, 2100, fora=1),
    )))
    assert reta.avisos == []
    # Com quadros-chave no meio dos vaos, o mesmo movimento (x = 100 + 40t - t^2)
    # deixa de gerar aviso: o desvio cai para 6 px.
    densa = _converter(_xml(_trilha(
        7, _box(0, 100), _box(5, 275), _box(10, 400), _box(15, 475), _box(20, 500),
        _box(21, 500, fora=1),
    )))
    assert not any("interpolada" in a for a in densa.avisos)


def test_curvatura_nao_atravessa_a_lacuna_fora_de_cena():
    # Antes e depois de uma oclusao a placa esta em lugares sem relacao: os dois
    # trechos nao formam uma trinca.
    res = _converter(_xml(_trilha(
        2, _box(0, 100), _box(10, 200), _box(11, 200, fora=1),
        _box(40, 900), _box(50, 1000), _box(51, 1000, fora=1),
    )))
    assert not any("interpolada" in a for a in res.avisos)


def test_ate_gera_o_gabarito_de_um_corte_do_video():
    # Tarefa do CVAT com o video inteiro, avaliacao num corte de 10 quadros.
    xml = _xml(
        _trilha(1, _box(2, 100), _box(6, 140), _box(8, 140, fora=1)),      # toda antes do corte
        _trilha(2, _box(5, 100, texto="DEF2G34"), _box(15, 200, texto="DEF2G34"),
                _box(20, 200, texto="DEF2G34", fora=1)),                     # atravessa o corte
        _trilha(3, _box(12, 100, texto="HIJ5K67"),
                _box(14, 100, texto="HIJ5K67", fora=1)),                     # toda depois
        quadros=100,
    )
    res = cvat.converter(_et(xml), ate=10)
    assert res.erros == [] and res.avisos == []          # cortar nao e "faltou a marca de saida"
    assert {a.track_id for a in res.anotacoes} == {1, 2}
    da_2 = {a.frame_index: a for a in res.anotacoes if a.track_id == 2}
    assert sorted(da_2) == [5, 6, 7, 8, 9]
    # A ultima caixa vem da interpolacao rumo ao quadro-chave 15, que ficou depois do corte.
    assert da_2[9].bbox[0] == pytest.approx(140.0)
    assert (res.trilhas, res.cortadas, res.fora_do_corte) == (2, 1, 1)
    # Trilha 1: quadros 3-5 e o 7 (parado ate a saida); trilha 2: quadros 6-9.
    assert res.interpoladas == 4 + 4 and res.quadros_no_cvat == 10
    # Sem o corte, as tres trilhas entram inteiras.
    assert cvat.converter(_et(xml)).trilhas == 3


def test_sem_trilhas_do_rotulo_e_erro():
    assert _converter(_xml(_trilha(1, _box(1, 10), rotulo="carro"))).erros


def test_zip_exportado_vira_csv_que_a_avaliacao_le(tmp_path: Path):
    zipado = tmp_path / "exportacao.zip"
    with zipfile.ZipFile(zipado, "w") as z:
        z.writestr("annotations.xml", _xml(
            _trilha(0, _box(0, 100), _box(4, 140), _box(6, 140, fora=1)),
            _trilha(1, _box(2, 500, texto="XYZ9876"), _box(3, 500, texto="XYZ9876", fora=1)),
        ))
    saida = tmp_path / "v1_gt.csv"
    assert cvat.main([str(zipado), "--out", str(saida)]) == 0
    gt = load_ground_truth(saida)
    assert gt.has_track_ids()
    veiculos = gt.vehicles()
    assert set(veiculos) == {"id:0", "id:1"}
    assert sorted(veiculos["id:0"].boxes) == [0, 1, 2, 3, 4, 5]
    assert veiculos["id:1"].plate_text == "XYZ9876"


def test_erro_nao_grava_nada(tmp_path: Path):
    xml = tmp_path / "annotations.xml"
    xml.write_text(_xml(_trilha(1, _box(1, 10, texto=""), _box(2, 10, texto=""))), encoding="utf-8")
    saida = tmp_path / "gt.csv"
    assert cvat.main([str(xml), "--out", str(saida)]) == 1
    assert not saida.exists()


def test_conferencia_com_o_video_acusa_orientacao_e_quadros(tmp_path: Path):
    video = tmp_path / "v.mp4"
    escritor = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 32))
    for i in range(5):
        escritor.write(np.full((32, 64, 3), i * 40, dtype=np.uint8))
    escritor.release()
    res = _converter(_xml(_trilha(1, _box(1, 10), _box(2, 10, fora=1)),
                          quadros=50, largura=32, altura=64))
    avisos = cvat.conferir_video(res, video)
    assert any("orientacao diferente" in a for a in avisos)
    assert any("numero de quadros diferente" in a for a in avisos)
    assert cvat.conferir_video(res, tmp_path / "nao_existe.mp4")[0].startswith("nao consegui")

    # Exportacao de um trabalho (Job > Export): <original_size> vem direto em <meta>,
    # ao lado de <job>. A conferencia de orientacao tem de valer do mesmo jeito.
    de_job = _converter(
        "<annotations><version>1.1</version><meta><job><id>1</id><size>50</size>"
        "<mode>interpolation</mode><start_frame>0</start_frame><stop_frame>49</stop_frame>"
        "</job><dumped>2026-11-11</dumped><original_size><width>32</width><height>64</height>"
        "</original_size></meta>"
        + _trilha(1, _box(1, 10), _box(2, 10, fora=1)) + "</annotations>"
    )
    assert de_job.quadros_no_cvat == 50 and de_job.tamanho_no_cvat == (32, 64)
    avisos = cvat.conferir_video(de_job, video)
    assert any("orientacao diferente" in a for a in avisos)
    assert any("numero de quadros diferente" in a for a in avisos)
