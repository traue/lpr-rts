"""Diagnósticos do Stage 2.5 (scripts/artigo/diagnostico_anotacao.py e diagnostico_ritmo.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent


def _carregar(nome: str):
    spec = importlib.util.spec_from_file_location(f"{nome}_teste", RAIZ / "scripts" / "artigo" / f"{nome}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


diag = _carregar("diagnostico_anotacao")
ritmo = _carregar("diagnostico_ritmo")

CABECALHO = "frame_index,x1,y1,x2,y2,plate_text,object_class,track_id,kind,difficult\n"


def _gabarito(caminho: Path, trilhas: dict[int, list[tuple[int, float, float, int]]]) -> Path:
    """trilhas = {id: [(quadro, x1, largura, difícil), ...]}; a placa tem 40 px de altura."""
    linhas = [f"{q},{x1},100,{x1 + w},140,ABC1D23,carro,{tid},plate,{dif}\n"
              for tid, quadros in trilhas.items() for q, x1, w, dif in quadros]
    # uma caixa de veículo, que os diagnósticos têm de ignorar
    linhas.append("0,0,0,500,500,,carro,1,object,0\n")
    caminho.write_text(CABECALHO + "".join(linhas), encoding="utf-8")
    return caminho


def _veiculos(linhas: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(linhas, columns=["vehicle", "excluded_difficult", "detected", "published_correct",
                                         "time_to_correct_s", "first_frame", "max_plate_width_px"])


def test_ler_gabarito_fica_so_com_as_placas_e_nao_guarda_o_texto(tmp_path: Path):
    g = diag.ler_gabarito(_gabarito(tmp_path / "gt.csv", {1: [(0, 10, 150, 0), (1, 12, 152, 0)],
                                                           2: [(5, 300, 60, 1)]}))
    assert set(g) == {"id:1", "id:2"}
    assert g["id:1"][1] == ((12.0, 100.0, 164.0, 140.0), False)
    assert g["id:2"][5][1] is True
    assert "ABC1D23" not in json.dumps(g)


def test_tempo_nao_positivo_descreve_o_comeco_da_anotacao(tmp_path: Path):
    gabarito = diag.ler_gabarito(_gabarito(tmp_path / "gt.csv", {
        1: [(0, 10, 150, 0), (1, 10, 160, 0)],      # já legível no quadro zero; decisão antes da anotação
        2: [(40, 10, 60, 0), (41, 10, 140, 0)],     # começa pequena; decisão depois
        3: [(50, 10, 200, 0)],                      # correta no próprio primeiro quadro
        4: [(60, 10, 200, 0)],                      # na métrica, sem a placa correta
        5: [(70, 10, 200, 1)],                      # fora da métrica
    }))
    veic = _veiculos([("id:1", 0, 1, 1, -0.1, 0, 160.0), ("id:2", 0, 1, 1, 0.4, 40, 140.0),
                      ("id:3", 0, 1, 1, 0.0, 50, 200.0), ("id:4", 0, 1, 0, None, 60, 200.0),
                      ("id:5", 1, 1, 1, -0.2, 70, 200.0)])
    r = diag.tempos_nao_positivos(veic, gabarito, limiar_px=135.0)
    assert (r["na_metrica"], r["com_placa_correta"]) == (4, 3)
    cedo, tarde = r["tempo_nao_positivo"], r["tempo_positivo"]
    assert cedo["veiculos"] == 2 and tarde["veiculos"] == 1
    assert cedo["ja_acima_do_minimo_do_ocr_no_primeiro_quadro"] == 2
    assert cedo["anotacao_comeca_no_quadro_zero"] == 1
    assert cedo["tempo_minimo_s"] == pytest.approx(-0.1)
    assert tarde["ja_acima_do_minimo_do_ocr_no_primeiro_quadro"] == 0
    assert [v["veiculo"] for v in r["veiculos_de_tempo_nao_positivo"]] == ["id:1", "id:3"]


def test_sem_veiculo_correto_o_resumo_e_vazio(tmp_path: Path):
    gabarito = diag.ler_gabarito(_gabarito(tmp_path / "gt.csv", {1: [(0, 10, 150, 0)]}))
    r = diag.tempos_nao_positivos(_veiculos([("id:1", 0, 0, 0, None, 0, 150.0)]), gabarito, 135.0)
    assert r["tempo_nao_positivo"] == {"veiculos": 0} and r["veiculos_de_tempo_nao_positivo"] == []


def test_casar_manual_acha_o_veiculo_pela_posicao(tmp_path: Path):
    final = diag.ler_gabarito(_gabarito(tmp_path / "final.csv", {
        7: [(q, 10 + q, 150, 0) for q in range(10)],
        8: [(q, 900, 150, 0) for q in range(10)],
        9: [(q, 10 + q, 150, 0) for q in range(100, 103)],       # poucos quadros em comum
    }))
    manual = diag.ler_gabarito(_gabarito(tmp_path / "manual.csv", {
        1: [(q, 12 + q, 150, 0) for q in range(10)],              # quase a mesma caixa do 7
        2: [(q, 12 + q, 150, 0) for q in range(100, 103)],
    }))
    assert diag.casar_manual(manual, final) == {"id:7"}


def test_comparar_origens_separa_os_dois_grupos():
    veic = _veiculos([("id:1", 0, 1, 1, 0.2, 0, 150.0), ("id:2", 0, 0, 0, None, 5, 150.0),
                      ("id:3", 0, 1, 1, -0.1, 9, 150.0), ("id:4", 0, 1, 0, None, 9, 150.0),
                      ("id:5", 1, 1, 1, 0.1, 9, 150.0)])
    o = diag.comparar_origens(veic, {"id:1", "id:2", "id:5"})
    assert o["anotados_a_mao"]["na_metrica"] == 2 and o["anotados_a_mao"]["revocacao"] == 0.5
    assert o["anotados_a_mao"]["nao_detectados_pelo_sistema"] == 1
    assert o["pre_anotados"]["na_metrica"] == 2 and o["pre_anotados"]["tempo_nao_positivo"] == 1
    lo, hi = o["pre_anotados"]["ic95_wilson"]
    assert 0 <= lo < 0.5 < hi <= 1


def test_o_arquivo_gravado_nao_tem_texto_de_placa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runs, gabaritos = tmp_path / "runs", tmp_path / "gabaritos"
    gabaritos.mkdir()
    for v in diag.analise.VIDEOS:
        (runs / v).mkdir(parents=True)
        _gabarito(gabaritos / diag.analise.GT[v], {1: [(q, 10, 150, 0) for q in range(8)],
                                                   2: [(q, 600, 150, 0) for q in range(8)]})
        linhas = "vehicle,plate_text,fused_text,excluded_difficult,detected,published_correct," \
                 "time_to_correct_s,first_frame,max_plate_width_px\n" \
                 "id:1,ABC1D23,ABC1D23,0,1,1,-0.033,0,150.0\nid:2,XYZ9K88,XYZ9K80,0,1,0,,0,150.0\n"
        (runs / v / "base_off_veiculos.csv").write_text(linhas, encoding="utf-8")
        (runs / v / "base_rt_rep1_veiculos.csv").write_text(linhas, encoding="utf-8")
    saida = tmp_path / "diag.json"
    manual = _gabarito(tmp_path / "manual.csv", {40: [(q, 10, 150, 0) for q in range(8)]})
    assert diag.main(["--runs", str(runs), "--gabaritos", str(gabaritos), "--manual", f"V1={manual}",
                      "--out", str(saida)]) == 0
    texto = saida.read_text(encoding="utf-8")
    assert "ABC1D23" not in texto and "XYZ9K8" not in texto
    r = json.loads(texto)
    assert r["tempo_nao_positivo"]["V1"]["tempo_nao_positivo"]["veiculos"] == 1
    assert r["tempo_nao_positivo"]["V1"]["tempo_real_por_repeticao"] == [1]
    assert r["origem_da_anotacao"]["V1"]["anotados_a_mao"]["na_metrica"] == 1
    assert set(r["origem_da_anotacao"]) == {"V1"}


def test_ritmo_mede_so_o_trabalho_e_nao_o_sono():
    chamadas = []

    def quadro():
        chamadas.append(1)
        return 0.002, 0.001, 0.007

    r = ritmo.medir(quadro, quadros=5, periodo=0.001, aquecimento=2)
    assert len(chamadas) == 7 and r["quadros"] == 5
    assert r["veiculos_ms"] == pytest.approx(2.0) and r["placa_por_roi_ms"] == pytest.approx(1.0)
    assert r["quadro_ms"] == pytest.approx(7.0)


def test_ritmo_razoes_comparam_com_o_regime_sem_pausa():
    regimes = {"A sem pausa": {"quadro_ms": 8.0}, "A sem pausa, de novo": {"quadro_ms": 8.0},
               "B 30 quadros/s": {"quadro_ms": 16.0}, "B 30 quadros/s, de novo": {"quadro_ms": 18.0},
               "C 30 quadros/s, CPU ocupada": {"quadro_ms": 8.4}}
    assert ritmo.razoes(regimes) == {"B_sobre_A": pytest.approx(2.125), "C_sobre_A": pytest.approx(1.05)}
