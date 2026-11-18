#!/usr/bin/env python3
"""Diagnósticos do Stage 2.5 que dependem dos dados brutos da bateria.

Uso, na máquina que tem ``runs/artigo/`` e ``data/videos/gabaritos/``::

    python scripts/artigo/diagnostico_anotacao.py
    python scripts/artigo/diagnostico_anotacao.py --manual V1=caminho/anotacao_a_mao_gt.csv

Responde a duas perguntas do relatório de integridade (``docs/artigo/pipeline/``):

1. **Tempo até a placa correta menor ou igual a zero (Fig. 5b).** O tempo conta do primeiro
   quadro anotado do veículo. Em alguns veículos a decisão correta sai nesse quadro ou
   antes. Aqui se vê em que vídeos isso acontece e como é o começo da anotação desses
   veículos: se a placa já está acima do mínimo do OCR no primeiro quadro anotado, a
   anotação começou depois de o sistema já poder ler.
2. **Efeito da pré-anotação.** A pré-anotação partiu das detecções do sistema, e por isso
   só entrou o veículo que ele detectou. Os veículos anotados à mão não dependem disso.
   Com ``--manual``, o script acha no gabarito final os veículos da anotação à mão (pela
   posição das caixas) e compara os dois grupos: revocação e veículos que o sistema não
   detectou. O arquivo é um gabarito no formato de ``scripts/cvat_para_gt.py``; se a
   anotação à mão estiver num zip do CVAT, converta com esse script antes.

Grava só agregados e identificadores de trilha, sem texto de placa, em
``docs/artigo/resultados/diagnostico_anotacao.json``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "docs" / "artigo" / "resultados" / "diagnostico_anotacao.json"

_spec = importlib.util.spec_from_file_location("analise_artigo_diag", Path(__file__).with_name("analise.py"))
analise = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = analise
_spec.loader.exec_module(analise)
fig_base = analise.fig_base

# Só as colunas sem texto de placa do relatório por veículo
COLUNAS = ["vehicle", "excluded_difficult", "detected", "published_correct", "time_to_correct_s",
           "first_frame", "max_plate_width_px"]
Caixa = tuple[float, float, float, float]


def ler_gabarito(caminho: Path) -> dict[str, dict[int, tuple[Caixa, bool]]]:
    """Caixas de placa por veículo e quadro: ``{"id:7": {quadro: (caixa, difícil)}}``.

    A chave é a mesma do relatório por veículo. O texto da placa não é lido.
    """
    veiculos: dict[str, dict[int, tuple[Caixa, bool]]] = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha.get("kind", "plate") != "plate" or not (linha.get("track_id") or "").strip():
                continue
            caixa = (float(linha["x1"]), float(linha["y1"]), float(linha["x2"]), float(linha["y2"]))
            dificil = linha["difficult"].strip().lower() in ("1", "true")
            veiculos.setdefault(f"id:{linha['track_id'].strip()}", {})[int(linha["frame_index"])] = (caixa, dificil)
    return veiculos


def ler_veiculos(caminho: Path) -> pd.DataFrame:
    """Relatório por veículo de uma sessão, sem as colunas que têm texto de placa."""
    return pd.read_csv(caminho, usecols=COLUNAS)


def tempos_nao_positivos(veic: pd.DataFrame, gabarito: dict, limiar_px: float) -> dict:
    """Entre os veículos com a placa correta, separa os de tempo <= 0 e descreve o começo da anotação."""
    metrica = veic[veic.excluded_difficult == 0]
    corretos = metrica[metrica.published_correct == 1]
    linhas = []
    for r in corretos.itertuples():
        quadros = gabarito.get(r.vehicle, {})
        if not quadros:
            continue
        primeiro = min(quadros)
        caixa, dificil = quadros[primeiro]
        linhas.append({"veiculo": r.vehicle, "tempo_s": float(r.time_to_correct_s),
                       "primeiro_quadro": int(primeiro),
                       "largura_no_primeiro_px": round(caixa[2] - caixa[0], 1),
                       "dificil_no_primeiro": bool(dificil),
                       "largura_max_px": float(r.max_plate_width_px)})
    tab = pd.DataFrame(linhas, columns=["veiculo", "tempo_s", "primeiro_quadro", "largura_no_primeiro_px",
                                       "dificil_no_primeiro", "largura_max_px"])

    def resumo(g: pd.DataFrame) -> dict:
        if g.empty:
            return {"veiculos": 0}
        return {"veiculos": int(len(g)),
                "ja_acima_do_minimo_do_ocr_no_primeiro_quadro": int((g.largura_no_primeiro_px >= limiar_px).sum()),
                "anotacao_comeca_no_quadro_zero": int((g.primeiro_quadro == 0).sum()),
                "dificil_no_primeiro_quadro": int(g.dificil_no_primeiro.sum()),
                "largura_no_primeiro_quadro_mediana_px": float(g.largura_no_primeiro_px.median()),
                "tempo_minimo_s": float(g.tempo_s.min()), "tempo_mediano_s": float(g.tempo_s.median())}

    cedo = tab[tab.tempo_s <= 0]
    return {"na_metrica": int(len(metrica)), "com_placa_correta": int(len(corretos)),
            "minimo_do_ocr_px": float(limiar_px),
            "tempo_nao_positivo": resumo(cedo), "tempo_positivo": resumo(tab[tab.tempo_s > 0]),
            "veiculos_de_tempo_nao_positivo": cedo.sort_values("tempo_s").to_dict("records")}


def iou(a: Caixa, b: Caixa) -> float:
    largura = min(a[2], b[2]) - max(a[0], b[0])
    altura = min(a[3], b[3]) - max(a[1], b[1])
    if largura <= 0 or altura <= 0:
        return 0.0
    inter = largura * altura
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def casar_manual(manual: dict, final: dict, min_comuns: int = 5, min_iou: float = 0.5) -> set[str]:
    """Veículos do gabarito final que são os da anotação à mão: mesma caixa nos mesmos quadros."""
    achados = set()
    for chave, quadros in final.items():
        for trilha in manual.values():
            comuns = quadros.keys() & trilha.keys()
            if len(comuns) < min_comuns:
                continue
            if np.median([iou(quadros[q][0], trilha[q][0]) for q in comuns]) >= min_iou:
                achados.add(chave)
                break
    return achados


def comparar_origens(veic: pd.DataFrame, a_mao: set[str]) -> dict:
    """Revocação e veículos não detectados, nos anotados à mão e nos pré-anotados."""
    metrica = veic[veic.excluded_difficult == 0]
    saida = {}
    for nome, grupo in (("anotados_a_mao", metrica[metrica.vehicle.isin(a_mao)]),
                        ("pre_anotados", metrica[~metrica.vehicle.isin(a_mao)])):
        n, k = int(len(grupo)), int(grupo.published_correct.sum())
        corretos = grupo[grupo.published_correct == 1]
        saida[nome] = {"na_metrica": n, "com_placa_correta": k,
                       "revocacao": k / n if n else None,
                       "ic95_wilson": list(analise.wilson(k, n)) if n else None,
                       "nao_detectados_pelo_sistema": int((grupo.detected == 0).sum()),
                       "tempo_nao_positivo": int((corretos.time_to_correct_s <= 0).sum())}
    return saida


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=analise.RUNS)
    ap.add_argument("--gabaritos", type=Path, default=analise.GABARITOS)
    ap.add_argument("--manual", action="append", default=[], metavar="VIDEO=GABARITO.csv",
                    help="gabarito só com a anotação à mão de um vídeo (pode repetir)")
    ap.add_argument("--out", type=Path, default=SAIDA)
    args = ap.parse_args(argv)
    manuais = dict(m.split("=", 1) for m in args.manual)

    resultado: dict = {"tempo_nao_positivo": {}, "origem_da_anotacao": {}}
    for v in analise.VIDEOS:
        gabarito = ler_gabarito(args.gabaritos / analise.GT[v])
        limiar = fig_base.limiar_ocr(analise.LARGURA_VIDEO[v], 1280)
        sem_prazo = ler_veiculos(args.runs / v / "base_off_veiculos.csv")
        r = tempos_nao_positivos(sem_prazo, gabarito, limiar)
        reps = sorted((args.runs / v).glob("base_rt_rep*_veiculos.csv"))
        r["tempo_real_por_repeticao"] = [
            int(((d.excluded_difficult == 0) & (d.published_correct == 1) & (d.time_to_correct_s <= 0)).sum())
            for d in map(ler_veiculos, reps)]
        resultado["tempo_nao_positivo"][v] = r
        cedo, tarde = r["tempo_nao_positivo"], r["tempo_positivo"]
        print(f"{v}: {r['com_placa_correta']} veículos com a placa correta; tempo <= 0 em {cedo['veiculos']}"
              + (f" ({cedo['ja_acima_do_minimo_do_ocr_no_primeiro_quadro']} já acima do mínimo do OCR no primeiro"
                 f" quadro anotado; {cedo['anotacao_comeca_no_quadro_zero']} com a anotação no quadro zero)"
                 if cedo["veiculos"] else "")
              + (f"; entre os outros {tarde['veiculos']}, "
                 f"{tarde['ja_acima_do_minimo_do_ocr_no_primeiro_quadro']} já acima do mínimo"
                 if tarde["veiculos"] else ""))
        if v in manuais:
            a_mao = casar_manual(ler_gabarito(Path(manuais[v])), gabarito)
            o = comparar_origens(sem_prazo, a_mao)
            resultado["origem_da_anotacao"][v] = o
            for nome, g in o.items():
                print(f"   {nome}: {g['com_placa_correta']} de {g['na_metrica']} com a placa correta"
                      + (f" ({100 * g['revocacao']:.0f}%)" if g["na_metrica"] else "")
                      + f"; não detectados pelo sistema: {g['nao_detectados_pelo_sistema']}")
    if not manuais:
        print("\nSem --manual, a comparação entre anotados à mão e pré-anotados não rodou.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analise._limpo(resultado), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"gravado: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
