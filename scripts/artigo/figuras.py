#!/usr/bin/env python3
"""Figuras do artigo (docs/artigo/figuras/) a partir dos gabaritos e, depois, da bateria.

Uso::

    python scripts/artigo/figuras.py dados        # caracterização do conjunto de dados
    python scripts/artigo/figuras.py fig2         # Figura 2: dados, janela de leitura e teto

Os gabaritos (`data/videos/gabaritos/*_gt.csv`) têm textos de placa e ficam fora do git; as figuras
só levam larguras e contagens, sem texto de placa, e podem ser versionadas.

Estilo: PDF vetorial na largura de uma coluna do AASTeX (3,35 in), fonte sem serifa de
7 pt, paleta categórica de referência da skill de visualização em ordem fixa (azul,
laranja, ...), e forma de marcador como segundo canal para leitura em preto e branco.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "docs" / "artigo" / "figuras"

# Os quatro vídeos do artigo, na ordem da bateria (V1..V4): gabarito e largura do quadro
# exibido (o vertical tem 2160 px de largura).
VIDEOS = [
    ("V1", "Paulista, dia", "IMG_1179c_gt.csv", 3840),
    ("V2", "Paulista, noite", "IMG_1184b_gt.csv", 3840),
    ("V3", "Consolação, dia", "IMG_0834_gt.csv", 3840),
    ("V4", "Consolação, dia (vertical)", "IMG_0833_gt.csv", 2160),
]
ESCOPO_PX = 50          # escopo da avaliação, em px do quadro original (docs/AVALIACAO.md)
OCR_MIN_PROC = 45       # ocr.min_plate_width do perfil de tempo real, em px do quadro reduzido
LARGURA_PROC = 1280     # source.resize_width do perfil de tempo real

# Paleta categórica de referência (slots 1 e 2) e tintas de texto.
AZUL, LARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#d9d8d4"


@dataclass(frozen=True)
class Veiculo:
    trilha: str
    largura_max: float      # maior largura da caixa da placa, em px do quadro original
    legivel: bool           # tem ao menos um quadro não marcado como difícil
    tipo: str


def veiculos_do_gabarito(caminho: Path) -> list[Veiculo]:
    """Um registro por veículo (trilha) do gabarito, sem o texto da placa."""
    largura: dict[str, float] = {}
    legivel: dict[str, bool] = {}
    tipo: dict[str, str] = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha.get("kind", "plate") != "plate":
                continue
            t = linha["track_id"]
            w = float(linha["x2"]) - float(linha["x1"])
            largura[t] = max(largura.get(t, 0.0), w)
            dificil = linha["difficult"].strip().lower() in ("1", "true")
            legivel[t] = legivel.get(t, False) or not dificil
            tipo[t] = linha.get("object_class") or ""
    return [Veiculo(t, largura[t], legivel[t], tipo[t]) for t in sorted(largura, key=_chave)]


def _chave(t: str) -> tuple[int, str]:
    return (int(t), "") if t.isdigit() else (sys.maxsize, t)


def ocr_min_original(largura_video: int) -> float:
    """Menor placa que o OCR do perfil de tempo real tenta, em px do quadro original."""
    return OCR_MIN_PROC * largura_video / LARGURA_PROC


def resumo(veiculos: list[Veiculo], largura_video: int) -> dict[str, float]:
    legiveis = [v for v in veiculos if v.legivel]
    limiar = ocr_min_original(largura_video)
    alcancam = sum(1 for v in legiveis if v.largura_max >= limiar)
    return {
        "veiculos": len(veiculos),
        "legiveis": len(legiveis),
        "dificeis": len(veiculos) - len(legiveis),
        "ocr_min_px": limiar,
        "legiveis_ate_ocr": alcancam,
        "frac_legiveis_ate_ocr": alcancam / len(legiveis) if legiveis else float("nan"),
    }


FPS = 30.0
OCR_A_CADA = 2          # ocr.ocr_every_n do perfil de tempo real
LEITURAS_MIN = 3        # fusion.min_reads: leituras para a primeira decisão
QUADROS_MIN = OCR_A_CADA * LEITURAS_MIN     # quadros acima do limiar para caber 3 tentativas
RESOLUCOES = (960, 1280, 1920)              # larguras de processamento da bateria (E1)


def larguras_por_veiculo(caminho: Path) -> dict[str, list[tuple[int, float]]]:
    """(quadro, largura da placa) de cada veículo legível, em ordem de quadro."""
    por: dict[str, list[tuple[int, float]]] = {}
    legivel: dict[str, bool] = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha.get("kind", "plate") != "plate":
                continue
            t = linha["track_id"]
            por.setdefault(t, []).append((int(linha["frame_index"]),
                                          float(linha["x2"]) - float(linha["x1"])))
            dificil = linha["difficult"].strip().lower() in ("1", "true")
            legivel[t] = legivel.get(t, False) or not dificil
    return {t: sorted(v) for t, v in por.items() if legivel[t]}


def limiar_ocr(largura_video: int, resolucao: int) -> float:
    """Menor placa (px do quadro original) que o OCR tenta quando o quadro é reduzido a ``resolucao``."""
    return OCR_MIN_PROC * largura_video / resolucao


def janela_s(larguras: list[tuple[int, float]], limiar: float) -> float:
    """Janela de leitura: T = ∫ 1[largura(t) ≥ limiar] dt, somada quadro a quadro (1/30 s)."""
    return sum(1 for _, w in larguras if w >= limiar) / FPS


def teto(veiculos: dict[str, list[tuple[int, float]]], limiar: float) -> float:
    """Fração dos veículos legíveis cuja janela comporta as tentativas da primeira decisão."""
    if not veiculos:
        return float("nan")
    cabe = sum(1 for v in veiculos.values() if janela_s(v, limiar) * FPS >= QUADROS_MIN)
    return cabe / len(veiculos)


CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # slots 1-4 da paleta, ordem fixa
MARCAS = ["o", "s", "^", "D"]


def _painel_teto(ax, dir_gabaritos: Path, compacto: bool = False) -> list[dict[str, float]]:
    """Teto da revocação em função do limiar do OCR, por vídeo, com as três resoluções."""
    import matplotlib
    import numpy as np

    tabelas = []
    limiares = np.arange(40, 301, 2)
    for (vid, nome, arquivo, largura_video), cor, marca in zip(VIDEOS, CORES, MARCAS, strict=True):
        vs = larguras_por_veiculo(dir_gabaritos / arquivo)
        curva = [teto(vs, w) for w in limiares]
        ax.plot(limiares, curva, color=cor, linewidth=1.4,
                label=f"{vid} ({nome.replace(', dia (vertical)', ', vertical')})", zorder=3)
        for r in RESOLUCOES:
            w = limiar_ocr(largura_video, r)
            ax.plot([w], [teto(vs, w)], marker=marca, markersize=4.2, color=cor,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        tabelas.append({"video": vid, "legiveis": len(vs),
                        **{f"teto_{r}": teto(vs, limiar_ocr(largura_video, r)) for r in RESOLUCOES},
                        **{f"limiar_{r}": limiar_ocr(largura_video, r) for r in RESOLUCOES},
                        "janela_mediana_1280": float(np.median([janela_s(v, limiar_ocr(largura_video, 1280))
                                                                for v in vs.values()]))})
    for r in RESOLUCOES:
        w = limiar_ocr(3840, r)
        ax.axvline(w, color=GRADE, linewidth=0.8, zorder=1)
        if compacto:
            # no alto, a 960 px, nenhuma curva passa; embaixo, o rótulo ficaria sob o marcador do V1
            y, va = (0.98, "top") if r == 960 else (0.02, "bottom")
            ax.text(w + 2, y, f"{r} px", ha="left", va=va, fontsize=5.4, color=TINTA_2,
                    rotation=90, transform=ax.get_xaxis_transform())
        else:
            ax.text(w + 2, 0.015, f"{r} px", ha="left", va="bottom", fontsize=6, color=TINTA_2,
                    transform=ax.get_xaxis_transform())
    ax.set_xlim(40, 300)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Mínimo do OCR (px do quadro original)" if compacto
                  else "Menor placa que o OCR tenta (px do quadro original)")
    ax.set_ylabel("Teto da revocação")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(color=GRADE, linewidth=0.4, zorder=0)
    if compacto:
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.26), ncol=2, frameon=False, handlelength=1.4,
                  fontsize=5.6, columnspacing=0.9)
    else:
        ax.legend(loc="upper right", frameon=False, handlelength=1.6, fontsize=6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    return tabelas


def _painel_exemplo(ax, dir_gabaritos: Path, compacto: bool = False) -> None:
    """A integral num veículo do V1: a área acima do limiar vira tempo."""
    import numpy as np

    vs1 = larguras_por_veiculo(dir_gabaritos / VIDEOS[0][2])
    w135 = limiar_ocr(3840, 1280)
    candidatos = [v for v in vs1.values() if 0.4 <= janela_s(v, w135) <= 1.6 and len(v) >= 90]
    exemplo = max(candidatos, key=len) if candidatos else max(vs1.values(), key=len)
    t0 = exemplo[0][0]
    ts = np.array([(q - t0) / FPS for q, _ in exemplo])
    ws = np.array([w for _, w in exemplo])
    ax.plot(ts, ws, color=CORES[0], linewidth=1.3, zorder=3)
    ax.fill_between(ts, w135, ws, where=ws >= w135, color=CORES[0], alpha=0.18, linewidth=0, zorder=2)
    ax.axhline(w135, color=TINTA_2, linestyle=(0, (1, 1.5)), linewidth=0.8, zorder=1)
    ax.text(ts[-1], w135 - 1.5, "mínimo do OCR\n(1280 px)", ha="right", va="top", fontsize=6, color=TINTA_2,
            multialignment="right")
    tj = janela_s(exemplo, w135)
    acima = ts[ws >= w135]
    if len(acima):
        ax.annotate(f"T = {tj:.2f} s".replace(".", ","), xy=(float(np.median(acima)), w135 + 4),
                    xytext=(22, 14), textcoords="offset points", ha="left", fontsize=6.5,
                    color=TINTA, arrowprops={"arrowstyle": "-", "color": TINTA_2, "lw": 0.5})
    ax.set_xlabel("Tempo desde a primeira anotação (s)" if not compacto else "Tempo (s)")
    ax.set_ylabel("Largura da placa (px)")
    ax.grid(color=GRADE, linewidth=0.4, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _paineis_dados(eixos, dir_gabaritos: Path, fonte: float = 6.8, compacto: bool = False) -> list[dict[str, float]]:
    """Uma faixa por vídeo: maior largura da placa de cada veículo, legível ou difícil."""
    import numpy as np

    rng = np.random.default_rng(7)          # tremor vertical reprodutível
    resumos, dados, x_max = [], [], 0.0
    for vid, nome, arquivo, largura_video in VIDEOS:
        vs = veiculos_do_gabarito(dir_gabaritos / arquivo)
        dados.append((vid, nome, largura_video, vs))
        x_max = max(x_max, max(v.largura_max for v in vs))
    x_max = float(np.ceil(x_max / 50.0) * 50.0)
    for ax, (vid, nome, largura_video, vs) in zip(eixos, dados, strict=True):
        r = resumo(vs, largura_video)
        resumos.append({"video": vid, **r})
        for legivel, cor, marca, rotulo in ((False, LARANJA, "x", "difícil"), (True, AZUL, "o", "legível")):
            xs = np.array([v.largura_max for v in vs if v.legivel == legivel])
            ys = rng.uniform(-0.32, 0.32, size=len(xs))
            if marca == "o":
                ax.scatter(xs, ys, s=5, marker=marca, facecolor=cor, edgecolor="white",
                           linewidth=0.3, label=rotulo, zorder=3)
            else:
                ax.scatter(xs, ys, s=5, marker=marca, color=cor, linewidth=0.6, alpha=0.8, label=rotulo, zorder=2)
        ax.axvline(ESCOPO_PX, color=TINTA_2, linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
        ax.axvline(r["ocr_min_px"], color=TINTA_2, linestyle=(0, (1, 1.5)), linewidth=0.8, zorder=1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlim(0, x_max)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.grid(axis="x", color=GRADE, linewidth=0.4, zorder=0)
        ax.text(0.0, 1.02, f"{vid} · {nome}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=fonte, color=TINTA)
        contagem = (f"{r['legiveis']:.0f} | {r['dificeis']:.0f}" if compacto else
                    f"{_n(r['legiveis'], 'legível', 'legíveis')}, {_n(r['dificeis'], 'difícil', 'difíceis')}")
        ax.text(1.0, 1.02, contagem, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=fonte - 0.5, color=TINTA_2)
    for ax in eixos[:-1]:
        ax.tick_params(labelbottom=False)
    eixos[-1].set_xlabel("Maior largura da placa (px do original)" if compacto
                         else "Maior largura da placa no vídeo (px do quadro original)")
    return resumos


def _legenda_dados(eixo0):
    from matplotlib.lines import Line2D
    alcas, nomes = eixo0.get_legend_handles_labels()
    alcas = alcas[::-1] + [Line2D([], [], color=TINTA_2, linestyle=(0, (3, 2)), linewidth=0.7),
                           Line2D([], [], color=TINTA_2, linestyle=(0, (1, 1.5)), linewidth=0.8)]
    nomes = nomes[::-1] + [f"escopo da avaliação ({ESCOPO_PX} px)", "mínimo do OCR em tempo real"]
    return alcas, nomes


def figura_janela(dir_gabaritos: Path, saida: Path) -> list[dict[str, float]]:
    """(a) a integral num veículo; (b) o teto de revocação em função do limiar do OCR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _estilo(plt)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 2.25),
                                     gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
    tabelas = _painel_teto(ax_b, dir_gabaritos)
    _painel_exemplo(ax_a, dir_gabaritos)
    ax_a.set_title("(a)", loc="left", fontsize=7.5, fontweight="bold")
    ax_b.set_title("(b)", loc="left", fontsize=7.5, fontweight="bold")
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return tabelas


def _n(n: float, singular: str, plural: str) -> str:
    return f"{n:.0f} {singular if round(n) == 1 else plural}"


def _estilo(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": TINTA_2, "axes.linewidth": 0.6, "xtick.color": TINTA_2,
        "ytick.color": TINTA_2, "text.color": TINTA, "axes.labelcolor": TINTA,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def figura_dados(dir_gabaritos: Path, saida: Path) -> list[dict[str, float]]:
    """Largura máxima da placa de cada veículo, por vídeo, legível ou difícil.

    Uma linha por vídeo (pequenos múltiplos com o mesmo eixo x); a linha tracejada é o
    escopo da avaliação e a pontilhada, o mínimo do OCR em tempo real naquele vídeo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _estilo(plt)
    fig, eixos = plt.subplots(len(VIDEOS), 1, figsize=(3.35, 2.75), sharex=True)
    resumos = _paineis_dados(eixos, dir_gabaritos)
    alcas, nomes = _legenda_dados(eixos[0])
    fig.legend(alcas, nomes, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.3, columnspacing=1.2, handlelength=1.8)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.22, hspace=0.95)
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return resumos


def figura_dados_janela(dir_gabaritos: Path, saida: Path) -> list[dict[str, float]]:
    """Figura 2 do artigo, em largura total: (a) os dados, (b) a integral, (c) o teto previsto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _estilo(plt)
    fig = plt.figure(figsize=(7.1, 2.3))
    gs = fig.add_gridspec(len(VIDEOS), 3, width_ratios=[1.2, 0.85, 1.05], wspace=0.42, hspace=1.05)
    eixos = [fig.add_subplot(gs[i, 0]) for i in range(len(VIDEOS))]
    for ax in eixos[1:]:
        ax.sharex(eixos[0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])
    _paineis_dados(eixos, dir_gabaritos, fonte=6.0, compacto=True)
    alcas, nomes = _legenda_dados(eixos[0])
    eixos[-1].legend(alcas, nomes, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -1.75),
                     handletextpad=0.3, columnspacing=1.0, handlelength=1.8, fontsize=5.8)
    _painel_exemplo(ax_b, dir_gabaritos, compacto=True)
    tabelas = _painel_teto(ax_c, dir_gabaritos, compacto=True)
    eixos[0].set_title("(a)", loc="left", fontsize=7.5, fontweight="bold", pad=10)
    ax_b.set_title("(b)", loc="left", fontsize=7.5, fontweight="bold")
    ax_c.set_title("(c)", loc="left", fontsize=7.5, fontweight="bold")
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return tabelas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("figura", choices=["dados", "janela", "fig2"])
    ap.add_argument("--gabaritos", type=Path, default=RAIZ / "data" / "videos" / "gabaritos")
    ap.add_argument("--saida", type=Path, default=SAIDA)
    args = ap.parse_args(argv)
    if args.figura == "dados":
        for r in figura_dados(args.gabaritos, args.saida / "dados.pdf"):
            print(f"{r['video']}: {r['veiculos']:.0f} veículos, {r['legiveis']:.0f} legíveis, "
                  f"{r['dificeis']:.0f} difíceis; OCR em tempo real a partir de "
                  f"{r['ocr_min_px']:.0f} px: {r['legiveis_ate_ocr']:.0f} legíveis chegam lá "
                  f"({100 * r['frac_legiveis_ate_ocr']:.0f}%)")
        print(f"gravado: {args.saida / 'dados.pdf'}")
    if args.figura == "janela":
        for r in figura_janela(args.gabaritos, args.saida / "janela.pdf"):
            tetos = ", ".join(f"{res} px → {100 * r[f'teto_{res}']:.0f}% (limiar {r[f'limiar_{res}']:.0f} px)"
                              for res in RESOLUCOES)
            print(f"{r['video']} ({r['legiveis']} legíveis): teto {tetos}; janela mediana a 1280 px "
                  f"{r['janela_mediana_1280']:.2f} s")
        print(f"gravado: {args.saida / 'janela.pdf'}")
    if args.figura == "fig2":
        figura_dados_janela(args.gabaritos, args.saida / "dados_janela.pdf")
        print(f"gravado: {args.saida / 'dados_janela.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
