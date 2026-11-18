"""Tabelas e figuras descritivas para o artigo, a partir dos bancos de sessoes.

Sem ground truth nao ha acuracia: o que sai daqui e o que o sistema MEDIU -
composicao do trafego, o funil das placas (veiculo -> placa detectada ->
tentativa de OCR -> leitura valida -> decisao publicada), o OCR por tamanho de
placa, o custo de cada estagio e o cumprimento do prazo de tempo real. Para
acuracia, ``lprts eval`` com anotacoes.

Nenhum texto de placa sai do banco: as tabelas tem so contagens e taxas (placa
e dado pessoal, LGPD), e podem ir para o artigo e para o repositorio de dados.

Objetos sao contados por **trilha confirmada com permanencia minima**
(``min_dwell_s``). Uma trilha que dura uma fracao de segundo e, na maioria das
vezes, um fragmento: o mesmo veiculo perdido numa oclusao e retomado como
objeto novo. As duas contagens (todas as trilhas e as confirmadas) saem lado a
lado, para que a escolha do limiar fique explicita.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.taxonomy import ObjectCategory, info
from ..utils.logging import get_logger

log = get_logger(__name__)

# Classes cujas trilhas entram no funil das placas (no modo so-placa a trilha e
# a propria placa).
CLASSES_COM_PLACA = ("carro", "motocicleta", "onibus", "caminhao", "placa_veicular")

# Faixas de largura da placa, em pixels da imagem PROCESSADA (o que o OCR ve).
FAIXAS_LARGURA = (0, 40, 60, 80, 100, 140, 10_000)

ESTAGIOS = (("detect_ms", "deteccao"), ("track_ms", "rastreio"), ("ocr_ms", "OCR"),
            ("analytics_ms", "analitica"))

ORDEM_CATEGORIAS = (
    ObjectCategory.PESSOA, ObjectCategory.VEICULO_MOTORIZADO,
    ObjectCategory.VEICULO_NAO_MOTORIZADO, ObjectCategory.ANIMAL,
    ObjectCategory.MOBILIARIO_URBANO, ObjectCategory.OUTRO,
)
ROTULO_CATEGORIA = {
    ObjectCategory.PESSOA: "pessoa",
    ObjectCategory.VEICULO_MOTORIZADO: "veiculo motorizado",
    ObjectCategory.VEICULO_NAO_MOTORIZADO: "veiculo nao motorizado",
    ObjectCategory.ANIMAL: "animal",
    ObjectCategory.MOBILIARIO_URBANO: "mobiliario urbano",
    ObjectCategory.OUTRO: "outro",
}


@dataclass
class StudyData:
    """As tabelas do relatorio, uma lista de linhas (dicionarios) cada."""

    videos: list[dict[str, Any]] = field(default_factory=list)
    desempenho: list[dict[str, Any]] = field(default_factory=list)
    objetos: list[dict[str, Any]] = field(default_factory=list)
    funil: list[dict[str, Any]] = field(default_factory=list)
    ocr_largura: list[dict[str, Any]] = field(default_factory=list)
    latencias: dict[str, np.ndarray] = field(default_factory=dict)   # rotulo -> ms


def _pct(valores: list[float] | np.ndarray, q: float) -> float | None:
    v = np.asarray([x for x in valores if x is not None], dtype=float)
    return round(float(np.percentile(v, q)), 2) if v.size else None


def _rotulo(video: str, sid: str, repetido: bool) -> str:
    return f"{video} ({sid[:6]})" if repetido else video


def _sessoes(con: sqlite3.Connection, pedidas: list[str] | None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM sessions"
    params: tuple = ()
    if pedidas:
        sql += f" WHERE session_id IN ({','.join('?' * len(pedidas))})"
        params = tuple(pedidas)
    return list(con.execute(sql + " ORDER BY started_at", params))


def collect(
    db_paths: Sequence[str | Path], sessions: list[str] | None = None, min_dwell_s: float = 0.5,
) -> StudyData:
    """Le as sessoes dos bancos e monta as tabelas."""
    dados = StudyData()
    linhas_sessao: list[tuple[Path, sqlite3.Row]] = []
    for db in db_paths:
        db = Path(db)
        if not db.exists():
            raise FileNotFoundError(f"Banco nao encontrado: {db}")
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
            con.row_factory = sqlite3.Row
            linhas_sessao += [(db, s) for s in _sessoes(con, sessions)]
    if not linhas_sessao:
        raise ValueError("Nenhuma sessao encontrada nos bancos informados.")
    nomes = [Path(s["source_uri"]).name for _, s in linhas_sessao]
    for db, s in linhas_sessao:
        video = Path(s["source_uri"]).name
        rotulo = _rotulo(video, s["session_id"], nomes.count(video) > 1)
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
            con.row_factory = sqlite3.Row
            _coletar_sessao(con, s, rotulo, min_dwell_s, dados)
    return dados


def _coletar_sessao(
    con: sqlite3.Connection, s: sqlite3.Row, rotulo: str, min_dwell_s: float, dados: StudyData,
) -> None:
    sid = s["session_id"]
    cfg = json.loads(s["config_json"] or "{}")
    env = json.loads(s["environment_json"] or "{}")
    quadros = con.execute(
        "SELECT video_time_s, latency_ms, total_ms, capture_ms, queue_ms, detect_ms, track_ms, "
        "ocr_ms, analytics_ms, deadline_miss FROM frames WHERE session_id=?", (sid,),
    ).fetchall()
    tempos = [q["video_time_s"] for q in quadros if q["video_time_s"] is not None]
    fps = float(s["source_fps"] or 0) or 30.0
    duracao_s = (max(tempos) - min(tempos) + 1 / fps) if tempos else 0.0
    perf = con.execute(
        "SELECT * FROM v_session_performance WHERE session_id=?", (sid,)
    ).fetchone()
    escala = (s["processing_width"] / s["source_width"]
              if s["processing_width"] and s["source_width"] else 1.0)

    ocr = s["ocr_engine"] or ""
    if ocr == "fastplate":
        ocr += f" ({cfg.get('ocr', {}).get('model_name', '?')})"
    comp = env.get("components") or {}

    def componente(chave: str) -> str:
        c = comp.get(chave) or {}
        if not c:
            return "-"
        name = str(c.get("name") or "")
        nome = {"plate": "yolo", "classic_plate": "classico"}.get(name, name)
        pesos = Path(c["weights"]).name if c.get("weights") else ""
        return f"{nome} {pesos} [{c.get('runtime')}@{c.get('device')}]".replace("  ", " ")
    dados.videos.append({
        "video": rotulo, "sessao": sid, "inicio_utc": s["started_at"],
        "fonte": s["source_kind"], "modo": "tempo real" if s["realtime"] else "offline",
        "resolucao": f"{s['source_width']}x{s['source_height']}",
        "largura_processada": s["processing_width"], "fps": round(fps, 2),
        "quadros_fonte": s["frame_count"], "quadros_processados": len(quadros),
        "duracao_s": round(duracao_s, 1),
        "objetos": cfg.get("detector", {}).get("classes", ""),
        "detector_objetos": componente("object_detector"),
        "detector_placa": componente("plate_detector"), "ocr": ocr,
        "leitura_alta_res": int(bool(cfg.get("ocr", {}).get("full_resolution", False))),
        "placa_min_px": cfg.get("ocr", {}).get("min_plate_width"),
        "processador": (env.get("hardware") or {}).get("cpu_model") or env.get("processor"),
        "energia": (env.get("power") or {}).get("power_source"),
        "codigo": s["code_version"], "config_hash": s["config_hash"],
    })

    linha_perf: dict[str, Any] = {
        "video": rotulo, "modo": "tempo real" if s["realtime"] else "offline",
        "vazao_qps": round(perf["throughput_fps"], 1) if perf and perf["throughput_fps"] else None,
        "prazo_ms": round(s["deadline_ms"], 1) if s["deadline_ms"] else None,
        "latencia_p50_ms": _pct([q["latency_ms"] for q in quadros], 50),
        "latencia_p95_ms": _pct([q["latency_ms"] for q in quadros], 95),
        "latencia_p99_ms": _pct([q["latency_ms"] for q in quadros], 99),
        "latencia_max_ms": _pct([q["latency_ms"] for q in quadros], 100),
        "prazo_perdido_pct": (round(100 * perf["deadline_miss_rate"], 2)
                              if perf and perf["deadline_miss_rate"] is not None else None),
        "descarte_pct": (round(100 * perf["drop_rate"], 2)
                         if perf and perf["drop_rate"] is not None else None),
    }
    for coluna, nome in (("capture_ms", "captura"), ("queue_ms", "fila"), *ESTAGIOS,
                         ("total_ms", "servico")):
        valores = [q[coluna] for q in quadros]
        linha_perf[f"{nome}_p50_ms"] = _pct(valores, 50)
        linha_perf[f"{nome}_p95_ms"] = _pct(valores, 95)
        v = [x for x in valores if x is not None]
        linha_perf[f"{nome}_media_ms"] = round(float(np.mean(v)), 2) if v else None
    dados.desempenho.append(linha_perf)
    dados.latencias[rotulo] = np.asarray(
        [q["latency_ms"] for q in quadros if q["latency_ms"]], dtype=float)

    # -- objetos -------------------------------------------------------------
    trilhas = con.execute(
        "SELECT track_id, object_class, dwell_s, published, first_seen_s, published_time_s "
        "FROM tracks WHERE session_id=? AND end_reason<>''", (sid,),
    ).fetchall()
    por_classe: dict[str, list[sqlite3.Row]] = {}
    for t in trilhas:
        por_classe.setdefault(t["object_class"], []).append(t)
    minutos = duracao_s / 60 if duracao_s > 0 else None
    for classe, ts in sorted(por_classe.items(), key=lambda kv: -len(kv[1])):
        conf = [t for t in ts if (t["dwell_s"] or 0) >= min_dwell_s]
        ci = info(classe)
        dados.objetos.append({
            "video": rotulo, "classe": classe, "categoria": ci.category.value,
            "vulneravel": int(ci.vulnerable), "trilhas": len(ts), "confirmadas": len(conf),
            "por_minuto": round(len(conf) / minutos, 2) if minutos else None,
            "permanencia_mediana_s": _pct([t["dwell_s"] for t in conf], 50),
        })

    # -- funil das placas ------------------------------------------------------
    com_placa = {r[0] for r in con.execute(
        "SELECT DISTINCT track_id FROM track_points WHERE session_id=? AND plate_x1 IS NOT NULL",
        (sid,))}
    leituras: dict[int, list[sqlite3.Row]] = {}
    for r in con.execute(
        "SELECT track_id, text, valid_syntax, plate_w, confidence, sharpness FROM reads "
        "WHERE session_id=?", (sid,)):
        leituras.setdefault(r["track_id"], []).append(r)
    decisoes = {r["track_id"]: r for r in con.execute(
        "SELECT track_id, n_reads FROM v_decisions WHERE session_id=?", (sid,))}
    for classe in CLASSES_COM_PLACA:
        ts = [t for t in por_classe.get(classe, []) if (t["dwell_s"] or 0) >= min_dwell_s]
        if not ts:
            continue
        ids = [t["track_id"] for t in ts]
        tentadas = [i for i in ids if leituras.get(i)]
        validas = [i for i in ids if any(r["valid_syntax"] for r in leituras.get(i, []))]
        publicadas = [t for t in ts if t["published"]]
        latencia_decisao = [t["published_time_s"] - t["first_seen_s"] for t in publicadas
                            if t["published_time_s"] is not None]
        dados.funil.append({
            "video": rotulo, "classe": classe, "veiculos": len(ts),
            "placa_detectada": sum(i in com_placa for i in ids),
            "com_tentativa_ocr": len(tentadas), "com_leitura_valida": len(validas),
            "publicadas": len(publicadas),
            "publicadas_pct": round(100 * len(publicadas) / len(ts), 1),
            "leituras_na_decisao_mediana": _pct(
                [decisoes[t["track_id"]]["n_reads"] for t in publicadas
                 if t["track_id"] in decisoes], 50),
            "tempo_ate_decisao_mediana_s": _pct(latencia_decisao, 50),
        })

    # -- OCR por largura da placa ---------------------------------------------
    todas = [r for rs in leituras.values() for r in rs if r["plate_w"]]
    for lo, hi in zip(FAIXAS_LARGURA[:-1], FAIXAS_LARGURA[1:], strict=True):
        faixa = [r for r in todas if lo <= r["plate_w"] * escala < hi]
        if not faixa:
            continue
        com_texto = [r for r in faixa if r["text"]]
        dados.ocr_largura.append({
            "video": rotulo,
            "largura_px": f"{lo}-{hi}" if hi < FAIXAS_LARGURA[-1] else f">={lo}",
            "tentativas": len(faixa),
            "com_texto_pct": round(100 * len(com_texto) / len(faixa), 1),
            "sintaxe_valida_pct": round(
                100 * sum(1 for r in faixa if r["valid_syntax"]) / len(faixa), 1),
            "confianca_media": round(float(np.mean([r["confidence"] for r in com_texto])), 3)
            if com_texto else None,
            "nitidez_mediana": _pct([r["sharpness"] for r in faixa], 50),
        })


# ---------------------------------------------------------------------------
# Saida
# ---------------------------------------------------------------------------

TABELAS = (
    ("videos", "Videos e condicoes de execucao"),
    ("objetos", "Objetos por classe (trilhas confirmadas)"),
    ("funil", "Funil das placas por classe de veiculo"),
    ("ocr_largura", "OCR por largura da placa (px da imagem processada)"),
    ("desempenho", "Desempenho e tempo real"),
)


def _csv(linhas: list[dict], caminho: Path) -> Path:
    with caminho.open("w", newline="", encoding="utf-8") as fh:
        if linhas:
            w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)
    return caminho


def _markdown(linhas: list[dict], colunas: list[str] | None = None) -> str:
    if not linhas:
        return "_sem dados_\n"
    colunas = colunas or list(linhas[0])
    def fmt(v: Any) -> str:
        return "-" if v is None else str(v)
    cab = "| " + " | ".join(colunas) + " |\n|" + "---|" * len(colunas) + "\n"
    return cab + "".join("| " + " | ".join(fmt(r.get(c)) for c in colunas) + " |\n"
                         for r in linhas)


_COLUNAS_MD = {
    "videos": ["video", "modo", "resolucao", "largura_processada", "duracao_s",
               "quadros_processados", "objetos", "detector_objetos", "detector_placa", "ocr",
               "leitura_alta_res", "placa_min_px", "processador", "energia", "codigo"],
    "desempenho": ["video", "modo", "vazao_qps", "prazo_ms", "latencia_p50_ms",
                   "latencia_p95_ms", "latencia_p99_ms", "latencia_max_ms", "prazo_perdido_pct",
                   "descarte_pct", "captura_p50_ms", "deteccao_p50_ms", "deteccao_p95_ms",
                   "OCR_media_ms", "servico_p95_ms"],
}


def write_tables(dados: StudyData, out_dir: Path, min_dwell_s: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arquivos = [_csv(getattr(dados, nome), out_dir / f"{nome}.csv") for nome, _ in TABELAS]
    partes = [
        "# Relatorio descritivo das sessoes\n\n",
        "Gerado por `lprts report` a partir dos bancos de resultados. Sao medidas do "
        "sistema, nao acuracia: sem ground truth nao se sabe quantas placas publicadas "
        "estao certas nem quantos objetos passaram de fato (use `lprts eval`). "
        "Nenhum texto de placa e exportado.\n\n",
        f"Objetos contados por trilha confirmada com permanencia >= {min_dwell_s:g} s; "
        "a coluna `trilhas` traz todas, inclusive fragmentos.\n",
    ]
    for nome, titulo in TABELAS:
        partes.append(f"\n## {titulo}\n\n")
        partes.append(_markdown(getattr(dados, nome), _COLUNAS_MD.get(nome)))
    md = out_dir / "relatorio.md"
    md.write_text("".join(partes), encoding="utf-8")
    return [*arquivos, md]


# Paleta de referencia (validada para daltonismo na ordem fixa) e tintas de texto.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
           "#e34948")
_TINTA, _TINTA_2, _GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def _eixo(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_TINTA_2)
    ax.tick_params(colors=_TINTA_2, labelsize=8)
    ax.grid(axis="x", color=_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)


def write_figures(dados: StudyData, out_dir: Path) -> list[Path]:
    """Figuras para o artigo (PDF vetorial + PNG). Exige matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib ausente: figuras nao geradas (pip install -e '.[analysis]').")
        return []
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelcolor": _TINTA_2,
                         "text.color": _TINTA, "pdf.fonttype": 42})
    saidas: list[Path] = []

    def salvar(fig, nome: str) -> None:
        for ext in ("pdf", "png"):
            caminho = out_dir / f"{nome}.{ext}"
            fig.savefig(caminho, dpi=200, bbox_inches="tight", facecolor="white")
            saidas.append(caminho)
        plt.close(fig)

    videos = [v["video"] for v in dados.videos]

    # 1. Composicao por categoria (barras empilhadas horizontais).
    if dados.objetos:
        fig, ax = plt.subplots(figsize=(6.5, 0.5 + 0.45 * len(videos)))
        esquerda = np.zeros(len(videos))
        presentes = [c for c in ORDEM_CATEGORIAS
                     if any(o["categoria"] == c.value for o in dados.objetos)]
        for i, cat in enumerate(presentes):
            vals = np.array([sum(o["confirmadas"] for o in dados.objetos
                                 if o["video"] == v and o["categoria"] == cat.value)
                             for v in videos], dtype=float)
            ax.barh(videos, vals, left=esquerda, color=_SERIES[i % len(_SERIES)],
                    edgecolor="white", linewidth=1.5, height=0.6, label=ROTULO_CATEGORIA[cat])
            esquerda += vals
        for y, total in enumerate(esquerda):
            ax.text(total, y, f" {int(total)}", va="center", fontsize=8, color=_TINTA)
        ax.set_xlabel("objetos unicos (trilhas confirmadas)")
        ax.invert_yaxis()
        _eixo(ax)
        ax.legend(frameon=False, fontsize=7, ncol=min(3, len(presentes)),
                  loc="upper center", bbox_to_anchor=(0.5, -0.35 if len(videos) < 3 else -0.15))
        salvar(fig, "fig_composicao")

    # 2. Distribuicao acumulada da latencia ponta a ponta, um painel por sessao.
    series = [(k, v) for k, v in dados.latencias.items() if v.size]
    if series:
        n = len(series)
        fig, eixos = plt.subplots(n, 1, figsize=(6.5, 1.3 * n + 0.4), sharex=True, squeeze=False)
        prazos = {d["video"]: d["prazo_ms"] for d in dados.desempenho}
        for ax, (rot, lat) in zip(eixos[:, 0], series, strict=True):
            x = np.sort(lat)
            ax.plot(x, np.arange(1, x.size + 1) / x.size, color=_SERIES[0], linewidth=2)
            if prazos.get(rot):
                ax.axvline(prazos[rot], color=_TINTA_2, linewidth=1, linestyle="--")
                ax.text(prazos[rot], 0.05, f" prazo {prazos[rot]:.0f} ms", fontsize=7,
                        color=_TINTA_2)
            ax.set_ylabel("fracao", fontsize=7)
            ax.set_title(rot, loc="left", fontsize=8, color=_TINTA)
            ax.set_ylim(0, 1.02)
            _eixo(ax)
            ax.grid(axis="y", color=_GRADE, linewidth=0.6)
        eixos[-1, 0].set_xlabel("latencia ponta a ponta (ms)")
        fig.subplots_adjust(hspace=0.55)
        salvar(fig, "fig_latencia")

    # 3. Tempo medio de servico por estagio (barras empilhadas).
    if dados.desempenho:
        fig, ax = plt.subplots(figsize=(6.5, 0.5 + 0.45 * len(videos)))
        esquerda = np.zeros(len(dados.desempenho))
        rotulos = [d["video"] for d in dados.desempenho]
        for i, (_, nome) in enumerate(ESTAGIOS):
            vals = np.array([d.get(f"{nome}_media_ms") or 0.0 for d in dados.desempenho])
            ax.barh(rotulos, vals, left=esquerda, color=_SERIES[i], edgecolor="white",
                    linewidth=1.5, height=0.6, label=nome)
            esquerda += vals
        ax.set_xlabel("tempo medio de servico por quadro (ms)")
        ax.invert_yaxis()
        _eixo(ax)
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.35 if len(rotulos) < 3 else -0.15))
        salvar(fig, "fig_estagios")

    # 4. Leitura valida por largura da placa, um painel por sessao.
    if dados.ocr_largura:
        com_ocr = [v for v in videos if any(o["video"] == v for o in dados.ocr_largura)]
        fig, eixos = plt.subplots(1, len(com_ocr), figsize=(3.2 * len(com_ocr) + 0.4, 2.4),
                                  sharey=True, squeeze=False)
        for ax, v in zip(eixos[0], com_ocr, strict=True):
            faixas = [o for o in dados.ocr_largura if o["video"] == v]
            ax.bar([f["largura_px"] for f in faixas], [f["sintaxe_valida_pct"] for f in faixas],
                   color=_SERIES[0], width=0.6)
            for bar_index, f in enumerate(faixas):
                ax.text(bar_index, f["sintaxe_valida_pct"], f"n={f['tentativas']}", ha="center",
                        va="bottom", fontsize=6, color=_TINTA_2)
            ax.set_title(v, loc="left", fontsize=8)
            ax.set_xlabel("largura da placa (px)")
            ax.tick_params(axis="x", labelrotation=45)
            _eixo(ax)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", color=_GRADE, linewidth=0.6)
        eixos[0, 0].set_ylabel("leituras com sintaxe valida (%)")
        salvar(fig, "fig_ocr_largura")
    return saidas


def build_report(
    db_paths: Sequence[str | Path], out_dir: str | Path, sessions: list[str] | None = None,
    min_dwell_s: float = 0.5, figures: bool = True,
) -> list[Path]:
    """Gera as tabelas (CSV + Markdown) e, se possivel, as figuras."""
    out_dir = Path(out_dir)
    dados = collect(db_paths, sessions, min_dwell_s)
    arquivos = write_tables(dados, out_dir, min_dwell_s)
    if figures:
        arquivos += write_figures(dados, out_dir)
    return arquivos
