#!/usr/bin/env python3
"""Análise da bateria do artigo (plano: docs/artigo/pipeline/plano_analise.md).

Uso::

    python scripts/artigo/analise.py            # números e as Figuras 3 a 5

Lê ``runs/artigo/V*/`` (bancos e relatórios da bateria) e ``data/videos/gabaritos/``,
ambos fora do git porque têm textos de placa. Grava só agregados, sem placa:
``docs/artigo/resultados/analise.json`` (todo número citado no texto) e
``docs/artigo/figuras/{tempo,otimizacoes,acuracia}.pdf``.

Sessões suspensas. O macOS pôs o computador para dormir durante algumas sessões da
bateria (registro do ``pmset``). O relógio monotônico do processo para durante o sono, o
de parede não: uma sessão cuja duração de parede passa a monotônica em mais de
``SUSPENSAO_S`` ficou suspensa e sai das análises de tempo. A acurácia por veículo não
depende do relógio e usa todas as sessões.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[2]
RUNS = RAIZ / "runs" / "artigo"
GABARITOS = RAIZ / "data" / "videos" / "gabaritos"
FIGURAS = RAIZ / "docs" / "artigo" / "figuras"
SAIDA_JSON = RAIZ / "docs" / "artigo" / "resultados" / "analise.json"

_spec = importlib.util.spec_from_file_location("figuras_artigo", Path(__file__).with_name("figuras.py"))
fig_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fig_base
_spec.loader.exec_module(fig_base)

VIDEOS = [v[0] for v in fig_base.VIDEOS]                 # V1..V4
NOMES = {v[0]: v[1] for v in fig_base.VIDEOS}
GT = {v[0]: v[2] for v in fig_base.VIDEOS}
LARGURA_VIDEO = {v[0]: v[3] for v in fig_base.VIDEOS}
FPS = 30.0
PRAZO_MS = 1000.0 / FPS
SUSPENSAO_S = 1.0
PORTA_PLACA = ("car", "motorcycle", "bus", "truck")
ETAPAS = [("capture_ms", "captura"), ("queue_ms", "fila"), ("detect_ms", "detecção"),
          ("track_ms", "rastreamento"), ("ocr_ms", "OCR"), ("analytics_ms", "gravação")]
RESOLUCOES = {"res960": 960, "base": 1280, "res1920": 1920}
# Otimização: (rótulo, condição sem a otimização, modos avaliados)
OTIMIZACOES = [
    ("O1", "Core ML direto", "o1_ultralytics", ("rt",)),
    ("O2", "placa só em veículo legível", "o2_todas_placas", ("off", "rt")),
    ("O3", "recorte do OCR no original", "o3_recorte_reduzido", ("off", "rt")),
    ("O4", "decodificação em hardware", "o4_ffmpeg", ("rt",)),
]
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # slots 1-4 da paleta, ordem fixa
MARCAS = ["o", "s", "^", "D"]
SEMENTE = 20261116
N_BOOT = 2000


# --------------------------------------------------------------------------- leitura

def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def sessoes(db: Path) -> pd.DataFrame:
    """Sessões do banco, com as durações de parede e monotônica e a marca de suspensão."""
    with _ro(db) as con:
        s = pd.read_sql("SELECT session_id, started_at, finished_at, frame_count, source_fps, "
                        "realtime, processing_width FROM sessions ORDER BY started_at", con)
        m = pd.read_sql("SELECT session_id, MIN(capture_t_s) AS t0, MAX(done_t_s) AS t1, "
                        "COUNT(*) AS n_quadros FROM frames GROUP BY session_id", con)
    s = s.merge(m, on="session_id")
    s["parede_s"] = [(datetime.fromisoformat(f) - datetime.fromisoformat(i)).total_seconds()
                     for i, f in zip(s.started_at, s.finished_at, strict=True)]
    s["mono_s"] = s.t1 - s.t0
    s["suspensa"] = (s.parede_s - s.mono_s) > SUSPENSAO_S
    return s


def quadros(db: Path, validas: set[str] | None = None) -> pd.DataFrame:
    cols = ("session_id, frame_index, capture_t_s, done_t_s, detection_ran, n_rois_skipped, "
            + ", ".join(c for c, _ in ETAPAS) + ", total_ms, latency_ms, deadline_miss, dropped_frames")
    with _ro(db) as con:
        q = pd.read_sql(f"SELECT {cols} FROM frames", con)
    return q if validas is None else q[q.session_id.isin(validas)]


def buscas(db: Path) -> pd.DataFrame:
    """Veículos com busca de placa por quadro: portadores de placa detectados menos os pulados."""
    marcas = ",".join(f"'{p}'" for p in PORTA_PLACA)
    with _ro(db) as con:
        d = pd.read_sql(f"SELECT session_id, frame_index, COUNT(*) AS portadores FROM detections "
                        f"WHERE kind='object' AND label IN ({marcas}) GROUP BY session_id, frame_index", con)
    return d


def relatorios(video: str, cond: str) -> list[dict]:
    reps = sorted((RUNS / video).glob(f"{cond}_rep*.json"))
    arqs = reps or [RUNS / video / f"{cond}.json"]
    return [json.loads(a.read_text(encoding="utf-8")) for a in arqs]


def veiculos(video: str, cond: str) -> pd.DataFrame:
    """Uma linha por veículo na métrica e por repetição (rep 0 no modo sem prazo)."""
    reps = sorted((RUNS / video).glob(f"{cond}_rep*_veiculos.csv"))
    arqs = reps or [RUNS / video / f"{cond}_veiculos.csv"]
    partes = []
    for i, a in enumerate(arqs):
        d = pd.read_csv(a)
        d = d[d.excluded_difficult == 0].copy()
        d["rep"] = i + 1 if reps else 0
        d["video"] = video
        partes.append(d[["video", "vehicle", "rep", "published_correct", "max_plate_width_px",
                         "first_frame", "last_frame", "time_to_correct_s"]])
    return pd.concat(partes, ignore_index=True)


def _validas(video: str, cond: str) -> set[str]:
    s = sessoes(RUNS / video / f"{cond}.db")
    return set(s.session_id[~s.suspensa])


# --------------------------------------------------------------------------- estatística

def bootstrap_ic(valores_por_estrato: list[np.ndarray], estat, n: int = N_BOOT, semente: int = SEMENTE):
    """IC de 95% por bootstrap estratificado (reamostra dentro de cada vídeo)."""
    rng = np.random.default_rng(semente)
    sim = []
    for _ in range(n):
        amostra = [v[rng.integers(0, len(v), len(v))] for v in valores_por_estrato]
        sim.append(estat(np.concatenate(amostra)))
    return float(np.percentile(sim, 2.5)), float(np.percentile(sim, 97.5))


def mcnemar_exato(com: np.ndarray, sem: np.ndarray) -> tuple[int, int, float]:
    """b = acertos só com a otimização; c = acertos só sem; p bilateral exato."""
    b = int(np.sum((com == 1) & (sem == 0)))
    c = int(np.sum((com == 0) & (sem == 1)))
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


def holm(ps: list[float]) -> list[float]:
    ordem = np.argsort(ps)
    m = len(ps)
    ajust = np.empty(m)
    acum = 0.0
    for k, i in enumerate(ordem):
        acum = max(acum, min(1.0, (m - k) * ps[i]))
        ajust[i] = acum
    return ajust.tolist()


def ols_bootstrap_blocos(x: np.ndarray, y: np.ndarray, sessao: np.ndarray, bloco: int = 300):
    """y = a + b·x por mínimos quadrados; IC de 95% reamostrando blocos de quadros (10 s)."""
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    ids = np.zeros(len(x), dtype=int)
    k = 0
    for s in np.unique(sessao):
        idx = np.flatnonzero(sessao == s)
        ids[idx] = k + np.arange(len(idx)) // bloco
        k = ids[idx].max() + 1
    grupos = [np.flatnonzero(ids == g) for g in range(k)]
    rng = np.random.default_rng(SEMENTE)
    sim = []
    for _ in range(N_BOOT // 4):
        sel = np.concatenate([grupos[g] for g in rng.integers(0, k, k)])
        sim.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0])
    sim = np.array(sim)
    lo, hi = np.percentile(sim, [2.5, 97.5], axis=0)
    return coef, lo, hi


def kaplan_meier(t: np.ndarray, evento: np.ndarray):
    """S(t) de produto-limite: tempos dos degraus e sobrevida logo depois de cada um."""
    ordem = np.argsort(t)
    t, evento = t[ordem], evento[ordem]
    tempos = np.unique(t[evento == 1])
    s, sob = 1.0, []
    for u in tempos:
        risco = np.sum(t >= u)
        d = np.sum((t == u) & (evento == 1))
        s *= 1.0 - d / risco
        sob.append(s)
    return tempos, np.array(sob)


def rmst(t: np.ndarray, evento: np.ndarray, tau: float) -> float:
    """Tempo médio restrito: ∫₀^τ S(t) dt, a área sob a curva de Kaplan–Meier até τ."""
    tempos, sob = kaplan_meier(t, evento)
    area, t_ant, s_ant = 0.0, 0.0, 1.0
    for u, s in zip(tempos, sob, strict=True):
        if u >= tau:
            break
        area += s_ant * (u - t_ant)
        t_ant, s_ant = u, s
    return area + s_ant * (tau - t_ant)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de 95% de Wilson para uma proporção k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def logistica(x: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Regressão logística P(y=1) = 1/(1+exp(-(b0+b1·x))) por Newton–Raphson."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p) + 1e-9
        passo = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(2), X.T @ (y - p))
        b += passo
        if np.max(np.abs(passo)) < 1e-8:
            break
    return b


# --------------------------------------------------------------------------- análises

def suspensoes() -> dict:
    out = {}
    for v in VIDEOS:
        for db in sorted((RUNS / v).glob("*.db")):
            s = sessoes(db)
            for _, r in s[s.suspensa].iterrows():
                out.setdefault(v, []).append({"condicao": db.stem, "sessao": r.session_id,
                                              "parede_s": round(r.parede_s, 1), "mono_s": round(r.mono_s, 1),
                                              "tempo_real": bool(r.realtime)})
    return out


def b1_etapas() -> dict:
    """Latência por etapa no perfil de tempo real (base_rt), quadros das sessões válidas."""
    out = {}
    for v in VIDEOS:
        q = quadros(RUNS / v / "base_rt.db", _validas(v, "base_rt"))
        linha = {}
        for c, nome in ETAPAS + [("total_ms", "serviço"), ("latency_ms", "ponta a ponta")]:
            x = q[c].dropna().to_numpy()
            linha[nome] = {"media": float(x.mean()), "p50": float(np.percentile(x, 50)),
                           "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}
        linha["prazo_perdido"] = float(q.deadline_miss.mean())
        linha["n_quadros"] = int(len(q))
        out[v] = linha
    return out


def b2_custo() -> dict:
    """Tempo de serviço por quadro = a + b·n, n = veículos com busca de placa (base_rt)."""
    out = {}
    for v in VIDEOS:
        db = RUNS / v / "base_rt.db"
        q = quadros(db, _validas(v, "base_rt"))
        q = q[q.detection_ran == 1].merge(buscas(db), on=["session_id", "frame_index"], how="left")
        q["n"] = (q.portadores.fillna(0) - q.n_rois_skipped.fillna(0)).clip(lower=0)
        coef, lo, hi = ols_bootstrap_blocos(q.n.to_numpy(float), q.total_ms.to_numpy(float),
                                            q.session_id.to_numpy())
        por_n = q.groupby("n").total_ms.agg(["mean", "count",
                                             lambda s: np.percentile(s, 10), lambda s: np.percentile(s, 90)])
        por_n.columns = ["media", "quadros", "p10", "p90"]
        bench = json.loads((RUNS / v / "bench_componentes.json").read_text(encoding="utf-8"))["results"]
        out[v] = {"a_ms": float(coef[0]), "a_ic": [float(lo[0]), float(hi[0])],
                  "b_ms": float(coef[1]), "b_ic": [float(lo[1]), float(hi[1])],
                  "n_medio": float(q.n.mean()),
                  "por_n": {int(k): {kk: float(vv) for kk, vv in r.items()} for k, r in por_n.iterrows()},
                  "rede_veiculos_ms": float(bench["objetos YOLO11n .mlpackage  CoreML direto"]["mean_ms"]),
                  "rede_placa_por_roi_ms": float(bench["placas plate_yolo imgsz=320  por ROI  CoreML direto"]["mean_ms"]),
                  "ocr_por_leitura_ms": float(bench["OCR fastplate  por leitura"]["mean_ms"])}
    return out


COND_RT = ["base_rt", "o1_ultralytics_rt", "o2_todas_placas_rt", "o3_recorte_reduzido_rt", "o4_ffmpeg_rt",
           "res960_rt", "res1920_rt"]


def b3_fila() -> dict:
    """Fila nas 28 condições de tempo real (7 condições × 4 vídeos).

    ρ = λ·E[S] (λ = vazão medida; S = tempo de serviço do processamento, ou o da captura).
    L = (1/T)∫N(t)dt é o número médio de quadros no sistema, da chegada à saída; pela lei
    de Little, L = λ·W. Logo L − ρ = λ·E[espera] é o número médio de quadros esperando.
    ``atraso`` é a fração do vídeo que uma câmera ao vivo perderia: com arquivo, a fonte
    atrasa em vez de descartar.
    """
    out = {}
    for v in VIDEOS:
        for cond in COND_RT:
            validas = _validas(v, cond)
            q = quadros(RUNS / v / f"{cond}.db", validas)
            s = sessoes(RUNS / v / f"{cond}.db")
            s = s[s.session_id.isin(validas)]
            duracao_video = (s.frame_count / s.source_fps).sum()
            T = s.mono_s.sum()
            lam = len(q) / T
            L = (q.latency_ms / 1000).sum() / T
            rho = lam * q.total_ms.mean() / 1000
            out[f"{v}_{cond}"] = {
                "video": v, "condicao": cond,
                "resolucao": {"res960_rt": 960, "res1920_rt": 1920}.get(cond, 1280),
                "rho_captura": float(lam * q.capture_ms.mean() / 1000), "rho_processamento": float(rho),
                "L_quadros": float(L), "esperando": float(L - rho),
                "espera_media_ms": float(q.queue_ms.mean()),
                "prazo_perdido": float(q.deadline_miss.mean()),
                "descarte": float(np.mean([r["performance"]["drop_rate"] for r in relatorios(v, cond)
                                           if r["session_id"] in validas])),
                "vazao_fps": float(lam), "atraso": float(max(0.0, 1 - duracao_video / T)),
                "latencia_p99_ms": float(np.percentile(q.latency_ms, 99)),
                "sessoes": int(len(s)),
            }
    return out


def _por_veiculo(video: str, cond: str) -> pd.DataFrame:
    """Acerto por veículo; em tempo real, a fração das repetições com a placa correta."""
    d = veiculos(video, cond)
    return d.groupby("vehicle").published_correct.mean().rename(cond)


def _erradas_por_veiculo(video: str, cond: str) -> pd.Series:
    """Placa publicada errada por veículo na métrica; em tempo real, na maioria das repetições.

    Só lê as colunas sem texto de placa.
    """
    reps = sorted((RUNS / video).glob(f"{cond}_rep*_veiculos.csv"))
    arqs = reps or [RUNS / video / f"{cond}_veiculos.csv"]
    d = pd.concat([pd.read_csv(a, usecols=["vehicle", "excluded_difficult", "published", "published_correct"])
                   for a in arqs], ignore_index=True)
    d = d[d.excluded_difficult == 0]
    errada = ((d.published == 1) & (d.published_correct == 0)).astype(float)
    return (errada.groupby(d.vehicle).mean() > 0.5).rename(cond)


def faixa_entre_repeticoes(com: list[float], sem: list[float]) -> list[float]:
    """Menor e maior diferença com − sem que as repetições permitem (pior caso dos dois lados)."""
    return [float(min(com) - max(sem)), float(max(com) - min(sem))]


def c_otimizacoes() -> dict:
    linhas = []
    for rot, nome, sem, modos in OTIMIZACOES:
        for modo in modos:
            com_c, sem_c = f"base_{modo}", f"{sem}_{modo}"
            estratos, todos_com, todos_sem, por_video = [], [], [], {}
            err_com, err_sem = [], []
            for v in VIDEOS:
                par = pd.concat([_por_veiculo(v, com_c), _por_veiculo(v, sem_c)], axis=1).dropna()
                dif = (par[com_c] - par[sem_c]).to_numpy()
                estratos.append(dif)
                todos_com.append((par[com_c] >= 0.5).astype(int).to_numpy())
                todos_sem.append((par[sem_c] >= 0.5).astype(int).to_numpy())
                # placas publicadas erradas, pareadas pelos mesmos veículos (revisão do Stage 3)
                par_e = pd.concat([_erradas_por_veiculo(v, com_c), _erradas_por_veiculo(v, sem_c)], axis=1)
                par_e = par_e.dropna().astype(bool)
                err_com.append(par_e[com_c].astype(int).to_numpy())
                err_sem.append(par_e[sem_c].astype(int).to_numpy())
                lat, p99s, servs = {}, {}, {}
                if modo == "rt":
                    for c in (com_c, sem_c):
                        rs = [r for r in relatorios(v, c) if r["session_id"] in _validas(v, c)]
                        p99s[c] = [float(r["performance"]["latency_p99_ms"]) for r in rs]
                        # serviço médio de cada repetição (o mesmo serviço da b3_fila, por sessão)
                        q = quadros(RUNS / v / f"{c}.db", _validas(v, c))
                        servs[c] = [float(x) for x in q.groupby("session_id").total_ms.mean().sort_values()]
                        # mediana das repetições: robusta a um soluço isolado numa delas
                        lat[c] = (float(np.median(p99s[c])),
                                  float(np.median([r["performance"]["deadline_miss_rate"] for r in rs])),
                                  float(np.median([r["performance"]["drop_rate"] for r in rs])), len(rs))
                por_video[v] = {
                    "delta_revocacao": float(dif.mean()), "n": int(len(dif)),
                    "erradas_com": int(par_e[com_c].sum()), "erradas_sem": int(par_e[sem_c].sum()),
                    **({"delta_p99_ms": lat[com_c][0] - lat[sem_c][0],
                        "delta_prazo_pp": 100 * (lat[com_c][1] - lat[sem_c][1]),
                        "delta_descarte_pp": 100 * (lat[com_c][2] - lat[sem_c][2]),
                        "sessoes_validas": [lat[com_c][3], lat[sem_c][3]],
                        "p99_reps_com_ms": p99s[com_c], "p99_reps_sem_ms": p99s[sem_c],
                        "delta_p99_faixa_ms": faixa_entre_repeticoes(p99s[com_c], p99s[sem_c]),
                        "servico_reps_com_ms": servs[com_c], "servico_reps_sem_ms": servs[sem_c]}
                       if lat else {}),
                }
            dif_all = np.concatenate(estratos)
            b, c, p = mcnemar_exato(np.concatenate(todos_com), np.concatenate(todos_sem))
            eb, ec, ep = mcnemar_exato(np.concatenate(err_com), np.concatenate(err_sem))
            linhas.append({"otimizacao": rot, "nome": nome, "modo": modo,
                           "delta_revocacao": float(dif_all.mean()),
                           "ic95": bootstrap_ic(estratos, np.mean),
                           "mcnemar_b": b, "mcnemar_c": c, "p": p, "n_veiculos": int(len(dif_all)),
                           "erradas_com": int(np.concatenate(err_com).sum()),
                           "erradas_sem": int(np.concatenate(err_sem).sum()),
                           "erradas_mcnemar_b": eb, "erradas_mcnemar_c": ec, "erradas_p": ep,
                           "por_video": por_video})
    for linha, p_adj in zip(linhas, holm([lin["p"] for lin in linhas]), strict=True):
        linha["p_holm"] = p_adj
    return {"linhas": linhas}


def f_exploratorias() -> dict:
    """Análises exploratórias da revisão (fora do plano; plano_analise.md, Desvios e incidentes).

    A detecção por quadro, em média, sem prazo (as chamadas se seguem, a CPU fica ocupada) contra
    tempo real com decodificação em hardware (o processo espera entre os quadros) e com
    decodificação por software (a decodificação mantém a CPU ocupada). Se o processador desacelera
    quando fica ocioso entre os quadros, a razão é maior que 1 em hardware e perto de 1 em
    software. Sessões suspensas ficam de fora; um vídeo sem sessão sem prazo válida fica sem razão.
    """
    desac = {}
    for v in VIDEOS:
        t, reps = {}, {}
        for cond in ("base_off", "base_rt", "o4_ffmpeg_rt"):
            q = quadros(RUNS / v / f"{cond}.db", _validas(v, cond))
            q = q[q.detection_ran == 1]
            t[cond] = float(q["detect_ms"].mean()) if len(q) else None
            # média de cada repetição (o modo sem prazo tem uma sessão só)
            reps[cond] = [float(x) for x in q.groupby("session_id").detect_ms.mean().sort_values()]
        off = t["base_off"]
        desac[v] = {"deteccao_sem_prazo_ms": off, "deteccao_tempo_real_ms": t["base_rt"],
                    "deteccao_tempo_real_software_ms": t["o4_ffmpeg_rt"],
                    "razao_hardware": t["base_rt"] / off if off else None,
                    "razao_software": t["o4_ffmpeg_rt"] / off if off else None,
                    "razao_hardware_reps": [x / off for x in reps["base_rt"]] if off else None,
                    "razao_software_reps": [x / off for x in reps["o4_ffmpeg_rt"]] if off else None}
    return {"desaceleracao": desac}


def d_temporeal(tau: float = 3.0) -> dict:
    """Mesmo perfil sem prazo e em tempo real: revocação e tempo até a decisão correta."""
    pares = [("base", "1280 px"), ("o2_todas_placas", "O2 desligada"), ("o3_recorte_reduzido", "O3 desligada"),
             ("res960", "960 px"), ("res1920", "1920 px")]
    pontos = []
    for v in VIDEOS:
        for pref, rot in pares:
            off = relatorios(v, f"{pref}_off")[0]
            rts = relatorios(v, f"{pref}_rt")
            validas = _validas(v, f"{pref}_rt")
            rt_val = [r for r in rts if r["session_id"] in validas] or rts
            pontos.append({"video": v, "condicao": rot,
                           "revocacao_off": off["vehicle_level"]["recall"],
                           "revocacao_rt": [r["vehicle_level"]["recall"] for r in rts],
                           "prazo_perdido": float(np.mean([r["performance"]["deadline_miss_rate"] for r in rt_val])),
                           "descarte": float(np.mean([r["performance"]["drop_rate"] for r in rt_val]))})
    # Kaplan–Meier do tempo (de vídeo) até a placa correta, perfil de 1280 px, os 4 vídeos.
    # Uma observação por veículo: em tempo real, acerto na maioria das repetições e a
    # mediana do tempo entre as que acertaram. Bootstrap pareado (os mesmos veículos nos
    # dois modos), estratificado por vídeo.
    por_video = []
    for v in VIDEOS:
        d_off = veiculos(v, "base_off").set_index("vehicle")
        d_rt = veiculos(v, "base_rt")
        g = d_rt.groupby("vehicle")
        rt = pd.DataFrame({"acerto": g.published_correct.mean() >= 0.5,
                           "t": g.apply(lambda x: x.loc[x.published_correct == 1, "time_to_correct_s"].median(),
                                        include_groups=False)})
        cena = (d_off.last_frame - d_off.first_frame + 1) / FPS
        tab = pd.DataFrame({
            "t_off": np.where(d_off.published_correct == 1, d_off.time_to_correct_s.fillna(cena), cena),
            "e_off": (d_off.published_correct == 1).astype(int),
        }, index=d_off.index).join(rt)
        tab["t_rt"] = np.where(tab.acerto, tab.t.fillna(cena), cena)
        tab["e_rt"] = tab.acerto.astype(int)
        por_video.append(tab[["t_off", "e_off", "t_rt", "e_rt"]].clip(lower=0).to_numpy(float))
    tudo = np.concatenate(por_video)
    curvas = {}
    for modo, (it, ie) in {"off": (0, 1), "rt": (2, 3)}.items():
        tempos, sob = kaplan_meier(tudo[:, it], tudo[:, ie].astype(int))
        curvas[modo] = {"t": tempos.tolist(), "S": sob.tolist(),
                        "rmst": rmst(tudo[:, it], tudo[:, ie].astype(int), tau),
                        "n": int(len(tudo)), "eventos": int(tudo[:, ie].sum())}
    rng = np.random.default_rng(SEMENTE)
    difs = []
    for _ in range(N_BOOT // 2):
        a = np.concatenate([m[rng.integers(0, len(m), len(m))] for m in por_video])
        difs.append(rmst(a[:, 2], a[:, 3].astype(int), tau) - rmst(a[:, 0], a[:, 1].astype(int), tau))
    return {"pontos": pontos, "km": curvas, "tau_s": tau,
            "rmst_dif_s": curvas["rt"]["rmst"] - curvas["off"]["rmst"],
            "rmst_dif_ic": [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))]}


def e_largura() -> dict:
    """P(placa correta | maior largura) por resolução, sem prazo, e o teto previsto x observado."""
    # razão entre a largura da placa detectada (onde o OCR decide) e a anotada, nas leituras
    razoes = []
    for v in VIDEOS:
        gt = pd.read_csv(GABARITOS / GT[v])
        gt = gt[gt.kind == "plate"].assign(w=lambda g: g.x2 - g.x1, vehicle=lambda g: "id:" + g.track_id.astype(str))
        le = pd.read_csv(RUNS / v / "base_off_leituras.csv")
        m = le.merge(gt[["frame_index", "vehicle", "w"]], left_on=["frame_index", "gt_vehicle"],
                     right_on=["frame_index", "vehicle"])
        razoes.append((m.plate_w / m.w).to_numpy())
    razao = float(np.median(np.concatenate(razoes)))
    curvas = {}
    for pref, res in RESOLUCOES.items():
        d = pd.concat([veiculos(v, f"{pref}_off") for v in VIDEOS if LARGURA_VIDEO[v] == 3840])
        x = np.log2(d.max_plate_width_px.to_numpy(float))
        y = d.published_correct.to_numpy(float)
        b = logistica(x, y)
        grade = np.linspace(np.log2(40), np.log2(400), 120)
        rng = np.random.default_rng(SEMENTE)
        sim = []
        for _ in range(400):
            i = rng.integers(0, len(x), len(x))
            bb = logistica(x[i], y[i])
            sim.append(1 / (1 + np.exp(-np.clip(bb[0] + bb[1] * grade, -50, 50))))
        sim = np.array(sim)
        # Degrau: abaixo do mínimo do OCR (na largura anotada equivalente) quase nada é lido;
        # acima, o acerto fica num patamar. A logística padrão força a assíntota em 1.
        trava = fig_base.limiar_ocr(3840, res) / razao
        w = d.max_plate_width_px.to_numpy(float)
        abaixo, acima = y[w < trava], y[w >= trava]
        bordas = np.array([40, 60, 80, 100, 125, 150, 180, 220, 270, 330, 420], float)
        faixas = []
        for lo, hi in zip(bordas[:-1], bordas[1:], strict=True):
            sel = (w >= lo) & (w < hi)
            if sel.sum() >= 3:
                k, n = int(y[sel].sum()), int(sel.sum())
                faixas.append({"centro_px": float(np.sqrt(lo * hi)), "k": k, "n": n, "p": k / n,
                               "ic": wilson(k, n)})
        degrau = {"trava_px": float(trava), "abaixo_k": int(abaixo.sum()), "abaixo_n": int(len(abaixo)),
                  "acima_k": int(acima.sum()), "acima_n": int(len(acima)),
                  "patamar": float(acima.mean()), "patamar_ic": wilson(int(acima.sum()), int(len(acima))),
                  "faixas": faixas}
        curvas[res] = {"degrau": degrau,
                       "b0": float(b[0]), "b1": float(b[1]), "w50_px": float(2 ** (-b[0] / b[1])),
                       "grade_px": (2 ** grade).tolist(),
                       "p": (1 / (1 + np.exp(-(b[0] + b[1] * grade)))).tolist(),
                       "p_lo": np.percentile(sim, 2.5, axis=0).tolist(),
                       "p_hi": np.percentile(sim, 97.5, axis=0).tolist(),
                       "x_px": d.max_plate_width_px.tolist(), "y": y.tolist(), "n": int(len(y))}
    # teto: registrado (6 quadros, largura anotada) e corrigido (5 quadros, largura detectada)
    tetos = []
    for v in VIDEOS:
        vs = fig_base.larguras_por_veiculo(GABARITOS / GT[v])
        for pref, res in RESOLUCOES.items():
            lim = fig_base.limiar_ocr(LARGURA_VIDEO[v], res)
            reg = fig_base.teto(vs, lim)
            corr = sum(1 for w in vs.values() if sum(1 for _, x in w if x * razao >= lim) >= 5) / len(vs)
            obs = relatorios(v, f"{pref}_off")[0]["vehicle_level"]["recall"]
            tetos.append({"video": v, "resolucao": res, "teto_registrado": reg, "teto_corrigido": corr,
                          "revocacao": obs})
    return {"razao_detectada_anotada": razao, "curvas": curvas, "tetos": tetos}


# --------------------------------------------------------------------------- figuras

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_base._estilo(plt)
    return plt


def _virgula(x: float, _pos=None) -> str:
    """Rótulo de eixo com vírgula decimal (pt-BR)."""
    return f"{x:g}".replace(".", ",").replace("-", "−")


def _eixo(ax, grade: str = "x", formatar: str = "xy") -> None:
    from matplotlib.ticker import FuncFormatter
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    if grade:
        ax.grid(axis=grade, color=fig_base.GRADE, linewidth=0.5)
    ax.set_axisbelow(True)
    for nome, eixo in (("x", ax.xaxis), ("y", ax.yaxis)):
        if nome in formatar and eixo.get_scale() == "linear":
            eixo.set_major_formatter(FuncFormatter(_virgula))


def _salvar(fig, plt, saida: Path) -> None:
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figura_tempo(b1: dict, b2: dict, b3: dict, saida: Path) -> None:
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2), gridspec_kw={"wspace": 0.45, "width_ratios": [1.2, 1, 1]})
    # (a) composição do tempo médio por quadro e o p99 ponta a ponta, perfil de 1280 px
    ax = axs[0]
    grupos = [("captura", ["captura"], "#9ec5f0"), ("fila", ["fila"], "#5a9be3"),
              ("detecção", ["detecção"], "#1f5fae"), ("rastreio, OCR e gravação", ["rastreamento", "OCR", "gravação"], "#0b2c55")]
    for i, v in enumerate(VIDEOS[::-1]):
        esq = 0.0
        for nome, partes, tom in grupos:
            w = sum(b1[v][p]["media"] for p in partes)
            ax.barh(i, w, left=esq, color=tom, height=0.6, edgecolor="white", linewidth=0.6,
                    label=nome if i == 0 else None)
            esq += w
        ax.plot(b1[v]["ponta a ponta"]["p99"], i, marker="|", color=fig_base.TINTA, markersize=9, mew=1.3,
                label="p99 ponta a ponta" if i == 0 else None)
    ax.axvline(PRAZO_MS, color=fig_base.LARANJA, ls="--", lw=0.9, label="prazo (33 ms)")
    ax.set_yticks(range(len(VIDEOS)), VIDEOS[::-1])

    ax.set_xlabel("tempo médio por quadro (ms)")
    ax.set_title("(a) onde vai o tempo, 1280 px", loc="left", fontsize=7)
    ax.set_xlim(0, 42)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.24), frameon=False, fontsize=5.4, ncol=3,
              handlelength=1.1, columnspacing=0.8)
    _eixo(ax, formatar="x")
    ax.set_yticks(range(len(VIDEOS)), VIDEOS[::-1])
    # (b) tempo de serviço por número de veículos com busca de placa (média e p10–p90)
    ax = axs[1]
    for v, cor, marca in zip(VIDEOS, CORES, MARCAS, strict=True):
        pn = b2[v]["por_n"]
        ns = sorted(k for k, r in pn.items() if r["quadros"] >= 50)
        ax.errorbar(ns, [pn[k]["media"] for k in ns],
                    yerr=[[pn[k]["media"] - pn[k]["p10"] for k in ns], [pn[k]["p90"] - pn[k]["media"] for k in ns]],
                    fmt=marca, color=cor, ms=3.2, lw=0.6, capsize=0, mec="white", mew=0.4,
                    label=f"{v}: {b2[v]['a_ms']:.1f} + {b2[v]['b_ms']:.2f}·n".replace(".", ","))
        xx = np.array([0, max(ns)])
        ax.plot(xx, b2[v]["a_ms"] + b2[v]["b_ms"] * xx, color=cor, lw=0.8)
    ax.set_xlabel("veículos com busca de placa (n)")
    ax.set_ylabel("tempo de serviço (ms)")
    ax.set_title("(b) custo por quadro: a + b·n", loc="left", fontsize=7)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    _eixo(ax, "y")
    # (c) lei de Little: quadros no sistema (L) x utilização (ρ); acima da diagonal, espera
    ax = axs[2]
    for v, cor, marca in zip(VIDEOS, CORES, MARCAS, strict=True):
        pts = [p for p in b3.values() if p["video"] == v]
        ax.plot([p["rho_processamento"] for p in pts], [p["L_quadros"] for p in pts], marca, color=cor,
                ms=3.6, ls="none", mec="white", mew=0.4, label=v)
    pior = max(b3.values(), key=lambda p: p["prazo_perdido"])
    if pior["prazo_perdido"] > 0.02:
        nome = {"o1_ultralytics_rt": "sem O1"}.get(pior["condicao"], pior["condicao"])
        ax.annotate(f"{pior['video']} {nome}:\n{100 * pior['prazo_perdido']:.0f}% fora do prazo",
                    (pior["rho_processamento"], pior["L_quadros"]), textcoords="offset points",
                    xytext=(-4, 6), ha="right", fontsize=5.2, color=fig_base.TINTA_2)
    ax.plot([0, 1], [0, 1], color=fig_base.TINTA_2, ls=":", lw=0.7)
    ax.text(0.62, 0.5, "L = ρ\n(ninguém espera)", fontsize=5.3, color=fig_base.TINTA_2, rotation=0)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("utilização do processamento, ρ = λ·E[S]")
    ax.set_ylabel("quadros no sistema, L = λ·W")
    ax.set_title("(c) fila: L × ρ, 28 condições", loc="left", fontsize=7)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handletextpad=0.1)
    _eixo(ax, "y")
    _salvar(fig, plt, saida)


def figura_otimizacoes(c: dict, saida: Path) -> None:
    plt = _plt()
    linhas = c["linhas"][::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.35, 2.0), sharey=True,
                                   gridspec_kw={"wspace": 0.1, "width_ratios": [1, 1.1]})
    ys = np.arange(len(linhas))
    for y, lin in zip(ys, linhas, strict=True):
        lo, hi = lin["ic95"]
        ax1.plot([100 * lo, 100 * hi], [y, y], color=fig_base.AZUL, lw=1.3, solid_capstyle="butt")
        ax1.plot(100 * lin["delta_revocacao"], y, "o", color=fig_base.AZUL, ms=3.6, mec="white", mew=0.4)
        # cada vídeo numa altura própria, para as faixas entre repetições não se cobrirem
        for k, (v, cor, marca) in enumerate(zip(VIDEOS, CORES, MARCAS, strict=True)):
            pv = lin["por_video"][v]
            if "delta_p99_ms" in pv:
                yv = y + (k - 1.5) * 0.15
                lo_r, hi_r = pv["delta_p99_faixa_ms"]
                ax2.plot([lo_r, hi_r], [yv, yv], color=cor, lw=0.7, alpha=0.7, solid_capstyle="butt")
                ax2.plot(pv["delta_p99_ms"], yv, marca, color=cor, ms=3.0, mec="white", mew=0.3,
                         label=v if y == ys[0] else None)
    rotulos = [f"{lin['otimizacao']}, {'sem prazo' if lin['modo'] == 'off' else 'tempo real'}" for lin in linhas]
    ax1.set_ylim(-0.6, len(linhas) - 0.4)
    ax1.set_xlim(-3, 3.6)
    ax1.set_xticks([-2, 0, 2])
    for ax in (ax1, ax2):
        ax.axvline(0, color=fig_base.TINTA_2, lw=0.6)
    ax2.set_xscale("symlog", linthresh=10)
    extremos = [x for lin in linhas for v in VIDEOS if "delta_p99_faixa_ms" in lin["por_video"][v]
                for x in lin["por_video"][v]["delta_p99_faixa_ms"]]
    ax2.set_xlim(min(-3.0, 1.4 * min(extremos)), max(3.0, 1.4 * max(extremos)))
    marcas_x = [m for m in (-50, -10, -5, 0, 5, 10) if ax2.get_xlim()[0] <= m <= ax2.get_xlim()[1]]
    ax2.set_xticks(marcas_x, [_virgula(m) for m in marcas_x])
    ax1.set_xlabel("Δ revocação (p.p.)\ncom − sem, IC 95%")
    ax2.set_xlabel("Δ latência p99 (ms), com − sem;\ntraço: faixa entre repetições")
    ax2.legend(frameon=False, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4,
               handletextpad=0.05, columnspacing=0.5, borderaxespad=0.1)
    _eixo(ax1, formatar="x")
    _eixo(ax2, formatar="")
    ax1.set_yticks(ys, rotulos)
    _salvar(fig, plt, saida)


def figura_acuracia(d: dict, e: dict, saida: Path) -> None:
    """Uma linha, quatro painéis: tempo real × sem prazo, Kaplan–Meier, degrau e teto."""
    plt = _plt()
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.95), gridspec_kw={"wspace": 0.42})
    # (a) revocação em tempo real x sem prazo
    ax = axs[0]
    for v, cor, marca in zip(VIDEOS, CORES, MARCAS, strict=True):
        pts = [p for p in d["pontos"] if p["video"] == v]
        ax.plot([100 * p["revocacao_off"] for p in pts], [100 * np.mean(p["revocacao_rt"]) for p in pts],
                marca, color=cor, ms=3.3, ls="none", label=v, mec="white", mew=0.4)
    ax.plot([0, 100], [0, 100], color=fig_base.TINTA_2, lw=0.6, ls=":")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("revocação sem prazo (%)")
    ax.set_ylabel("revocação em tempo real (%)")
    ax.set_title("(a) 20 pares, 5 repetições", loc="left", fontsize=6.8)
    _eixo(ax, "y")
    # (b) Kaplan–Meier até τ
    ax = axs[1]
    tau = d["tau_s"]
    for modo, cor, rot, ls in (("off", fig_base.AZUL, "sem prazo", "-"), ("rt", fig_base.LARANJA, "tempo real", "--")):
        km = d["km"][modo]
        t = np.concatenate([[0.0], np.repeat(km["t"], 2), [tau]])
        s = np.concatenate([[1.0, 1.0], np.repeat(km["S"], 2)[:-1], [km["S"][-1], km["S"][-1]]])[: len(t)]
        ax.plot(t, s, color=cor, lw=1.0, ls=ls, label=f"{rot}: {km['rmst']:.2f} s".replace(".", ","))
        if modo == "off":
            ax.fill_between(t, s, color=cor, alpha=0.12, lw=0)
    ax.text(tau * 0.5, 0.16, r"$\mathrm{RMST}=\int_0^{\tau} S(t)\,dt$", fontsize=5.6, color=fig_base.TINTA_2,
            ha="center")
    ax.set_xlim(0, tau)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("tempo desde a 1ª aparição (s)")
    ax.set_ylabel("ainda sem placa correta, S(t)")
    ax.set_title("(b) Kaplan–Meier, 1280 px", loc="left", fontsize=6.8)
    ax.legend(frameon=False, fontsize=5.2, loc="upper right", title="RMST até 3 s", title_fontsize=5.2,
              handlelength=1.6, borderaxespad=0.1)
    _eixo(ax, "y", formatar="x")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,50", "0,75", "1"])
    # (c) o degrau no mínimo do OCR
    ax = axs[2]
    desloc = {"960": 0.9, "1280": 1.0, "1920": 1.1}
    for (res, cv), cor, marca, ls in zip(e["curvas"].items(), CORES[:3], MARCAS[:3], ("-", "--", "-."), strict=True):
        dg = cv["degrau"]
        xs = [f["centro_px"] * desloc[str(res)] for f in dg["faixas"]]
        ax.errorbar(xs, [f["p"] for f in dg["faixas"]],
                    yerr=[[max(0.0, f["p"] - f["ic"][0]) for f in dg["faixas"]],
                          [max(0.0, f["ic"][1] - f["p"]) for f in dg["faixas"]]],
                    fmt=marca, color=cor, ms=2.6, lw=0.45, alpha=0.75, capsize=0, mec="white", mew=0.3)
        ax.plot([40, dg["trava_px"], dg["trava_px"], 420], [0, 0, dg["patamar"], dg["patamar"]],
                color=cor, lw=0.9, ls=ls, label=f"{res}: {100 * dg['patamar']:.0f}%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([50, 100, 200, 400], ["50", "100", "200", "400"])
    ax.set_xlim(45, 420)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("maior largura da placa (px)")
    ax.set_ylabel("P(placa correta)")
    ax.set_title("(c) acerto × tamanho, V1–V3", loc="left", fontsize=6.8)
    _eixo(ax, "y")
    # (d) teto previsto x observado
    ax = axs[3]
    for v, cor, marca in zip(VIDEOS, CORES, MARCAS, strict=True):
        pts = [p for p in e["tetos"] if p["video"] == v]
        for p in pts:
            ax.plot([100 * p["teto_registrado"], 100 * p["teto_corrigido"]], [100 * p["revocacao"]] * 2,
                    color=cor, lw=0.6, alpha=0.8)
        ax.plot([100 * p["teto_registrado"] for p in pts], [100 * p["revocacao"] for p in pts], marca,
                mfc="white", mec=cor, ms=3.3, mew=0.8, ls="none")
        ax.plot([100 * p["teto_corrigido"] for p in pts], [100 * p["revocacao"] for p in pts], marca,
                color=cor, ms=3.3, ls="none", mec="white", mew=0.4)
    ax.plot([0, 100], [0, 100], color=fig_base.TINTA_2, lw=0.6, ls=":")
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 103)
    ax.set_xlabel("teto previsto (%)")
    ax.set_ylabel("revocação observada (%)")
    ax.set_title("(d) previsto × observado", loc="left", fontsize=6.8)
    _eixo(ax, "y")
    # legenda única dos vídeos (painéis a e d) e das marcas do teto
    from matplotlib.lines import Line2D
    alcas = [Line2D([], [], marker=m, color=c, ls="none", ms=3.3, mec="white", mew=0.4)
             for c, m in zip(CORES, MARCAS, strict=True)]
    alcas += [Line2D([], [], marker="o", mfc="white", mec=fig_base.TINTA_2, ls="none", ms=3.3, mew=0.8),
              Line2D([], [], marker="o", color=fig_base.TINTA_2, ls="none", ms=3.3)]
    nomes = [f"{v} ({NOMES[v].replace(', dia (vertical)', ', vertical')})" for v in VIDEOS]
    nomes += ["teto registrado", "teto corrigido"]
    for (res, cv), cor, ls in zip(e["curvas"].items(), CORES[:3], ("-", "--", "-."), strict=True):
        alcas.append(Line2D([], [], color=cor, lw=0.9, ls=ls))
        nomes.append(f"degrau a {res} px: patamar de {100 * cv['degrau']['patamar']:.0f}%")
    fig.legend(alcas, nomes, loc="upper center", ncol=5, frameon=False, fontsize=5.6, bbox_to_anchor=(0.5, -0.05),
               handletextpad=0.2, columnspacing=1.0)
    _salvar(fig, plt, saida)


# --------------------------------------------------------------------------- principal

def _limpo(o):
    """JSON sem NaN e com floats curtos."""
    if isinstance(o, dict):
        return {str(k): _limpo(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpo(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(argv: list[str] | None = None) -> int:
    res = {"suspensoes": suspensoes(), "b1_etapas": b1_etapas(), "b2_custo": b2_custo(), "b3_fila": b3_fila(),
           "c_otimizacoes": c_otimizacoes(), "d_temporeal": d_temporeal(), "e_largura": e_largura(),
           "f_exploratorias": f_exploratorias()}
    SAIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_JSON.write_text(json.dumps(_limpo(res), ensure_ascii=False, indent=1), encoding="utf-8")
    figura_tempo(res["b1_etapas"], res["b2_custo"], res["b3_fila"], FIGURAS / "tempo.pdf")
    figura_otimizacoes(res["c_otimizacoes"], FIGURAS / "otimizacoes.pdf")
    figura_acuracia(res["d_temporeal"], res["e_largura"], FIGURAS / "acuracia.pdf")
    print(f"gravado: {SAIDA_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
