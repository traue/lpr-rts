"""Avaliacao de uma sessao GRAVADA contra o ground truth.

Tudo e calculado a partir do banco (``detections``, ``track_points``, ``reads``,
``events``, ``frames``): qualquer sessao pode ser reavaliada depois - com outro
limiar, outro ground truth corrigido - sem reprocessar o video.

Pareamento por posicao, nunca por texto
---------------------------------------
Cada leitura, trilha e evento e associado a um veiculo anotado pela
**sobreposicao espacial no mesmo quadro** (IoU entre a caixa da placa do
sistema e a caixa anotada). Uma versao anterior pareava pelo texto mais
parecido: uma leitura errada era comparada com a placa de outro veiculo que
ela por acaso lembrasse, veiculos perdidos nao entravam na conta e duplicatas
preenchiam o lugar deles - a exatidao reportada era sistematicamente otimista.

O que e medido
--------------
* **Por veiculo** (metrica principal): revocacao = veiculos com a placa correta
  publicada / veiculos anotados; precisao = eventos corretos / eventos;
  duplicatas; fragmentacao; latencia de decisao.

Dois relogios, dois nomes
-------------------------
* **Latencia de decisao** (``decision_delay_*``, ``time_to_correct_s``) e tempo
  de **video**: do primeiro quadro anotado do veiculo a primeira decisao correta.
  Inclui o atraso de deteccao e nao depende da velocidade da maquina - e a
  medida comparavel entre o modo offline e o tempo real.
* **Tempo ate a decisao no relogio** (``decision_elapsed_*``,
  ``elapsed_to_correct_s``) e o relogio monotonico do host: da primeira imagem
  da trilha disponivel ate a decisao (``events.decision_elapsed_ms``, esquema 4).
  So tem leitura de latencia em tempo real; offline mede a velocidade do
  processamento. Nulo em bancos anteriores ao esquema 4.
* **Efeito da fusao**: na MESMA populacao de veiculos, leitura ao acaso x
  leitura mais confiante x fusao x decisao publicada, com teste pareado.
* **Por tentativa de OCR**: inclui as tentativas sem texto no denominador.
* **Deteccao** (placas e objetos) por quadro, com ``difficult`` ignorado.
* **Rastreamento**: MOTA, IDF1 e trocas de identidade sobre as caixas de placa.
* **Desempenho**: vazao, latencia ponta a ponta (p50/p95/p99), descartes, prazos.

Escopo por largura minima (``scope_min_width``)
-----------------------------------------------
Num video de rua ha placas longe demais para qualquer leitura (a pista oposta, os
carros estacionados ao fundo). Sem anotacao, o que o sistema fizer com elas conta
como erro; anota-las uma a uma custa horas. Com o escopo, a avaliacao vale para
placas com pelo menos N pixels de largura, como o tamanho minimo de objeto do
KITTI: a caixa anotada mais estreita conta como ``difficult``, e a saida do
sistema mais estreita **que nao cai sobre nenhuma placa anotada** e ignorada
(deteccao, leitura, caixa de trilha e o evento de uma trilha sem veiculo cuja
placa tem largura mediana abaixo de N). Placa grande sem anotacao continua erro:
o gabarito ainda precisa estar completo para o que da para ler.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import PlateLayout, PlateRead
from ..fusion.voting import PlateVoter
from ..track.assignment import gated_assignment, linear_sum_assignment
from ..utils.geometry import iou, iou_matrix
from .groundtruth import GroundTruth, VehicleGT
from .metrics import (
    character_error_rate,
    evaluate_detection,
    evaluate_ocr,
    mean_average_precision,
)
from .report import EvaluationReport
from .stats import bootstrap_ci, paired_comparison, ratio_bootstrap_ci

BBox = tuple[float, float, float, float]


def _connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def latest_session(db_path: str | Path) -> str:
    """Sessao mais recente do banco."""
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise ValueError(f"Nenhuma sessao em {db_path}")
    return str(row["session_id"])


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _ci(trio: tuple[float, float, float]) -> tuple[float, list[float]]:
    est, lo, hi = trio
    return est, [lo, hi]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _load_reads(con: sqlite3.Connection, sid: str) -> list[dict[str, Any]]:
    linhas = []
    for r in con.execute("SELECT * FROM reads WHERE session_id=? ORDER BY id", (sid,)):
        d = dict(r)
        try:
            d["char_confidences"] = json.loads(d.get("char_confidences") or "[]")
        except json.JSONDecodeError:
            d["char_confidences"] = []
        linhas.append(d)
    return linhas


def _as_plate_read(d: dict[str, Any]) -> PlateRead:
    return PlateRead(
        text=d.get("text") or "",
        confidence=float(d.get("confidence") or 0.0),
        char_confidences=[float(c) for c in d.get("char_confidences") or []],
        layout=PlateLayout(d.get("layout") or "unknown"),
        valid_syntax=bool(d.get("valid_syntax")),
        frame_index=int(d.get("frame_index") or 0),
        engine=d.get("engine") or "",
        sharpness=float(d.get("sharpness") or 0.0),
        raw_text=d.get("raw_text") or "",
    )


def _track_boxes(
    con: sqlite3.Connection, sid: str, reads: list[dict[str, Any]]
) -> tuple[dict[int, dict[int, BBox]], dict[int, dict[int, BBox]], bool]:
    """Caixas de placa e de objeto de cada trilha, por quadro.

    Placas vem de ``track_points`` (placa associada no quadro); na ausencia
    dessa tabela, das caixas das leituras.
    """
    placas: dict[int, dict[int, BBox]] = defaultdict(dict)
    objetos: dict[int, dict[int, BBox]] = defaultdict(dict)
    tem_pontos = False
    for r in con.execute(
        """SELECT track_id, frame_index, predicted, x1, y1, x2, y2,
                  plate_x1, plate_y1, plate_x2, plate_y2
           FROM track_points WHERE session_id=?""",
        (sid,),
    ):
        tem_pontos = True
        if r["predicted"]:
            continue            # caixa so prevista: nao e saida medida do sistema
        objetos[r["track_id"]][r["frame_index"]] = (r["x1"], r["y1"], r["x2"], r["y2"])
        if r["plate_x1"] is not None:
            placas[r["track_id"]][r["frame_index"]] = (
                r["plate_x1"], r["plate_y1"], r["plate_x2"], r["plate_y2"]
            )
    for d in reads:
        if (d.get("bbox_x2") or 0) > (d.get("bbox_x1") or 0):
            placas[d["track_id"]].setdefault(
                d["frame_index"], (d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"])
            )
    return placas, objetos, tem_pontos


# ---------------------------------------------------------------------------
# Associacao trilha -> veiculo anotado
# ---------------------------------------------------------------------------

def associate_tracks(
    track_boxes: dict[int, dict[int, BBox]],
    gt_boxes: dict[str, dict[int, BBox]],
    iou_threshold: float = 0.3,
) -> dict[int, dict[str, Any]]:
    """Associa cada trilha ao objeto anotado com que mais coincide no espaco e no tempo.

    Para cada quadro da trilha, procura a anotacao de maior IoU naquele quadro;
    a trilha fica com o objeto que venceu mais quadros (votacao). ``purity`` e a
    fracao dos quadros da trilha em que ela coincidiu com esse objeto.
    """
    por_quadro: dict[int, list[tuple[str, BBox]]] = defaultdict(list)
    for chave, caixas in gt_boxes.items():
        for quadro, caixa in caixas.items():
            por_quadro[quadro].append((chave, caixa))

    saida: dict[int, dict[str, Any]] = {}
    for tid, quadros in track_boxes.items():
        votos: Counter = Counter()
        for quadro, caixa in quadros.items():
            melhor, match_key = 0.0, None
            for k, g in por_quadro.get(quadro, []):
                v = iou(caixa, g)
                if v > melhor:
                    melhor, match_key = v, k
            if match_key is not None and melhor >= iou_threshold:
                votos[match_key] += 1
        if votos:
            chave, n = votos.most_common(1)[0]
            saida[tid] = {
                "vehicle": chave, "overlap_frames": n, "track_frames": len(quadros),
                "purity": n / max(1, len(quadros)),
            }
    return saida


def _final_reads(reads: list[PlateRead], max_reads: int) -> list[PlateRead]:
    """Reproduz o historico que a trilha tinha ao fim (corte por confianca online)."""
    if not max_reads:
        return list(reads)
    retidas: list[PlateRead] = []
    for r in reads:
        retidas.append(r)
        if len(retidas) > max_reads:
            retidas.sort(key=lambda x: x.confidence, reverse=True)
            del retidas[max_reads:]
    return retidas


# ---------------------------------------------------------------------------
# Rastreamento (CLEAR-MOT e IDF1 sobre caixas de placa)
# ---------------------------------------------------------------------------

def tracking_metrics(
    sys_boxes: dict[int, dict[int, BBox]],
    gt_boxes: dict[str, dict[int, BBox]],
    frames: set[int],
    iou_threshold: float = 0.5,
    ignore: dict[str, set[int]] | None = None,
) -> dict[str, Any]:
    """MOTA (CLEAR-MOT) e IDF1 sobre os quadros avaliados.

    Correspondencias do quadro anterior sao mantidas quando ainda validas (como
    em CLEAR-MOT); as demais saem de uma atribuicao hungara por IoU. Caixas
    anotadas como ``difficult`` nao contam como FN, e a trilha que as cobre nao
    conta como FP.
    """
    ignore = ignore or {}
    gt_q: dict[int, dict[str, BBox]] = defaultdict(dict)
    for k, caixas in gt_boxes.items():
        for f, b in caixas.items():
            gt_q[f][k] = b
    sys_q: dict[int, dict[int, BBox]] = defaultdict(dict)
    for t, caixas in sys_boxes.items():
        for f, b in caixas.items():
            sys_q[f][t] = b

    tp = fp = fn = idsw = 0
    n_gt = n_sys = 0
    ultimo: dict[str, int] = {}
    coocorrencia: Counter = Counter()
    for f in sorted(frames):
        g_all = gt_q.get(f, {})
        g = {k: b for k, b in g_all.items() if f not in ignore.get(k, set())}
        g_ign = {k: b for k, b in g_all.items() if f in ignore.get(k, set())}
        s = dict(sys_q.get(f, {}))
        # Trilhas sobre anotacoes dificeis saem da conta (nem TP nem FP).
        for t, sb in list(s.items()):
            if g_ign and max(iou(sb, gb) for gb in g_ign.values()) >= iou_threshold and not any(
                iou(sb, gb) >= iou_threshold for gb in g.values()
            ):
                del s[t]
        n_gt += len(g)
        n_sys += len(s)
        for k, gb in g.items():
            for t, sb in s.items():
                if iou(gb, sb) >= iou_threshold:
                    coocorrencia[(k, t)] += 1

        casados: dict[str, int] = {}
        livres_g = set(g)
        livres_s = set(s)
        for k, t in ultimo.items():
            if k in livres_g and t in livres_s and iou(g[k], s[t]) >= iou_threshold:
                casados[k] = t
                livres_g.discard(k)
                livres_s.discard(t)
        if livres_g and livres_s:
            lg, ls = sorted(livres_g), sorted(livres_s)
            m = iou_matrix(
                np.array([g[k] for k in lg], dtype=np.float32),
                np.array([s[t] for t in ls], dtype=np.float32),
            )
            rows, cols = gated_assignment(m, iou_threshold)
            for r, c in zip(rows, cols, strict=True):
                casados[lg[r]] = ls[c]
        for k, t in casados.items():
            if k in ultimo and ultimo[k] != t:
                idsw += 1
            ultimo[k] = t
        tp += len(casados)
        fp += len(s) - len(casados)
        fn += len(g) - len(casados)

    # IDF1: atribuicao global identidade-anotada <-> identidade-do-sistema.
    idtp = 0
    if coocorrencia:
        ks = sorted({k for k, _ in coocorrencia})
        ts = sorted({t for _, t in coocorrencia})
        matriz = np.zeros((len(ks), len(ts)), dtype=np.float64)
        for (k, t), n in coocorrencia.items():
            matriz[ks.index(k), ts.index(t)] = n
        rows, cols = linear_sum_assignment(-matriz)
        idtp = int(sum(matriz[r, c] for r, c in zip(rows, cols, strict=True)))
    return {
        "iou_threshold": iou_threshold,
        "frames": len(frames),
        "gt_boxes": n_gt,
        "sys_boxes": n_sys,
        "tp": tp, "fp": fp, "fn": fn, "id_switches": idsw,
        "mota": 1.0 - (fn + fp + idsw) / n_gt if n_gt else None,
        "idf1": 2.0 * idtp / (n_gt + n_sys) if (n_gt + n_sys) else None,
        "idtp": idtp,
    }


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------

def _objetos_nao_medidos(
    config: dict[str, Any], estatisticas: dict[str, Any], gt: GroundTruth,
) -> str | None:
    """Por que a deteccao de objetos da sessao nao pode ser avaliada (``None`` = pode).

    Sessoes sem a chave usam o padrao da configuracao (detector ligado).
    """
    if not config.get("detector", {}).get("detect_vehicles", True):
        return "o detector de objetos estava desligado na sessao"
    if estatisticas.get("fallback_active"):
        return ("o fallback automatico desligou o detector de objetos no meio da sessao "
                "(ver a nota da sessao)")
    if not gt.objects():
        return "o ground truth nao anota objetos"
    return None


def _no_escopo(gt: GroundTruth, largura: float) -> tuple[GroundTruth, int]:
    """O ground truth com as placas mais estreitas que ``largura`` marcadas ``difficult``.

    Devolve tambem quantas caixas mudaram. O original nao e alterado.
    """
    anotacoes = []
    mudadas = 0
    for a in gt.annotations:
        if a.kind == "plate" and not a.difficult and a.width < largura:
            a = replace(a, difficult=True)
            mudadas += 1
        anotacoes.append(a)
    return GroundTruth(video=gt.video, fps=gt.fps, annotations=anotacoes), mudadas


def evaluate_session(
    db_path: str | Path,
    session_id: str | None,
    gt: GroundTruth,
    iou_threshold: float = 0.5,
    assoc_iou: float = 0.3,
    readable_min_width: float | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    scope_min_width: float | None = None,
) -> EvaluationReport:
    """Avalia uma sessao do banco contra o ground truth (ver docstring do modulo).

    ``scope_min_width`` (pixels da imagem original) restringe a avaliacao as placas
    com pelo menos essa largura; ``None`` ou 0 avalia tudo, como sempre.
    """
    sid = session_id or latest_session(db_path)
    avisos: list[str] = []
    escopo = float(scope_min_width or 0.0)
    if escopo < 0:
        raise ValueError("scope_min_width deve ser >= 0")
    estreitas = 0
    if escopo > 0:
        gt, estreitas = _no_escopo(gt, escopo)
    with _connect(db_path) as con:
        sessao = con.execute("SELECT * FROM sessions WHERE session_id=?", (sid,)).fetchone()
        if sessao is None:
            raise ValueError(f"Sessao {sid} nao encontrada em {db_path}")
        sessao = dict(sessao)
        config = json.loads(sessao.get("config_json") or "{}")
        ambiente = json.loads(sessao.get("environment_json") or "{}")
        estatisticas = json.loads(sessao.get("stats_json") or "{}")
        frames = [dict(r) for r in con.execute(
            "SELECT * FROM frames WHERE session_id=? ORDER BY frame_index", (sid,)
        )]
        dets = [dict(r) for r in con.execute(
            "SELECT * FROM detections WHERE session_id=?", (sid,)
        )]
        reads = _load_reads(con, sid)
        events = [dict(r) for r in con.execute(
            "SELECT * FROM events WHERE session_id=? ORDER BY id", (sid,)
        )]
        placas_trilha, objetos_trilha, tem_pontos = _track_boxes(con, sid, reads)

    fps = float(sessao.get("source_fps") or gt.fps or 30.0)
    largura_src = float(sessao.get("source_width") or 0)
    largura_proc = float(sessao.get("processing_width") or largura_src or 0)
    escala_proc = (largura_proc / largura_src) if largura_src else 1.0
    cfg_ocr = config.get("ocr", {})
    cfg_fusao = config.get("fusion", {})

    quadros_proc = {f["frame_index"] for f in frames}
    quadros_det = {f["frame_index"] for f in frames if f.get("detection_ran", 1)}
    if not frames:
        avisos.append("Sessao sem telemetria de quadros: cobertura e desempenho indisponiveis.")
        quadros_det = {d["frame_index"] for d in dets} | {r["frame_index"] for r in reads}

    veiculos_all = gt.vehicles()
    excluidos = {k for k, v in veiculos_all.items() if v.all_difficult}
    if not gt.has_track_ids():
        avisos.append(
            "Ground truth sem track_id: cada veiculo e identificado pelo texto da placa "
            "(invalido se alguma placa aparece em mais de um veiculo)."
        )
    conflitos = gt.plate_text_conflicts()
    if conflitos:
        avisos.append(f"Identidades anotadas com mais de um texto de placa: {conflitos[:5]}")
    if cfg_ocr.get("max_reads_published", 0):
        avisos.append(
            "ocr.max_reads_published > 0: trilhas publicadas param de ser lidas, entao "
            "a exatidao por tentativa depende do acerto do proprio motor."
        )
    if cfg_fusao.get("max_reads_per_track", 0):
        avisos.append(
            "fusion.max_reads_per_track > 0: a fusao descartou leituras de menor "
            "confianca; a comparacao da fusao replica esse corte."
        )

    # ------------------------------------------------------------------
    # Deteccao por quadro (placas e objetos), so nos quadros detectados
    # ------------------------------------------------------------------
    gt_placas: dict[int, list[BBox]] = defaultdict(list)
    gt_placas_ign: dict[int, list[BBox]] = defaultdict(list)
    gt_obj: dict[int, list[BBox]] = defaultdict(list)
    gt_obj_ign: dict[int, list[BBox]] = defaultdict(list)
    for a in gt.annotations:
        alvo = (gt_placas_ign if a.difficult else gt_placas) if a.kind == "plate" else (
            gt_obj_ign if a.difficult else gt_obj
        )
        alvo[a.frame_index].append(a.bbox)

    fora = {"deteccoes": 0, "leituras": 0, "eventos": 0}

    def fora_do_escopo(caixa: BBox, quadro: int) -> bool:
        """Saida do sistema mais estreita que o escopo e sem placa anotada por baixo."""
        if escopo <= 0 or not 0 < caixa[2] - caixa[0] < escopo:
            return False
        anotadas = gt_placas.get(quadro, []) + gt_placas_ign.get(quadro, [])
        return all(iou(caixa, g) < iou_threshold for g in anotadas)

    pred_placas: dict[int, list] = defaultdict(list)
    pred_obj: dict[int, list] = defaultdict(list)
    for d in dets:
        caixa_det = (d["x1"], d["y1"], d["x2"], d["y2"])
        if d["kind"] == "plate" and fora_do_escopo(caixa_det, d["frame_index"]):
            fora["deteccoes"] += 1
            continue
        prediction = (caixa_det, float(d["score"]))
        (pred_placas if d["kind"] == "plate" else pred_obj)[d["frame_index"]].append(prediction)

    deteccao = objetos_det = None
    mapa: dict = {}
    recorded_detections = bool(dets) or (
        config.get("analytics", {}).get("store_detections") is True and bool(quadros_det)
    )
    if dets or gt_placas or gt_obj:
        if not recorded_detections:
            avisos.append("Sessao sem medicao de deteccoes (persistencia ou telemetria ausente).")
        else:
            deteccao = evaluate_detection(
                pred_placas, gt_placas, iou_threshold, gt_placas_ign, quadros_det
            )
            mapa = mean_average_precision(
                pred_placas, gt_placas, ignore_by_frame=gt_placas_ign, frames=quadros_det
            )
            # Zero deteccao com o detector LIGADO e erro medido; com ele desligado,
            # ou sem objetos no ground truth, nao ha o que medir. Contar como erro o
            # que nao foi anotado, ou o que nao foi procurado, inventa uma revocacao.
            sem_objetos = _objetos_nao_medidos(config, estatisticas, gt)
            if sem_objetos is None:
                objetos_det = evaluate_detection(
                    pred_obj, gt_obj, iou_threshold, gt_obj_ign, quadros_det
                )
            elif gt_obj or pred_obj:
                avisos.append(f"Deteccao de objetos nao avaliada: {sem_objetos}.")

    # ------------------------------------------------------------------
    # Associacao trilha -> veiculo anotado
    # ------------------------------------------------------------------
    gt_caixas_placa = {k: dict(v.boxes) for k, v in veiculos_all.items()}
    assoc = associate_tracks(placas_trilha, gt_caixas_placa, assoc_iou)
    gt_objetos_por_id: dict[str, dict[int, BBox]] = defaultdict(dict)
    for a in gt.objects():
        if a.track_id is not None:
            gt_objetos_por_id[f"id:{a.track_id}"][a.frame_index] = a.bbox
    if gt_objetos_por_id:
        # Trilhas sem caixa de placa (veiculo cuja placa nunca foi achada) ainda
        # podem ser associadas pela caixa do objeto.
        sem_placa = {t: q for t, q in objetos_trilha.items() if t not in assoc}
        for tid, info in associate_tracks(sem_placa, gt_objetos_por_id, assoc_iou).items():
            if info["vehicle"] in veiculos_all:
                assoc[tid] = info

    # ------------------------------------------------------------------
    # OCR por tentativa (pareado por posicao)
    # ------------------------------------------------------------------
    por_quadro_gt: dict[int, list[tuple[str, BBox, bool]]] = defaultdict(list)
    for a in gt.plates():
        por_quadro_gt[a.frame_index].append((a.vehicle_key, a.bbox, a.difficult))
    pares: list[tuple[str, str]] = []
    acertos_veic: dict[str, int] = defaultdict(int)
    tentativas_veic: dict[str, int] = defaultdict(int)
    sem_par = vazias = 0
    leituras_rot: list[dict[str, Any]] = []
    for d in reads:
        caixa = (d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"])
        melhor, chave, dificil = 0.0, None, False
        if caixa[2] > caixa[0]:
            for k, gb, dif in por_quadro_gt.get(d["frame_index"], []):
                v = iou(caixa, gb)
                if v > melhor:
                    melhor, chave, dificil = v, k, dif
        verdade = veiculos_all[chave].plate_text if chave in veiculos_all else ""
        casou = chave is not None and melhor >= iou_threshold and not dificil
        texto = d.get("text") or ""
        linha = {
            "read_id": d["id"], "track_id": d["track_id"], "frame_index": d["frame_index"],
            "video_time_s": d["video_time_s"], "gt_vehicle": chave if casou else "",
            "gt_plate": verdade if casou else "", "iou": round(melhor, 4),
            "text": texto, "raw_text": d.get("raw_text") or "",
            "correct": int(casou and texto == verdade),
            "cer": round(character_error_rate(verdade, texto), 4) if casou else None,
            "empty": int(not texto), "confidence": d.get("confidence"),
            "n_corrections": d.get("n_corrections"), "valid_syntax": d.get("valid_syntax"),
            "layout": d.get("layout"), "engine": d.get("engine"),
            "sharpness": d.get("sharpness"), "brightness": d.get("brightness"),
            "contrast": d.get("contrast"), "plate_w": d.get("plate_w"),
            "plate_h": d.get("plate_h"), "ocr_ms": d.get("ocr_ms"), "wait_ms": d.get("wait_ms"),
            "collected_frame": d.get("collected_frame"),
        }
        leituras_rot.append(linha)
        if not casou or chave is None:
            if dificil:
                pass
            elif fora_do_escopo(caixa, d["frame_index"]):
                fora["leituras"] += 1
            else:
                sem_par += 1
            continue
        pares.append((verdade, texto))
        tentativas_veic[chave] += 1
        acertos_veic[chave] += int(texto == verdade)
        vazias += int(not texto)

    ocr_quadro = evaluate_ocr(pares) if pares else None
    nivel_quadro: dict[str, Any] = {}
    if ocr_quadro is not None:
        chaves = sorted(tentativas_veic)
        est, ci = _ci(ratio_bootstrap_ci(
            [acertos_veic[k] for k in chaves], [tentativas_veic[k] for k in chaves],
            n_boot=n_boot, seed=seed,
        ))
        pares_com_texto = [p for p in pares if p[1]]
        nivel_quadro = {
            "n_attempts": len(pares),
            "accuracy": est, "accuracy_ci": ci,
            "accuracy_nonempty": (
                sum(1 for r, h in pares_com_texto if r == h) / len(pares_com_texto)
                if pares_com_texto else None
            ),
            "rejection_rate": vazias / len(pares),
            "n_unmatched_reads": sem_par,
            "n_vehicles_with_reads": len(chaves),
            "censored": bool(cfg_ocr.get("max_reads_published", 0)),
        }

    # ------------------------------------------------------------------
    # Por veiculo
    # ------------------------------------------------------------------
    leituras_por_trilha: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in reads:
        leituras_por_trilha[d["track_id"]].append(d)

    # Decisao vigente de cada trilha: a de maior revisao. Bancos da 0.2 nao tem
    # a coluna - la toda decisao era a primeira e unica (revisao 0).
    ultima_rev: dict[int, int] = {}
    for e in events:
        ultima_rev[e["track_id"]] = max(ultima_rev.get(e["track_id"], 0), int(e.get("revision") or 0))

    eventos_rot: list[dict[str, Any]] = []
    eventos_por_veic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        association = assoc.get(e["track_id"])
        chave = association["vehicle"] if association else None
        verdade = veiculos_all[chave].plate_text if chave in veiculos_all else ""
        correto = bool(verdade) and e["plate_text"] == verdade
        # Trilha sem veiculo anotado cuja placa e estreita demais na maior parte dos
        # quadros: fora do escopo, nao e acerto nem erro.
        evento_fora = False
        if escopo > 0 and chave is None:
            larguras = sorted(c[2] - c[0] for c in placas_trilha.get(e["track_id"], {}).values())
            evento_fora = bool(larguras) and larguras[len(larguras) // 2] < escopo
        linha = {
            "event_id": e["id"], "track_id": e["track_id"], "frame_index": e["frame_index"],
            "video_time_s": e["video_time_s"], "plate_text": e["plate_text"],
            "gt_vehicle": chave or "", "gt_plate": verdade or "", "correct": int(correto),
            "excluded_difficult": int(chave in excluidos) if chave else 0,
            "confidence": e.get("confidence"), "agreement": e.get("agreement"),
            "n_reads": e.get("n_reads"), "valid_syntax": e.get("valid_syntax"),
            # Tempo de video desde o primeiro quadro da trilha (a medida da 0.4).
            "decision_delay_s": (
                e["video_time_s"] - e["track_first_seen_s"]
                if e.get("track_first_seen_s") is not None else None
            ),
            # Relogio do host desde a primeira imagem da trilha (esquema 4).
            "decision_elapsed_s": (
                e["decision_elapsed_ms"] / 1000.0
                if e.get("decision_elapsed_ms") is not None else None
            ),
            "decision_ts": e.get("ts"),
            "decision": e.get("decision") or "stable",
            "revision": int(e.get("revision") or 0),
            "in_force": int(int(e.get("revision") or 0) == ultima_rev[e["track_id"]]),
        }
        if escopo > 0:
            linha["out_of_scope"] = int(evento_fora)
            fora["eventos"] += int(evento_fora and linha["in_force"])
        eventos_rot.append(linha)
        if chave:
            eventos_por_veic[chave].append(linha)

    trilhas_por_veic: dict[str, list[int]] = defaultdict(list)
    for tid, info in assoc.items():
        trilhas_por_veic[info["vehicle"]].append(tid)

    voter = PlateVoter(
        min_reads=int(cfg_fusao.get("min_reads", 3)),
        min_agreement=float(cfg_fusao.get("min_agreement", 0.6)),
        min_confidence=float(cfg_fusao.get("min_confidence", 0.45)),
        require_valid_syntax=bool(cfg_fusao.get("require_valid_syntax", True)),
        mode=str(cfg_fusao.get("mode", "char")),
        max_corrections=int(cfg_ocr.get("max_grammar_corrections", 2)),
    )
    max_retidas = int(cfg_fusao.get("max_reads_per_track", 0) or 0)
    if readable_min_width is None:
        readable_min_width = float(cfg_ocr.get("min_plate_width", 60)) / max(escala_proc, 1e-9)

    linhas_veic: list[dict[str, Any]] = []
    atrasos: list[float] = []
    decorridos: list[float] = []
    sem_relogio = 0
    for chave, vehicle in sorted(veiculos_all.items()):
        emitidos = eventos_por_veic.get(chave, [])
        # Metricas de decisao sobre as VIGENTES; o que foi emitido e depois
        # revisado entra em first_published_correct e n_wrong_emitted.
        evs = [e for e in emitidos if e["in_force"]]
        primeiro = min(emitidos, key=lambda e: (e["video_time_s"], e["event_id"]), default=None)
        tids = trilhas_por_veic.get(chave, [])
        # Trilha dominante: a que mais leu este veiculo - base da comparacao da fusao.
        dominante = max(tids, key=lambda t: len(leituras_por_trilha.get(t, [])), default=None)
        tentativas = [_as_plate_read(d) for d in leituras_por_trilha.get(dominante, [])] \
            if dominante is not None else []
        com_texto = [r for r in tentativas if r.text]
        single = (
            sum(1 for r in tentativas if r.text == vehicle.plate_text) / len(tentativas)
            if tentativas else None
        )
        best_read = max(com_texto, key=lambda r: r.confidence) if com_texto else None
        fusao = voter.vote(_final_reads(com_texto, max_retidas)) if com_texto else None
        ev_dom = [e for e in evs if e["track_id"] == dominante]
        # A primeira decisao correta emitida (inclui a revisao que corrigiu): os
        # dois relogios saem do MESMO evento.
        primeira_ok = min((e for e in emitidos if e["correct"]),
                          key=lambda e: (e["video_time_s"], e["event_id"]), default=None)
        primeiro_ok = primeira_ok["video_time_s"] if primeira_ok else None
        t_inicio = vehicle.first_frame / fps if fps else None
        atraso = ((primeiro_ok - t_inicio)
                  if primeiro_ok is not None and t_inicio is not None else None)
        decorrido = primeira_ok["decision_elapsed_s"] if primeira_ok else None
        if chave not in excluidos:
            if atraso is not None:
                atrasos.append(atraso)
            if decorrido is not None:
                decorridos.append(decorrido)
            elif primeira_ok is not None:
                sem_relogio += 1
        linhas_veic.append({
            "vehicle": chave, "plate_text": vehicle.plate_text, "object_class": vehicle.object_class,
            "first_frame": vehicle.first_frame, "last_frame": vehicle.last_frame,
            "n_frames_annotated": vehicle.n_frames,
            "max_plate_width_px": round(vehicle.max_width, 1),
            "readable": int(vehicle.max_width >= readable_min_width),
            "excluded_difficult": int(chave in excluidos),
            "detected": int(bool(tids)), "n_tracks": len(tids),
            "n_attempts": len(tentativas), "n_reads_text": len(com_texto),
            "single_read_accuracy": None if single is None else round(single, 4),
            "best_read_correct": int(best_read.text == vehicle.plate_text) if best_read else 0,
            "fused_text": fusao.text if fusao else "",
            "fused_correct": int(fusao.text == vehicle.plate_text) if fusao else 0,
            "published": int(bool(evs)),
            "published_correct": int(any(e["correct"] for e in evs)),
            "dominant_published_correct": int(any(e["correct"] for e in ev_dom)),
            "first_published_correct": int(bool(primeiro and primeiro["correct"])),
            "published_exit_relaxed": int(any(e["decision"] == "exit_relaxed" for e in evs)),
            "n_events": len(evs), "n_wrong_events": sum(1 for e in evs if not e["correct"]),
            "n_revisions": sum(1 for e in emitidos if e["revision"] > 0),
            "n_wrong_emitted": sum(1 for e in emitidos if not e["correct"]),
            "first_correct_time_s": primeiro_ok,
            "time_to_correct_s": None if atraso is None else round(atraso, 3),
            "elapsed_to_correct_s": None if decorrido is None else round(decorrido, 3),
            "mean_sharpness": round(float(np.mean([r.sharpness for r in tentativas])), 2)
            if tentativas else None,
        })

    elegiveis = [r for r in linhas_veic if not r["excluded_difficult"]]
    legiveis = [r for r in elegiveis if r["readable"]]
    emitidos_validos = [e for e in eventos_rot
                        if not e["excluded_difficult"] and not e.get("out_of_scope")]
    ev_validos = [e for e in emitidos_validos if e["in_force"]]
    nivel_veic: dict[str, Any] = {"n_vehicles": len(elegiveis), "n_excluded_difficult": len(excluidos)}
    if elegiveis:
        rec, rec_ci = _ci(bootstrap_ci([r["published_correct"] for r in elegiveis], n_boot, seed=seed))
        nivel_veic.update({"recall": rec, "recall_ci": rec_ci})
    if legiveis:
        rl, rl_ci = _ci(bootstrap_ci([r["published_correct"] for r in legiveis], n_boot, seed=seed))
        nivel_veic.update({
            "recall_readable": rl, "recall_readable_ci": rl_ci, "n_readable": len(legiveis),
            "readable_min_width_px": readable_min_width,
        })
    if ev_validos:
        pr, pr_ci = _ci(bootstrap_ci([e["correct"] for e in ev_validos], n_boot, seed=seed))
        nivel_veic.update({"precision": pr, "precision_ci": pr_ci})
    n_rev = sum(1 for e in emitidos_validos if e["revision"] > 0)
    if n_rev or any(e["decision"] == "exit_relaxed" for e in ev_validos):
        # So com revisoes/decisoes relaxadas: sem elas, coincidem com recall/precision.
        pe, pe_ci = _ci(bootstrap_ci([e["correct"] for e in emitidos_validos], n_boot, seed=seed))
        fr, fr_ci = _ci(bootstrap_ci([r["first_published_correct"] for r in elegiveis],
                                     n_boot, seed=seed))
        nivel_veic.update({
            "precision_emitted": pe, "precision_emitted_ci": pe_ci,
            "first_decision_recall": fr, "first_decision_recall_ci": fr_ci,
            "n_revisions": n_rev,
            "n_exit_relaxed": sum(1 for e in ev_validos if e["decision"] == "exit_relaxed"),
        })
    p, r_ = nivel_veic.get("precision"), nivel_veic.get("recall")
    detectados = [r for r in elegiveis if r["detected"]]
    nivel_veic.update({
        "f1": (2 * p * r_ / (p + r_)) if p and r_ else 0.0,
        "n_events": len(ev_validos),
        "n_events_unmatched": sum(1 for e in ev_validos if not e["gt_vehicle"]),
        "n_duplicate_events": sum(max(0, r["n_events"] - 1) for r in elegiveis),
        "n_detected": len(detectados),
        "tracks_per_detected_vehicle": (
            float(np.mean([r["n_tracks"] for r in detectados])) if detectados else None
        ),
        "decision_delay_median_s": _pct(atrasos, 50),
        "decision_delay_p95_s": _pct(atrasos, 95),
        "decision_delay_n": len(atrasos),
        "decision_elapsed_median_s": _pct(decorridos, 50),
        "decision_elapsed_p95_s": _pct(decorridos, 95),
        "decision_elapsed_n": len(decorridos),
    })
    if sem_relogio:
        # Tempo desconhecido nao e tempo zero: esses veiculos ficam fora do relogio.
        avisos.append(
            f"{sem_relogio} decisao(oes) correta(s) sem decision_elapsed_ms (gravadas antes "
            "do esquema 4): o tempo ate a decisao no relogio nao inclui esses veiculos."
        )

    # Efeito da fusao: mesmos veiculos (os que receberam leitura com texto).
    com_leitura = [r for r in elegiveis if r["n_reads_text"] > 0]
    fusao_cmp: dict[str, Any] = {}
    if com_leitura:
        fusao_cmp = {
            "n": len(com_leitura),
            "single_read": float(np.mean([r["single_read_accuracy"] or 0.0 for r in com_leitura])),
            "best_read": float(np.mean([r["best_read_correct"] for r in com_leitura])),
            "fused": float(np.mean([r["fused_correct"] for r in com_leitura])),
            "published": float(np.mean([r["dominant_published_correct"] for r in com_leitura])),
            "fused_vs_best": paired_comparison(
                [bool(r["best_read_correct"]) for r in com_leitura],
                [bool(r["fused_correct"]) for r in com_leitura],
            ),
        }
        fusion_diff, lo, hi = bootstrap_ci(
            [r["fused_correct"] - (r["single_read_accuracy"] or 0.0) for r in com_leitura],
            n_boot, seed=seed,
        )
        fusao_cmp["fused_minus_single"] = {"diff": fusion_diff, "ci": [lo, hi]}

    # ------------------------------------------------------------------
    # Rastreamento
    # ------------------------------------------------------------------
    rastreio: dict[str, Any] = {}
    if tem_pontos and veiculos_all:
        ignorar = {k: set(v.difficult_frames) for k, v in veiculos_all.items()}
        placas_medidas = placas_trilha
        if escopo > 0:
            placas_medidas = {
                t: {q: c for q, c in quadros.items() if not fora_do_escopo(c, q)}
                for t, quadros in placas_trilha.items()
            }
        rastreio = tracking_metrics(
            placas_medidas, gt_caixas_placa, quadros_det, iou_threshold, ignorar
        )
        rastreio["tracks_per_vehicle"] = nivel_veic.get("tracks_per_detected_vehicle")
    elif veiculos_all:
        avisos.append("Sessao sem track_points: metricas de rastreamento indisponiveis.")

    # ------------------------------------------------------------------
    # Desempenho e cobertura
    # ------------------------------------------------------------------
    desempenho: dict[str, Any] = {}
    workers = int(estatisticas.get("pipeline_workers") or 0)
    if workers > 0 and not sessao.get("realtime"):
        # A acuracia e a do serial (bit a bit); o tempo por quadro nao.
        avisos.append(
            f"Sessao offline com {workers} workers em paralelo: a latencia por quadro inclui "
            "a fila da reordenacao e a disputa pelos nucleos - nao e o custo de processar um "
            "quadro. Para medi-lo, rode com --parallel 0. A acuracia e identica a do serial."
        )
    if frames:
        lat = [f["latency_ms"] for f in frames if f.get("latency_ms") is not None]
        svc = [f["total_ms"] for f in frames if f.get("total_ms") is not None]
        fila = [f["queue_ms"] for f in frames if f.get("queue_ms") is not None]
        feitos = [f["done_t_s"] for f in frames if f.get("done_t_s") is not None]
        descartes = max((f.get("dropped_frames") or 0) for f in frames)
        atrasados = [f["deadline_miss"] for f in frames if f.get("deadline_miss") is not None]
        esperas = [d["wait_ms"] for d in reads if d.get("wait_ms") is not None]
        desempenho = {
            "n_frames": len(frames),
            "throughput_fps": (
                (len(feitos) - 1) / (max(feitos) - min(feitos))
                if len(feitos) > 1 and max(feitos) > min(feitos) else None
            ),
            "latency_p50_ms": _pct(lat, 50), "latency_p95_ms": _pct(lat, 95),
            "latency_p99_ms": _pct(lat, 99), "latency_max_ms": max(lat) if lat else None,
            "service_p50_ms": _pct(svc, 50), "service_p95_ms": _pct(svc, 95),
            "service_p99_ms": _pct(svc, 99), "queue_p95_ms": _pct(fila, 95),
            "dropped_frames": descartes,
            "drop_rate": descartes / (len(frames) + descartes) if (len(frames) + descartes) else 0.0,
            "deadline_ms": sessao.get("deadline_ms"),
            "deadline_miss_rate": float(np.mean(atrasados)) if atrasados else None,
            "ocr_wait_p95_ms": _pct(esperas, 95),
            "realtime": bool(sessao.get("realtime")),
        }

    quadros_gt = {a.frame_index for a in gt.plates()}
    cobertura: dict[str, Any] = {
        "gt_frames": len(quadros_gt),
        "gt_frames_processed": len(quadros_gt & quadros_proc) if frames else None,
        "gt_frames_detected": len(quadros_gt & quadros_det),
    }
    if escopo > 0:
        nivel_veic["scope_min_width_px"] = escopo
        nivel_veic["n_events_out_of_scope"] = fora["eventos"]
        if nivel_quadro:
            nivel_quadro["n_reads_out_of_scope"] = fora["leituras"]
        cobertura["scope_min_width_px"] = escopo
        cobertura["gt_boxes_below_scope"] = estreitas
        cobertura["detections_out_of_scope"] = fora["deteccoes"]
        avisos.append(
            f"Escopo: so placas com pelo menos {escopo:g} px de largura. {estreitas} caixa(s) "
            f"anotada(s) mais estreita(s) contam como difficult; fora do escopo e sem placa "
            f"anotada, ignorados: {fora['deteccoes']} deteccao(oes), {fora['leituras']} "
            f"leitura(s) e {fora['eventos']} evento(s). Declare a largura no artigo."
        )

    return EvaluationReport(
        video=str(sessao.get("source_uri") or gt.video),
        session_id=sid,
        n_frames=len(frames),
        detection=deteccao,
        object_detection=objetos_det,
        map_scores=mapa,
        ocr_frame_level=ocr_quadro,
        frame_level=nivel_quadro,
        vehicle_level=nivel_veic,
        fusion_comparison=fusao_cmp,
        tracking=rastreio,
        performance=desempenho,
        coverage=cobertura,
        warnings=avisos,
        components=ambiente.get("components", {}),
        config=config,
        environment=ambiente,
        vehicles=linhas_veic,
        reads=leituras_rot,
        events=eventos_rot,
    )


def gt_vehicle_table(gt: GroundTruth) -> list[VehicleGT]:
    """Lista os veiculos anotados (util em notebooks)."""
    return list(gt.vehicles().values())
