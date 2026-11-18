"""Metricas de avaliacao: deteccao (P/R/F1, AP, mAP) e OCR (CER/WER, exatidao).

Convencoes seguidas (para que os numeros sejam comparaveis com a literatura):

* **Deteccao** - um par predicao/referencia conta como verdadeiro positivo se
  IoU >= limiar; cada referencia so pode ser casada uma vez, e as predicoes sao
  processadas em ordem decrescente de confianca (protocolo PASCAL VOC/COCO).
* **AP** - area sob a curva precisao-revocacao com interpolacao em 101 pontos
  (protocolo COCO). ``mAP@[.5:.95]`` e a media de AP nos limiares 0,50 a 0,95.
* **CER/WER** - distancia de Levenshtein normalizada pelo comprimento da
  referencia. Para placas, WER equivale a "taxa de placa inteira errada", ou
  seja, 1 - exatidao exata.

Um ponto metodologico importante: em LPR, a metrica que interessa ao usuario
final e a **exatidao por placa** (a placa inteira certa), nao o CER. Uma placa
com um caractere errado e inutil na pratica. Reportamos ambos.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..utils.geometry import iou_matrix

# ---------------------------------------------------------------------------
# Distancia de edicao
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Distancia de edicao com programacao dinamica em O(len(a) x len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Apenas duas linhas da matriz sao necessarias.
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = distancia de edicao / comprimento da referencia."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


# ---------------------------------------------------------------------------
# Deteccao
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DetectionMetrics:
    """Precisao/revocacao/F1 num limiar de IoU, mais AP."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_threshold: float = 0.5
    ap: float = 0.0
    mean_iou: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "ap": self.ap, "mean_iou": self.mean_iou, "iou_threshold": self.iou_threshold,
        }


def match_detections(
    preds: list[tuple[tuple[float, float, float, float], float]],
    gts: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
    ignore: list[tuple[float, float, float, float]] | None = None,
) -> tuple[list[bool], list[int], list[float]]:
    """Casa predicoes com referencias por IoU guloso, em ordem de confianca.

    Retorna ``(eh_tp, indice_gt_casada, iou_do_par)``, alinhados as predicoes
    ordenadas por confianca decrescente. Predicoes que nao casam com nenhuma
    referencia mas casam com uma caixa de ``ignore`` (anotacao ``difficult``)
    recebem ``indice_gt_casada = -2``: nao sao acerto nem falso positivo.
    """
    if not preds:
        return [], [], []
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    pred_boxes = np.array([preds[i][0] for i in order], dtype=np.float32)
    ignore = ignore or []
    ious_ign = (
        iou_matrix(pred_boxes, np.array(ignore, dtype=np.float32)) if ignore else None
    )

    def ignorada(row: int) -> bool:
        return ious_ign is not None and float(ious_ign[row].max()) >= iou_threshold

    if not gts:
        unmatched = [-2 if ignorada(r) else -1 for r in range(len(order))]
        return [False] * len(order), unmatched, [0.0] * len(order)

    gt_boxes = np.array(gts, dtype=np.float32)
    ious = iou_matrix(pred_boxes, gt_boxes)

    used: set[int] = set()
    is_tp: list[bool] = []
    matched: list[int] = []
    match_iou: list[float] = []
    for row in range(len(order)):
        best_iou, best_gt = 0.0, -1
        for col in range(len(gts)):
            if col in used:
                continue
            if ious[row, col] > best_iou:
                best_iou, best_gt = float(ious[row, col]), col
        if best_gt >= 0 and best_iou >= iou_threshold:
            used.add(best_gt)
            is_tp.append(True)
            matched.append(best_gt)
            match_iou.append(best_iou)
        else:
            is_tp.append(False)
            matched.append(-2 if ignorada(row) else -1)
            match_iou.append(best_iou)
    return is_tp, matched, match_iou


def average_precision(
    preds: list[tuple[tuple[float, float, float, float], float, int]],
    gts_by_frame: dict[int, list[tuple[float, float, float, float]]],
    iou_threshold: float = 0.5,
    ignore_by_frame: dict[int, list[tuple[float, float, float, float]]] | None = None,
) -> float:
    """AP no protocolo COCO (interpolacao em 101 pontos de revocacao).

    ``preds`` sao ``(bbox, score, frame_index)`` de todo o video; a ordenacao
    global por confianca e o que torna a curva P-R correta. Predicoes sobre
    anotacoes ``difficult`` (``ignore_by_frame``) sao descartadas da curva.
    """
    from ..utils.geometry import iou as iou_fn

    n_gt = sum(len(v) for v in gts_by_frame.values())
    if n_gt == 0:
        return 0.0
    if not preds:
        return 0.0
    ignore_by_frame = ignore_by_frame or {}

    ordered = sorted(preds, key=lambda p: p[1], reverse=True)
    used: dict[int, set[int]] = defaultdict(set)
    tps: list[float] = []
    fps: list[float] = []

    for box, _score, frame_index in ordered:
        gts = gts_by_frame.get(frame_index, [])
        best_iou, best_gt = 0.0, -1
        for j, gt_box in enumerate(gts):
            if j in used[frame_index]:
                continue
            value = iou_fn(box, gt_box)
            if value > best_iou:
                best_iou, best_gt = value, j
        if best_gt >= 0 and best_iou >= iou_threshold:
            used[frame_index].add(best_gt)
            tps.append(1.0)
            fps.append(0.0)
        elif any(iou_fn(box, g) >= iou_threshold for g in ignore_by_frame.get(frame_index, [])):
            continue            # sobre anotacao dificil: fora da curva
        else:
            tps.append(0.0)
            fps.append(1.0)
    if not tps:
        return 0.0

    cum_tp = np.cumsum(np.asarray(tps, dtype=np.float64))
    cum_fp = np.cumsum(np.asarray(fps, dtype=np.float64))
    recalls = cum_tp / n_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # Envoltoria monotona decrescente da precisao (padrao VOC/COCO).
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    levels = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(levels)
    for k, level in enumerate(levels):
        mask = recalls >= level
        if mask.any():
            interpolated[k] = np.max(precisions[mask])
    return float(interpolated.mean())


def evaluate_detection(
    preds_by_frame: dict[int, list[tuple[tuple[float, float, float, float], float]]],
    gts_by_frame: dict[int, list[tuple[float, float, float, float]]],
    iou_threshold: float = 0.5,
    ignore_by_frame: dict[int, list[tuple[float, float, float, float]]] | None = None,
    frames: set[int] | None = None,
) -> DetectionMetrics:
    """P/R/F1 e AP agregados sobre os quadros avaliados.

    ``frames`` restringe a avaliacao a um conjunto de quadros (por exemplo, os
    que de fato passaram pelo detector num modo com descarte). ``ignore_by_frame``
    traz as anotacoes ``difficult``: nao contam como falso negativo, e predicoes
    sobre elas nao contam como falso positivo.
    """
    ignore_by_frame = ignore_by_frame or {}
    tp = fp = fn = 0
    ious: list[float] = []
    all_frames = set(preds_by_frame) | set(gts_by_frame)
    if frames is not None:
        all_frames &= frames
    for frame in all_frames:
        preds = preds_by_frame.get(frame, [])
        gts = gts_by_frame.get(frame, [])
        is_tp, matched, match_iou = match_detections(
            preds, gts, iou_threshold, ignore_by_frame.get(frame)
        )
        frame_tp = sum(is_tp)
        ignoradas = sum(1 for m in matched if m == -2)
        tp += frame_tp
        fp += len(preds) - frame_tp - ignoradas
        fn += len(gts) - frame_tp
        ious.extend(v for v, hit in zip(match_iou, is_tp, strict=True) if hit)

    sel_preds = {f: v for f, v in preds_by_frame.items() if frames is None or f in frames}
    sel_gts = {f: v for f, v in gts_by_frame.items() if frames is None or f in frames}
    flat = [(b, s, f) for f, items in sel_preds.items() for (b, s) in items]
    return DetectionMetrics(
        tp=tp, fp=fp, fn=fn, iou_threshold=iou_threshold,
        ap=average_precision(flat, sel_gts, iou_threshold, ignore_by_frame),
        mean_iou=float(np.mean(ious)) if ious else 0.0,
    )


def mean_average_precision(
    preds_by_frame: dict[int, list[tuple[tuple[float, float, float, float], float]]],
    gts_by_frame: dict[int, list[tuple[float, float, float, float]]],
    thresholds: tuple[float, ...] = tuple(np.arange(0.5, 1.0, 0.05)),
    ignore_by_frame: dict[int, list[tuple[float, float, float, float]]] | None = None,
    frames: set[int] | None = None,
) -> dict[str, float | dict[str, float]]:
    """mAP@[.5:.95] junto com AP@0.5 e AP@0.75, como se reporta em COCO."""
    sel_preds = {f: v for f, v in preds_by_frame.items() if frames is None or f in frames}
    sel_gts = {f: v for f, v in gts_by_frame.items() if frames is None or f in frames}
    flat = [(b, s, f) for f, items in sel_preds.items() for (b, s) in items]
    aps = {
        round(t, 2): average_precision(flat, sel_gts, t, ignore_by_frame) for t in thresholds
    }
    return {
        "mAP@[.5:.95]": float(np.mean(list(aps.values()))) if aps else 0.0,
        "AP@0.50": aps.get(0.5, 0.0),
        "AP@0.75": aps.get(0.75, 0.0),
        "per_threshold": {str(k): v for k, v in aps.items()},
    }


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OCRMetrics:
    """Metricas de reconhecimento de texto de placas."""

    n: int = 0
    exact_matches: int = 0
    cer_sum: float = 0.0
    total_chars: int = 0
    total_edits: int = 0
    confusions: Counter = field(default_factory=Counter)   # (referencia, predito) -> n
    per_position_errors: Counter = field(default_factory=Counter)

    @property
    def exact_accuracy(self) -> float:
        """Fracao de placas 100% corretas - a metrica que importa na pratica."""
        return self.exact_matches / self.n if self.n > 0 else 0.0

    @property
    def cer(self) -> float:
        """CER agregado (soma de edicoes / soma de caracteres de referencia)."""
        return self.total_edits / self.total_chars if self.total_chars > 0 else 0.0

    @property
    def mean_cer(self) -> float:
        """Media dos CER por amostra (pondera cada placa igualmente)."""
        return self.cer_sum / self.n if self.n > 0 else 0.0

    @property
    def wer(self) -> float:
        """Taxa de placa inteira errada."""
        return 1.0 - self.exact_accuracy

    def top_confusions(self, k: int = 15) -> list[tuple[str, str, int]]:
        return [(ref, hyp, n) for (ref, hyp), n in self.confusions.most_common(k)]

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "exact_accuracy": self.exact_accuracy,
            "wer": self.wer,
            "cer": self.cer,
            "mean_cer": self.mean_cer,
            "total_chars": self.total_chars,
            "total_edits": self.total_edits,
            "top_confusions": [
                {"ref": r, "hyp": h, "n": n} for r, h, n in self.top_confusions()
            ],
            "per_position_errors": dict(self.per_position_errors),
        }


def evaluate_ocr(pairs: list[tuple[str, str]]) -> OCRMetrics:
    """Avalia pares ``(referencia, predicao)`` ja normalizados.

    Alem de CER/exatidao, acumula a matriz de confusao de caracteres e os erros
    por posicao - as duas analises que revelam *onde* o modelo falha (por
    exemplo, confusao sistematica 8<->B, ou erro concentrado no ultimo digito,
    tipicamente o mais distorcido pela perspectiva).
    """
    metrics = OCRMetrics()
    for reference, hypothesis in pairs:
        metrics.n += 1
        if reference == hypothesis:
            metrics.exact_matches += 1
        metrics.cer_sum += character_error_rate(reference, hypothesis)
        metrics.total_chars += len(reference)
        metrics.total_edits += levenshtein(reference, hypothesis)
        # Confusao posicional so faz sentido com o mesmo comprimento.
        if len(reference) == len(hypothesis):
            for pos, (rc, hc) in enumerate(zip(reference, hypothesis, strict=True)):
                if rc != hc:
                    metrics.confusions[(rc, hc)] += 1
                    metrics.per_position_errors[pos] += 1
    return metrics


def confusion_matrix(
    pairs: list[tuple[str, str]], alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
) -> tuple[np.ndarray, list[str]]:
    """Matriz de confusao de caracteres (linhas = referencia, colunas = predito)."""
    index = {ch: i for i, ch in enumerate(alphabet)}
    size = len(alphabet)
    matrix = np.zeros((size, size), dtype=np.int64)
    for reference, hypothesis in pairs:
        if len(reference) != len(hypothesis):
            continue
        for rc, hc in zip(reference, hypothesis, strict=True):
            if rc in index and hc in index:
                matrix[index[rc], index[hc]] += 1
    return matrix, list(alphabet)
