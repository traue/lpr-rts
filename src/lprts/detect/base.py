"""Interface comum dos detectores e NMS independente de framework."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import Detection


def file_sha256(path: str) -> str:
    """SHA-256 atual do arquivo/pacote; o mesmo caminho pode receber novos pesos."""
    p = Path(path)
    h = hashlib.sha256()
    arquivos = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    for arq in arquivos:
        if p.is_dir():
            h.update(str(arq.relative_to(p)).encode())
        with arq.open("rb") as fh:
            for bloco in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloco)
    return h.hexdigest()


class Detector(ABC):
    """Contrato minimo: recebe uma imagem BGR, devolve deteccoes na escala da imagem."""

    name: str = "detector"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Executa a inferencia num quadro."""

    def describe(self) -> dict[str, Any]:
        """Onde e como o detector roda de fato - vai para a proveniencia da sessao."""
        return {"name": self.name, "runtime": "?", "device": "?"}

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Inferencia em lote. A implementacao base itera; subclasses podem otimizar."""
        return [self.detect(img) for img in images]

    def warmup(self, width: int = 640, height: int = 640, runs: int = 2) -> None:
        """Aquece o grafo/alocadores.

        Sem isso, o primeiro quadro real leva de 10x a 100x mais tempo (compilacao
        do kernel Metal/CoreML) e polui a medicao de latencia.
        """
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(runs):
            try:
                self.detect(dummy)
            except Exception:  # noqa: BLE001 - aquecimento nunca deve derrubar o app
                return

    def close(self) -> None:
        return None


def nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    """Non-Maximum Suppression classico, por classe, com ordenacao por score."""
    if len(detections) <= 1:
        return list(detections)

    from ..utils.geometry import iou_matrix

    kept: list[Detection] = []
    by_label: dict[str, list[Detection]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)

    for group in by_label.values():
        group = sorted(group, key=lambda d: d.score, reverse=True)
        boxes = np.array([d.bbox for d in group], dtype=np.float32)
        suppressed = np.zeros(len(group), dtype=bool)
        ious = iou_matrix(boxes, boxes)
        for i in range(len(group)):
            if suppressed[i]:
                continue
            kept.append(group[i])
            suppressed |= (ious[i] > iou_threshold) & (np.arange(len(group)) > i)
    return sorted(kept, key=lambda d: d.score, reverse=True)


def merge_vehicle_duplicates(
    detections: list[Detection], iou_threshold: float = 0.7
) -> list[Detection]:
    """Funde caixas do MESMO veiculo marcadas com classes diferentes.

    O NMS do YOLO e por classe: num video real o mesmo veiculo sai como "car"
    e "truck" (ou "bus" e "truck") com IoU de 0,98-0,99. Cada caixa virava uma
    trilha, as duas liam a mesma placa e a placa era publicada duas vezes.
    Aqui o NMS e entre classes, mas so entre as que portam placa (carro,
    caminhao, onibus, moto): pessoa sobre a moto ou bicicleta (IoU ~0,5) nao
    sao veiculos duplicados e passam intactos. Fica a caixa de maior score.
    ``iou_threshold <= 0`` desliga.
    """
    if iou_threshold <= 0 or len(detections) <= 1:
        return list(detections)
    from ..core.taxonomy import has_plate
    from ..utils.geometry import iou_matrix

    veiculos = sorted((d for d in detections if has_plate(d.label)),
                      key=lambda d: d.score, reverse=True)
    outros = [d for d in detections if not has_plate(d.label)]
    if len(veiculos) <= 1:
        return list(detections)
    boxes = np.array([d.bbox for d in veiculos], dtype=np.float32)
    ious = iou_matrix(boxes, boxes)
    suprimido = np.zeros(len(veiculos), dtype=bool)
    mantidos: list[Detection] = []
    for i in range(len(veiculos)):
        if suprimido[i]:
            continue
        mantidos.append(veiculos[i])
        suprimido |= (ious[i] > iou_threshold) & (np.arange(len(veiculos)) > i)
    return sorted(mantidos + outros, key=lambda d: d.score, reverse=True)
