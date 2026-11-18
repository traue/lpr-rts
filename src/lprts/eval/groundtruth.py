"""Leitura e escrita de anotacoes de referencia (ground truth).

Dois formatos aceitos, ambos legiveis por humanos:

**CSV** (recomendado para anotar a mao ou exportar de planilha)::

    frame_index,x1,y1,x2,y2,plate_text,object_class,track_id,kind,difficult
    120,880,540,1010,585,ABC1D23,carro,3,plate,0
    120,700,380,1180,640,,carro,3,object,0

Cada linha descreve UMA caixa num quadro:

* ``kind = plate`` - a caixa da **placa** (``plate_text`` obrigatorio);
* ``kind = object`` - a caixa do **objeto** (veiculo, pedestre, bicicleta...).

Sem a coluna ``kind``, uma linha com ``plate_text`` e placa e uma sem e objeto.
``track_id`` identifica o objeto ao longo dos quadros (a placa e o veiculo que a
porta tem o mesmo ``track_id``). Ele e o que permite medir por VEICULO e avaliar
o rastreamento; sem ele, o veiculo e identificado pelo texto da placa, o que so
e valido se nenhuma placa se repete no video.

``difficult = 1`` marca caixas que nao devem contar nem como acerto nem como
erro (placa ilegivel, oclusao severa) - protocolo PASCAL VOC. Declare no artigo
quantas foram marcadas e por que.

As coordenadas estao em pixels da imagem ORIGINAL, com ``(x1, y1)`` no canto
superior esquerdo. O texto da placa e normalizado na leitura (maiusculas, sem
hifen).

**JSON** (para ferramentas de anotacao e metadados extras)::

    {"video": "rodovia.mp4", "fps": 30.0,
     "annotations": [{"frame_index": 120, "bbox": [880,540,1010,585],
                      "plate_text": "ABC1D23", "object_class": "carro",
                      "track_id": 3, "kind": "plate"}]}
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..core.taxonomy import from_label
from ..ocr.plates_br import normalize

KINDS = ("plate", "object")


def _read_class(row: dict) -> str:
    """Le a classe do objeto, aceitando o nome antigo ``vehicle_class``."""
    raw = row.get("object_class") or row.get("vehicle_class") or "carro"
    return from_label(str(raw)).value


def _bool(valor: object) -> bool:
    return str(valor or "").strip().lower() in ("1", "true", "sim", "yes")


@dataclass(slots=True)
class GTAnnotation:
    """Uma caixa anotada num quadro (placa ou objeto)."""

    frame_index: int
    bbox: tuple[float, float, float, float]
    plate_text: str
    object_class: str = "carro"
    track_id: int | None = None
    difficult: bool = False      # nao conta como acerto nem como erro
    kind: str = ""               # "plate" | "object"; vazio = inferido

    def __post_init__(self) -> None:
        self.plate_text = normalize(self.plate_text)
        if not self.kind:
            self.kind = "plate" if self.plate_text else "object"
        if self.kind not in KINDS:
            raise ValueError(f"kind invalido: {self.kind!r} (use 'plate' ou 'object')")

    @property
    def vehicle_key(self) -> str:
        """Identidade do objeto: track_id quando anotado, senao o texto da placa."""
        if self.track_id is not None:
            return f"id:{self.track_id}"
        return f"placa:{self.plate_text}" if self.plate_text else f"obj:{self.frame_index}"

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])


@dataclass(slots=True)
class VehicleGT:
    """Um veiculo anotado: a placa dele em cada quadro em que aparece."""

    key: str
    plate_text: str
    object_class: str
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    difficult_frames: set[int] = field(default_factory=set)

    @property
    def first_frame(self) -> int:
        return min(self.boxes) if self.boxes else -1

    @property
    def last_frame(self) -> int:
        return max(self.boxes) if self.boxes else -1

    @property
    def n_frames(self) -> int:
        return len(self.boxes)

    @property
    def max_width(self) -> float:
        return max((b[2] - b[0] for b in self.boxes.values()), default=0.0)

    @property
    def all_difficult(self) -> bool:
        """Todas as aparicoes marcadas como dificeis: fora da metrica."""
        return bool(self.boxes) and self.difficult_frames >= set(self.boxes)


@dataclass(slots=True)
class GroundTruth:
    """Conjunto de anotacoes de um video."""

    video: str = ""
    fps: float = 30.0
    annotations: list[GTAnnotation] = field(default_factory=list)

    def by_frame(self) -> dict[int, list[GTAnnotation]]:
        index: dict[int, list[GTAnnotation]] = {}
        for ann in self.annotations:
            index.setdefault(ann.frame_index, []).append(ann)
        return index

    def plates(self) -> list[GTAnnotation]:
        return [a for a in self.annotations if a.kind == "plate"]

    def objects(self) -> list[GTAnnotation]:
        return [a for a in self.annotations if a.kind == "object"]

    def has_track_ids(self) -> bool:
        placas = self.plates()
        return bool(placas) and all(a.track_id is not None for a in placas)

    def vehicles(self) -> dict[str, VehicleGT]:
        """Veiculos com placa anotada, indexados pela identidade."""
        veiculos: dict[str, VehicleGT] = {}
        for a in self.plates():
            v = veiculos.get(a.vehicle_key)
            if v is None:
                v = VehicleGT(a.vehicle_key, a.plate_text, a.object_class)
                veiculos[a.vehicle_key] = v
            v.boxes[a.frame_index] = a.bbox
            if a.difficult:
                v.difficult_frames.add(a.frame_index)
        return veiculos

    def plate_text_conflicts(self) -> list[str]:
        """Identidades cuja placa muda entre quadros - erro de anotacao provavel."""
        textos: dict[str, set[str]] = {}
        for a in self.plates():
            textos.setdefault(a.vehicle_key, set()).add(a.plate_text)
        return sorted(k for k, t in textos.items() if len(t) > 1)

    def unique_plates(self, include_difficult: bool = False) -> set[str]:
        return {
            a.plate_text
            for a in self.annotations
            if a.plate_text and (include_difficult or not a.difficult)
        }

    def __len__(self) -> int:
        return len(self.annotations)


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Carrega anotacoes de ``.csv`` ou ``.json``, detectando pelo sufixo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de ground truth nao encontrado: {path}")
    if path.suffix.lower() == ".json":
        return _load_json(path)
    return _load_csv(path)


def _load_csv(path: Path) -> GroundTruth:
    annotations: list[GTAnnotation] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("frame_index"):
                continue
            annotations.append(
                GTAnnotation(
                    frame_index=int(float(row["frame_index"])),
                    bbox=(
                        float(row.get("x1", 0)), float(row.get("y1", 0)),
                        float(row.get("x2", 0)), float(row.get("y2", 0)),
                    ),
                    plate_text=row.get("plate_text", "") or "",
                    object_class=_read_class(row),
                    track_id=int(float(row["track_id"])) if row.get("track_id") else None,
                    difficult=_bool(row.get("difficult")),
                    kind=(row.get("kind") or "").strip().lower(),
                )
            )
    return GroundTruth(video=path.stem, annotations=annotations)


def _load_json(path: Path) -> GroundTruth:
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = [
        GTAnnotation(
            frame_index=int(item["frame_index"]),
            bbox=tuple(float(v) for v in item.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
            plate_text=item.get("plate_text", "") or "",
            object_class=from_label(
                str(item.get("object_class") or item.get("vehicle_class") or "carro")
            ).value,
            track_id=item.get("track_id"),
            difficult=bool(item.get("difficult", False)),
            kind=str(item.get("kind") or "").lower(),
        )
        for item in data.get("annotations", [])
    ]
    return GroundTruth(
        video=data.get("video", path.stem), fps=float(data.get("fps", 30.0)), annotations=annotations
    )


def save_ground_truth(gt: GroundTruth, path: str | Path) -> Path:
    """Grava anotacoes; o formato segue o sufixo do arquivo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(
                {
                    "video": gt.video,
                    "fps": gt.fps,
                    "annotations": [
                        {
                            "frame_index": a.frame_index, "bbox": list(a.bbox),
                            "plate_text": a.plate_text, "object_class": a.object_class,
                            "track_id": a.track_id, "difficult": a.difficult, "kind": a.kind,
                        }
                        for a in gt.annotations
                    ],
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["frame_index", "x1", "y1", "x2", "y2", "plate_text", "object_class",
             "track_id", "kind", "difficult"]
        )
        for a in gt.annotations:
            writer.writerow(
                [a.frame_index, *a.bbox, a.plate_text, a.object_class,
                 a.track_id if a.track_id is not None else "", a.kind, int(a.difficult)]
            )
    return path
