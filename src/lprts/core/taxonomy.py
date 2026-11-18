"""Taxonomia de objetos detectaveis e suas propriedades.

O detector COCO reconhece 80 classes. Para analise de trafego, o que importa
nao e a lista crua, mas tres perguntas sobre cada objeto:

1. **A que categoria pertence?** Pessoa, veiculo motorizado, veiculo nao
   motorizado, animal ou mobiliario urbano. E o eixo natural de agregacao
   ("quantos pedestres versus quantos veiculos passaram").
2. **Tem placa?** Determina se vale gastar OCR. Uma bicicleta ou um pedestre
   nunca recebem OCR - no Brasil bicicletas nao sao emplacadas -, o que libera
   orcamento de tempo real para os objetos que de fato tem placa.
3. **E um usuario vulneravel da via?** Pedestres, ciclistas e motociclistas
   concentram a gravidade dos sinistros de transito. Marcar essa propriedade
   permite recortes de seguranca viaria diretamente nas consultas.

As tres propriedades ficam aqui, num unico lugar, em vez de espalhadas em
condicionais pelo pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ObjectCategory(str, Enum):
    """Agrupamento semantico usado nas estatisticas."""

    PESSOA = "pessoa"
    VEICULO_MOTORIZADO = "veiculo_motorizado"
    VEICULO_NAO_MOTORIZADO = "veiculo_nao_motorizado"
    ANIMAL = "animal"
    MOBILIARIO_URBANO = "mobiliario_urbano"
    OUTRO = "outro"


class ObjectClass(str, Enum):
    """Classes de interesse para analise de trafego (subconjunto do COCO)."""

    PESSOA = "pessoa"
    BICICLETA = "bicicleta"
    CARRO = "carro"
    MOTOCICLETA = "motocicleta"
    ONIBUS = "onibus"
    CAMINHAO = "caminhao"
    TREM = "trem"
    AVIAO = "aviao"
    BARCO = "barco"
    SEMAFORO = "semaforo"
    PLACA_PARE = "placa_pare"
    PARQUIMETRO = "parquimetro"
    BANCO = "banco"
    CACHORRO = "cachorro"
    GATO = "gato"
    CAVALO = "cavalo"
    MALA = "mala"
    MOCHILA = "mochila"
    GUARDA_CHUVA = "guarda_chuva"
    PLACA_VEICULAR = "placa_veicular"     # produzida pelo detector de placas
    DESCONHECIDO = "desconhecido"


@dataclass(frozen=True, slots=True)
class ClassInfo:
    """Propriedades de uma classe, consultadas pelo pipeline e pela analitica."""

    cls: ObjectClass
    category: ObjectCategory
    coco_id: int
    coco_name: str
    label_pt: str
    has_plate: bool = False
    vulnerable: bool = False       # usuario vulneravel da via
    is_vehicle: bool = False
    color: tuple[int, int, int] = (200, 200, 200)   # BGR, para a interface


# Indices COCO conforme os pesos oficiais do YOLO (person=0 ... ).
# Cores escolhidas para contrastar entre si e com asfalto/ceu.
_CLASSES: tuple[ClassInfo, ...] = (
    ClassInfo(ObjectClass.PESSOA, ObjectCategory.PESSOA, 0, "person", "pessoa",
              vulnerable=True, color=(80, 200, 255)),
    ClassInfo(ObjectClass.BICICLETA, ObjectCategory.VEICULO_NAO_MOTORIZADO, 1, "bicycle",
              "bicicleta", vulnerable=True, is_vehicle=True, color=(255, 200, 90)),
    ClassInfo(ObjectClass.CARRO, ObjectCategory.VEICULO_MOTORIZADO, 2, "car", "carro",
              has_plate=True, is_vehicle=True, color=(120, 220, 100)),
    # Motocicleta e o unico veiculo ao mesmo tempo emplacado e vulneravel.
    ClassInfo(ObjectClass.MOTOCICLETA, ObjectCategory.VEICULO_MOTORIZADO, 3, "motorcycle",
              "motocicleta", has_plate=True, vulnerable=True, is_vehicle=True,
              color=(255, 140, 200)),
    ClassInfo(ObjectClass.AVIAO, ObjectCategory.OUTRO, 4, "airplane", "aviao",
              is_vehicle=True, color=(180, 180, 180)),
    ClassInfo(ObjectClass.ONIBUS, ObjectCategory.VEICULO_MOTORIZADO, 5, "bus", "onibus",
              has_plate=True, is_vehicle=True, color=(90, 180, 250)),
    ClassInfo(ObjectClass.TREM, ObjectCategory.VEICULO_MOTORIZADO, 6, "train", "trem",
              is_vehicle=True, color=(160, 160, 220)),
    ClassInfo(ObjectClass.CAMINHAO, ObjectCategory.VEICULO_MOTORIZADO, 7, "truck", "caminhao",
              has_plate=True, is_vehicle=True, color=(70, 160, 220)),
    ClassInfo(ObjectClass.BARCO, ObjectCategory.OUTRO, 8, "boat", "barco",
              is_vehicle=True, color=(200, 160, 120)),
    ClassInfo(ObjectClass.SEMAFORO, ObjectCategory.MOBILIARIO_URBANO, 9, "traffic light",
              "semaforo", color=(120, 120, 255)),
    ClassInfo(ObjectClass.PLACA_PARE, ObjectCategory.MOBILIARIO_URBANO, 11, "stop sign",
              "placa de pare", color=(100, 100, 240)),
    ClassInfo(ObjectClass.PARQUIMETRO, ObjectCategory.MOBILIARIO_URBANO, 12, "parking meter",
              "parquimetro", color=(150, 150, 150)),
    ClassInfo(ObjectClass.BANCO, ObjectCategory.MOBILIARIO_URBANO, 13, "bench", "banco",
              color=(140, 140, 140)),
    ClassInfo(ObjectClass.GATO, ObjectCategory.ANIMAL, 15, "cat", "gato", color=(200, 220, 120)),
    ClassInfo(ObjectClass.CACHORRO, ObjectCategory.ANIMAL, 16, "dog", "cachorro",
              color=(180, 230, 140)),
    ClassInfo(ObjectClass.CAVALO, ObjectCategory.ANIMAL, 17, "horse", "cavalo",
              color=(160, 210, 160)),
    ClassInfo(ObjectClass.MOCHILA, ObjectCategory.OUTRO, 24, "backpack", "mochila",
              color=(190, 190, 190)),
    ClassInfo(ObjectClass.GUARDA_CHUVA, ObjectCategory.OUTRO, 25, "umbrella", "guarda-chuva",
              color=(190, 190, 190)),
    ClassInfo(ObjectClass.MALA, ObjectCategory.OUTRO, 28, "suitcase", "mala",
              color=(190, 190, 190)),
    ClassInfo(ObjectClass.PLACA_VEICULAR, ObjectCategory.OUTRO, -1, "plate", "placa",
              color=(60, 180, 255)),
    ClassInfo(ObjectClass.DESCONHECIDO, ObjectCategory.OUTRO, -2, "unknown", "desconhecido"),
)

_BY_CLASS: dict[ObjectClass, ClassInfo] = {c.cls: c for c in _CLASSES}
_BY_COCO_ID: dict[int, ClassInfo] = {c.coco_id: c for c in _CLASSES if c.coco_id >= 0}
_BY_COCO_NAME: dict[str, ClassInfo] = {c.coco_name: c for c in _CLASSES}
_BY_PT: dict[str, ClassInfo] = {c.cls.value: c for c in _CLASSES}

_DESCONHECIDO = _BY_CLASS[ObjectClass.DESCONHECIDO]


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------

def info(cls: ObjectClass | str | int) -> ClassInfo:
    """Propriedades de uma classe, aceita o enum, o nome (pt ou COCO) ou o id."""
    if isinstance(cls, ObjectClass):
        return _BY_CLASS.get(cls, _DESCONHECIDO)
    if isinstance(cls, int):
        return _BY_COCO_ID.get(cls, _DESCONHECIDO)
    key = str(cls).lower().strip()
    return _BY_PT.get(key) or _BY_COCO_NAME.get(key) or _DESCONHECIDO


def from_coco_id(idx: int) -> ObjectClass:
    return _BY_COCO_ID.get(idx, _DESCONHECIDO).cls


def from_label(label: str) -> ObjectClass:
    """Converte um rotulo de detector (COCO em ingles ou nome interno) na classe."""
    return info(label).cls


def category(cls: ObjectClass | str | int) -> ObjectCategory:
    return info(cls).category


def has_plate(cls: ObjectClass | str | int) -> bool:
    """True se a classe pode portar placa veicular no Brasil.

    Bicicletas ficam de fora de proposito: nao sao emplacadas. Motocicletas
    entram - a placa e traseira e, no padrao Mercosul, em duas linhas.
    """
    return info(cls).has_plate


def is_vulnerable(cls: ObjectClass | str | int) -> bool:
    return info(cls).vulnerable


def color(cls: ObjectClass | str | int) -> tuple[int, int, int]:
    return info(cls).color


def label_pt(cls: ObjectClass | str | int) -> str:
    return info(cls).label_pt


def classes_in(cat: ObjectCategory) -> list[ObjectClass]:
    return [c.cls for c in _CLASSES if c.category is cat]


def plate_bearing_classes() -> set[ObjectClass]:
    return {c.cls for c in _CLASSES if c.has_plate}


# ---------------------------------------------------------------------------
# Presets de deteccao
# ---------------------------------------------------------------------------

# Conjuntos prontos para o parametro `classes` do detector YOLO.
PRESET_VEICULOS = {c.coco_id: c.coco_name for c in _CLASSES if c.has_plate}

PRESET_TRAFEGO = {
    c.coco_id: c.coco_name
    for c in _CLASSES
    if c.coco_id >= 0
    and c.category
    in (
        ObjectCategory.PESSOA,
        ObjectCategory.VEICULO_MOTORIZADO,
        ObjectCategory.VEICULO_NAO_MOTORIZADO,
    )
}

PRESET_COMPLETO = {c.coco_id: c.coco_name for c in _CLASSES if c.coco_id >= 0}

PRESETS: dict[str, dict[int, str]] = {
    "veiculos": PRESET_VEICULOS,     # so o que tem placa: mais rapido
    "trafego": PRESET_TRAFEGO,       # pessoas + veiculos: analise de trafego
    "completo": PRESET_COMPLETO,     # tudo, inclusive mobiliario urbano
}


def preset(name: str) -> dict[int, str]:
    """Conjunto de classes COCO a detectar, pelo nome do preset."""
    key = (name or "trafego").lower().strip()
    if key not in PRESETS:
        raise ValueError(
            f"Preset de classes desconhecido: {name!r}. Opcoes: {sorted(PRESETS)}"
        )
    return PRESETS[key]
