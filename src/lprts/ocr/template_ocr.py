"""OCR classico por segmentacao + casamento de gabaritos (baseline).

Pipeline puramente determinista, sem rede neural e sem download de pesos:

1. normalizacao fotometrica do recorte (CLAHE + bilateral, em ``imgproc``);
2. binarizacao de Otsu (caracteres escuros sobre fundo claro - padrao BR);
3. segmentacao por componentes conexos, com filtros de altura/proporcao;
4. casamento de cada caractere contra gabaritos A-Z0-9 renderizados em varias
   fontes, por correlacao cruzada normalizada;
5. **decodificacao restrita pela gramatica**: em vez de escolher o melhor
   caractere isoladamente, escolhe-se a sequencia de maior soma de scores
   *dentro* de cada mascara brasileira (LLLNLNN / LLLNNNN), ficando com a
   mascara vencedora.

O passo 5 e o que torna este baseline competitivo: a restricao posicional
elimina de saida as confusoes O/0 e I/1 que dominam o erro em placas pequenas.

Precisao esperada: boa em placas frontais, nitidas e com >= 90 px de largura;
cai rapido com desfoque, perspectiva forte ou sujeira. Use um motor profundo
(``fast-plate-ocr``, ``PaddleOCR``) para os resultados finais e este como
referencia de comparacao.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..core.types import PlateRead
from ..utils.imgproc import deskew, enhance_plate
from . import plates_br
from .base import OCREngine, empty_read

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Resolucao do gabarito. Foi determinada empiricamente: a 20x32 os caracteres de
# contorno fechado e traco grosso (O/D/G/Q/0) empatam na correlacao, porque a
# grade e fina demais para resolver a diferenca entre um furo centrado (O) e um
# deslocado (D). A 32x48 a distincao reaparece. Os gabaritos sao cacheados e o
# casamento continua custando fracao de milissegundo.
CHAR_H, CHAR_W = 48, 32

# Fontes Hershey do OpenCV: nao sao a fonte oficial das placas (Mandatory/FE),
# mas cobrem variacao de traco suficiente para o casamento por correlacao.
#
# A ESPESSURA varia de proposito. Como os gabaritos sao normalizados para uma
# caixa fixa, um traco grosso preenche proporcionalmente mais area, e caracteres
# de contorno fechado passam a se parecer: um "O" grosso vira quase um "D". Ter
# gabaritos finos e grossos de cada caractere elimina essa confusao sistematica.
_FONTS = tuple(
    (font, 1.0, thickness)
    for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX)
    for thickness in (2, 4)
)


def _normalize_char(mask: np.ndarray) -> np.ndarray:
    """Encaixa o caractere numa caixa fixa PRESERVANDO a proporcao (letterbox).

    Esticar cada caractere ate CHAR_W x CHAR_H descarta a proporcao - que e
    exatamente o que separa um "O" (quase quadrado) de um "D" (mais estreito).
    Sem isso os dois empatam na correlacao e a escolha vira sorteio. Com o
    letterbox, a proporcao volta a ser uma caracteristica discriminante, desde
    que gabarito e consulta passem pela mesma normalizacao.
    """
    h, w = mask.shape[:2]
    if h < 1 or w < 1:
        return np.zeros((CHAR_H, CHAR_W), dtype=np.float32)
    escala = min(CHAR_W / w, CHAR_H / h)
    nova_w = max(1, int(round(w * escala)))
    nova_h = max(1, int(round(h * escala)))
    redimensionado = cv2.resize(mask, (nova_w, nova_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CHAR_H, CHAR_W), dtype=np.float32)
    ox = (CHAR_W - nova_w) // 2
    oy = (CHAR_H - nova_h) // 2
    canvas[oy : oy + nova_h, ox : ox + nova_w] = redimensionado.astype(np.float32)
    return canvas


@lru_cache(maxsize=1)
def _templates() -> dict[str, np.ndarray]:
    """Renderiza e cacheia o banco de gabaritos (uma vez por processo)."""
    bank: dict[str, np.ndarray] = {}
    for ch in ALPHABET:
        variants = []
        for font, scale, thickness in _FONTS:
            (tw, th), baseline = cv2.getTextSize(ch, font, scale, thickness)
            canvas = np.zeros((th + baseline + 8, tw + 8), dtype=np.uint8)
            cv2.putText(canvas, ch, (4, th + 4), font, scale, 255, thickness, cv2.LINE_AA)
            coords = cv2.findNonZero(canvas)
            if coords is None:
                continue
            x, y, w, h = cv2.boundingRect(coords)
            variants.append(_normalize_char(canvas[y : y + h, x : x + w]))
        if variants:
            bank[ch] = np.stack(variants)
    return bank


@lru_cache(maxsize=1)
def _template_matrix() -> tuple[np.ndarray, list[str]]:
    """Empilha os gabaritos ja centrados e normalizados numa unica matriz.

    Todas as variantes viram linhas de uma matriz ``(n_variantes, CHAR_H*CHAR_W)``
    com media zero e norma unitaria. Assim a correlacao cruzada normalizada de um
    caractere contra o banco inteiro vira **um unico produto matriz-vetor**, em vez
    de 216 laos em Python. E a diferenca entre ~20 ms e menos de 1 ms por placa -
    decisiva para o orcamento de tempo real.
    """
    linhas: list[np.ndarray] = []
    rotulos: list[str] = []
    for ch, variants in _templates().items():
        for tpl in variants:
            v = tpl.ravel().astype(np.float32)
            v = v - v.mean()
            norma = float(np.linalg.norm(v))
            if norma < 1e-6:
                continue
            linhas.append(v / norma)
            rotulos.append(ch)
    if not linhas:
        return np.zeros((0, CHAR_H * CHAR_W), dtype=np.float32), []
    return np.stack(linhas), rotulos


def _match_scores(char_img: np.ndarray) -> dict[str, float]:
    """Score em [0,1] de cada caractere do alfabeto para um recorte segmentado."""
    patch = _normalize_char(char_img).ravel()
    patch = patch - patch.mean()
    norma = float(np.linalg.norm(patch))
    if norma < 1e-6:
        return dict.fromkeys(ALPHABET, 0.0)

    matriz, rotulos = _template_matrix()
    if matriz.size == 0:
        return dict.fromkeys(ALPHABET, 0.0)

    # Correlacao de todas as variantes de uma vez; reescala de [-1,1] para [0,1].
    correlacoes = (matriz @ (patch / norma) + 1.0) * 0.5

    scores = dict.fromkeys(ALPHABET, 0.0)
    for rotulo, valor in zip(rotulos, correlacoes, strict=True):
        # Cada caractere fica com a melhor de suas variantes de fonte/espessura.
        if valor > scores[rotulo]:
            scores[rotulo] = float(valor)
    return scores


def segment_characters(plate_gray: np.ndarray, expected: int = 7) -> list[np.ndarray]:
    """Segmenta caracteres do recorte de placa por componentes conexos.

    Devolve mascaras binarias (caractere branco sobre fundo preto), na ordem
    de leitura.

    Trata tambem placas de moto (duas linhas): componentes sao agrupados por
    faixa vertical e depois ordenados linha a linha, da esquerda para a direita.
    """
    if plate_gray.size == 0 or plate_gray.shape[0] < 12:
        return []
    binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Se o fundo ficou branco, a polaridade estava invertida (placa escura).
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)

    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h_img, w_img = plate_gray.shape[:2]

    # Passada 1: filtros frouxos, apenas para descartar ruido evidente e a moldura.
    raw_boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.12 * h_img or h > 0.98 * h_img:
            continue
        if w < 2 or w > 0.45 * w_img:
            continue
        if not (0.08 <= w / max(h, 1) <= 1.6):
            continue
        if area < 0.10 * w * h:
            continue
        raw_boxes.append((int(x), int(y), int(w), int(h)))

    if not raw_boxes:
        return []

    # Passada 2: caracteres de uma placa tem altura praticamente identica.
    # Agrupar pela altura MEDIANA e bem mais robusto que uma fracao fixa de
    # h_img, que depende de quanta margem o detector deixou no recorte.
    median_h = float(np.median([b[3] for b in raw_boxes]))
    boxes = [
        b for b in raw_boxes if 0.62 * median_h <= b[3] <= 1.45 * median_h
    ]

    if not boxes:
        return []

    # Agrupa em linhas pelo centro vertical (placas de moto tem duas).
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    lines: list[list[tuple[int, int, int, int]]] = [[boxes[0]]]
    for box in boxes[1:]:
        ref = lines[-1][-1]
        if abs((box[1] + box[3] / 2) - (ref[1] + ref[3] / 2)) > 0.5 * max(ref[3], box[3]):
            lines.append([box])
        else:
            lines[-1].append(box)

    ordered: list[tuple[int, int, int, int]] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda b: b[0]))

    # Mais componentes que o esperado: fica com os de maior area (ruido e menor).
    if len(ordered) > expected:
        keep = sorted(ordered, key=lambda b: b[2] * b[3], reverse=True)[:expected]
        ordered = [b for b in ordered if b in keep]

    # Devolve a MASCARA BINARIA (caractere branco sobre fundo preto), que e a
    # mesma polaridade dos gabaritos - casar cinza com gabarito branco daria
    # correlacao negativa - e ainda elimina a dependencia de iluminacao.
    return [binary[y : y + h, x : x + w] for (x, y, w, h) in ordered]


def _decode_with_masks(
    per_char: list[dict[str, float]], max_corrections: int | None = None,
) -> tuple[str, float, list[float], plates_br.PlateLayout]:
    """Escolhe a sequencia de maior score dentro de cada mascara BR e compara."""
    from ..core.types import PlateLayout

    best_text, best_mean, best_chars, best_layout = "", 0.0, [], PlateLayout.UNKNOWN
    livre, _ = _decode_free(per_char)
    masks = (
        (plates_br.MASK_MERCOSUL, PlateLayout.MERCOSUL),
        (plates_br.MASK_LEGACY, PlateLayout.LEGACY),
    )
    for mask, layout in masks:
        if len(per_char) != len(mask):
            continue
        chars, confs = [], []
        for scores, kind in zip(per_char, mask, strict=True):
            pool = (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if kind == "L" else "0123456789"
            )
            ch = max(pool, key=lambda c: scores.get(c, 0.0))
            chars.append(ch)
            confs.append(scores.get(ch, 0.0))
        # A mascara tambem e uma correcao: compare-a ao OCR livre antes de
        # aceitar um texto que ja tera sintaxe valida em plates_br.correct.
        trocas = sum(a != b for a, b in zip(livre, chars, strict=True))
        if max_corrections is not None and trocas > max_corrections:
            continue
        mean = sum(confs) / len(confs)
        if mean > best_mean:
            best_text, best_mean, best_chars, best_layout = "".join(chars), mean, confs, layout

    if not best_text:  # comprimento fora do padrao: decodificacao livre
        best_text, best_chars = _decode_free(per_char)
        best_mean = sum(best_chars) / len(best_chars) if best_chars else 0.0
    return best_text, best_mean, best_chars, best_layout


def _decode_free(per_char: list[dict[str, float]]) -> tuple[str, list[float]]:
    """Melhor caractere de cada posicao, sem nenhuma restricao de gramatica."""
    chars = [max(ALPHABET, key=lambda c: s.get(c, 0.0)) for s in per_char]
    confs = [s.get(c, 0.0) for s, c in zip(per_char, chars, strict=True)]
    return "".join(chars), confs


class TemplateOCR(OCREngine):
    """Motor de OCR classico - baseline sem dependencias externas.

    A confianca e a correlacao media reescalada de [-1, 1] para [0, 1]: um
    caractere sem nenhuma correlacao ja vale 0,5. Nao e probabilidade e nao se
    compara com a confianca dos motores neurais.
    """

    name = "template"

    def __init__(self, target_height: int = 64, min_confidence: float = 0.0) -> None:
        self.target_height = target_height
        self.min_confidence = min_confidence
        _templates()  # pre-carrega o banco de gabaritos

    def describe(self) -> dict:
        import hashlib

        matriz, rotulos = _template_matrix()
        # O banco de gabaritos e o "modelo" deste motor: o hash identifica a
        # versao exata (fontes, espessuras, resolucao) que produziu os numeros.
        digest = hashlib.sha256(matriz.tobytes() + "".join(rotulos).encode()).hexdigest()[:16]
        return {"name": self.name, "runtime": "numpy/opencv", "device": "cpu",
                "templates_sha256": digest, "n_templates": len(rotulos)}

    def read(self, plate_crop: np.ndarray) -> PlateRead:
        import time

        t0 = time.perf_counter()
        if plate_crop is None or plate_crop.size == 0:
            return empty_read(self.name)

        gray = enhance_plate(deskew(plate_crop), self.target_height)
        chars = segment_characters(gray, expected=plates_br.PLATE_LENGTH)
        if not chars:
            return empty_read(self.name)

        per_char = [_match_scores(c) for c in chars]
        text, mean_conf, char_confs, _ = _decode_with_masks(per_char, self.max_corrections)
        # raw_text e a decodificacao LIVRE (melhor caractere por posicao). Antes
        # ele recebia o texto ja restrito pela mascara, o que tornava impossivel
        # medir quanto a gramatica contribui neste motor.
        livre, _ = _decode_free(per_char)

        janela, inicio = plates_br.salvage_span(text, self.max_corrections)
        candidate = plates_br.correct(janela, max_corrections=self.max_corrections)
        confs = [float(c) for c in char_confs[inicio : inicio + len(janela)]]
        if len(confs) != len(candidate.text):
            confs = []
        trocas = (
            sum(1 for a, b in zip(livre, candidate.text, strict=True) if a != b)
            if len(livre) == len(candidate.text)
            else candidate.corrections
        )
        return PlateRead(
            text=candidate.text,
            confidence=float(mean_conf),
            char_confidences=confs,
            layout=candidate.layout,
            valid_syntax=candidate.valid,
            engine=self.name,
            ocr_ms=(time.perf_counter() - t0) * 1000.0,
            raw_text=livre,
            n_corrections=trocas,
        )
