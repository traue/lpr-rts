"""Gramatica, validacao e correcao de placas veiculares brasileiras.

Dois layouts convivem no Brasil desde 2018:

* **Mercosul** ``LLLNLNN`` - 3 letras, 1 digito, 1 letra, 2 digitos (ex.: ABC1D23).
* **Antigo**  ``LLLNNNN`` - 3 letras e 4 digitos (ex.: ABC1234).

Ambos tem exatamente 7 caracteres, o que da uma restricao forte: sabemos, para
cada posicao, se o caractere *deve* ser letra ou digito. Isso permite resolver a
ambiguidade visual classica do OCR (O/0, I/1, S/5, B/8, Z/2...) usando a posicao
como contexto, em vez de apostar so na confianca do modelo - tecnica que reduz
substancialmente o CER em placas de baixa resolucao.

Motos usam o mesmo alfabeto em duas linhas; o texto normalizado e identico.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.types import PlateLayout

# Padroes sintaticos (sobre texto ja normalizado: A-Z0-9, sem hifen/espaco)
RE_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
RE_LEGACY = re.compile(r"^[A-Z]{3}[0-9]{4}$")

PLATE_LENGTH = 7

# Maximo de caracteres que a mascara pode trocar. Com 3, lixo virava placa:
# "1231231" -> "IZB1231" e "0000000" -> "OOO0000", ambas "validas". Com 2, as
# confusoes tipicas de OCR (O/0, I/1, B/8 em uma ou duas posicoes) continuam
# corrigidas e a invencao de placas para. Configuravel em
# ``ocr.max_grammar_corrections`` para quem quiser estudar o compromisso.
DEFAULT_MAX_CORRECTIONS = 2

# Mascara de posicoes: 'L' = letra obrigatoria, 'N' = digito obrigatorio.
MASK_MERCOSUL = "LLLNLNN"
MASK_LEGACY = "LLLNNNN"

# Confusoes visuais mais frequentes em OCR de placas (ambos os sentidos).
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
}
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "T": "1", "J": "1",
    "Z": "2", "B": "8", "S": "5", "A": "4",
    "G": "6", "C": "0", "E": "8", "H": "4", "Y": "7",
}

# Caracteres que os padroes brasileiros nunca usam em nenhuma posicao.
INVALID_CHARS = set("")  # o alfabeto BR usa A-Z completo; mantido por clareza/extensao

_STRIP_RE = re.compile(r"[^A-Z0-9]")


@dataclass(slots=True)
class PlateCandidate:
    """Resultado da normalizacao/validacao de um texto bruto de OCR."""

    text: str
    layout: PlateLayout
    valid: bool
    corrections: int = 0        # quantos caracteres foram trocados pela mascara
    raw: str = ""

    @property
    def formatted(self) -> str:
        """Forma legivel: ``ABC1D23`` (Mercosul) ou ``ABC-1234`` (antigo)."""
        if self.layout is PlateLayout.LEGACY and len(self.text) == PLATE_LENGTH:
            return f"{self.text[:3]}-{self.text[3:]}"
        return self.text


def normalize(raw: str) -> str:
    """Remove acentos/separadores e deixa apenas ``A-Z0-9`` em maiuscula."""
    if not raw:
        return ""
    text = raw.upper()
    # Alguns motores devolvem 'BRASIL', 'MERCOSUL' ou a UF impressos na placa.
    for noise in ("BRASIL", "MERCOSUL", "BRAZIL"):
        text = text.replace(noise, "")
    return _STRIP_RE.sub("", text)


def detect_layout(text: str) -> PlateLayout:
    """Classifica o texto *ja normalizado* num dos layouts conhecidos."""
    if RE_MERCOSUL.match(text):
        return PlateLayout.MERCOSUL
    if RE_LEGACY.match(text):
        return PlateLayout.LEGACY
    return PlateLayout.UNKNOWN


def is_valid(text: str) -> bool:
    return detect_layout(text) is not PlateLayout.UNKNOWN


def _apply_mask(text: str, mask: str) -> tuple[str, int]:
    """Forca cada caractere ao tipo exigido pela mascara; conta as correcoes."""
    out: list[str] = []
    corrections = 0
    for ch, kind in zip(text, mask, strict=True):
        if kind == "L" and ch.isdigit():
            fixed = DIGIT_TO_LETTER.get(ch)
            if fixed is None:
                return text, -1  # sem correcao plausivel: candidato invalido
            out.append(fixed)
            corrections += 1
        elif kind == "N" and ch.isalpha():
            fixed = LETTER_TO_DIGIT.get(ch)
            if fixed is None:
                return text, -1
            out.append(fixed)
            corrections += 1
        else:
            out.append(ch)
    return "".join(out), corrections


def correct(text: str, max_corrections: int = DEFAULT_MAX_CORRECTIONS) -> PlateCandidate:
    """Normaliza e tenta encaixar o texto num dos dois layouts brasileiros.

    Estrategia: se o texto ja e valido, aceita sem tocar. Caso contrario, testa
    as duas mascaras e escolhe a que exigir *menos* correcoes - o palpite de
    menor risco. Correcoes acima de ``max_corrections`` sao rejeitadas: a essa
    altura estariamos inventando uma placa em vez de corrigir OCR.

    ``corrections`` do resultado diz quantos caracteres foram trocados. Ele vai
    para o banco: uma leitura "valida" depois de duas trocas e evidencia mais
    fraca que uma leitura valida de origem, e a analise precisa poder separa-las.
    """
    raw = text or ""
    normalized = normalize(raw)

    if len(normalized) != PLATE_LENGTH:
        layout = detect_layout(normalized)
        return PlateCandidate(normalized, layout, layout is not PlateLayout.UNKNOWN, 0, raw)

    layout = detect_layout(normalized)
    if layout is not PlateLayout.UNKNOWN:
        return PlateCandidate(normalized, layout, True, 0, raw)

    best: PlateCandidate | None = None
    for mask, expected in ((MASK_MERCOSUL, PlateLayout.MERCOSUL), (MASK_LEGACY, PlateLayout.LEGACY)):
        fixed, n_corr = _apply_mask(normalized, mask)
        if n_corr < 0 or n_corr > max_corrections:
            continue
        if detect_layout(fixed) is not expected:
            continue
        if best is None or n_corr < best.corrections:
            best = PlateCandidate(fixed, expected, True, n_corr, raw)

    if best is not None:
        return best
    return PlateCandidate(normalized, PlateLayout.UNKNOWN, False, 0, raw)


def salvage(text: str, max_corrections: int = DEFAULT_MAX_CORRECTIONS) -> str:
    """Extrai a subsequencia de 7 caracteres mais plausivel de um texto longo.

    Motores de OCR frequentemente colam a UF, o nome do municipio ou a tarja
    'BRASIL' ao codigo. Aqui varremos todas as janelas de 7 e devolvemos a
    primeira que fecha um layout valido.
    """
    return salvage_span(text, max_corrections)[0]


def salvage_span(text: str, max_corrections: int = DEFAULT_MAX_CORRECTIONS) -> tuple[str, int]:
    """Como ``salvage``, mas devolve tambem o inicio da janela no texto normalizado.

    A posicao permite recortar, na mesma janela, as confiancas por caractere que
    o motor devolveu - sem ela, as confiancas ficariam desalinhadas do texto.
    A janela devolvida e a do texto normalizado, *antes* da correcao.
    """
    normalized = normalize(text)
    if len(normalized) <= PLATE_LENGTH:
        return normalized, 0
    for start in range(len(normalized) - PLATE_LENGTH + 1):
        window = normalized[start : start + PLATE_LENGTH]
        if is_valid(window):
            return window, start
    for start in range(len(normalized) - PLATE_LENGTH + 1):
        window = normalized[start : start + PLATE_LENGTH]
        if correct(window, max_corrections).valid:
            return window, start
    return normalized[:PLATE_LENGTH], 0


def position_alphabet(index: int, layout: PlateLayout) -> str:
    """Alfabeto permitido numa posicao - usado para restringir o decodificador."""
    mask = MASK_MERCOSUL if layout is PlateLayout.MERCOSUL else MASK_LEGACY
    if not 0 <= index < len(mask):
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if mask[index] == "L" else "0123456789"
