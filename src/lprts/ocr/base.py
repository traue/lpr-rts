"""Interface comum dos motores de OCR de placas."""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.types import PlateRead
from ..utils.logging import get_logger
from . import plates_br

log = get_logger(__name__)


class OCREngine(ABC):
    """Contrato: recebe o recorte da placa (BGR ou cinza) e devolve uma leitura.

    Motores de OCR de terceiros falham de formas variadas - API mudada, modelo
    ausente, entrada em formato inesperado. O contrato e nunca propagar a excecao
    (uma falha de leitura nao pode derrubar o pipeline), mas engolir em silencio
    e pior: o sintoma vira "nenhuma placa reconhecida", sem causa aparente.

    Por isso a falha e contada e, ao se repetir, reportada UMA vez como aviso,
    com o erro real. Falhar sempre e um defeito; falhar as vezes e a vida.

    Sobre ``confidence``: cada motor tem a sua escala (probabilidade media por
    caractere, score de deteccao de texto, correlacao reescalada...). Nao sao
    probabilidades calibradas nem comparaveis entre motores - limiares e modelos
    estatisticos que usem a confianca devem ser ajustados por motor.
    """

    name: str = "ocr"
    FALHAS_ATE_AVISAR: int = 5
    # Limite de trocas da gramatica; o factory aplica ocr.max_grammar_corrections.
    max_corrections: int = plates_br.DEFAULT_MAX_CORRECTIONS

    @abstractmethod
    def read(self, plate_crop: np.ndarray) -> PlateRead:
        """Le uma placa. Deve devolver ``PlateRead`` mesmo em caso de falha."""

    def read_batch(self, crops: list[np.ndarray]) -> list[PlateRead]:
        return [self.read(c) for c in crops]

    def warmup(self, runs: int = 1) -> None:
        dummy = np.full((64, 200, 3), 200, dtype=np.uint8)
        for _ in range(runs):
            try:
                self.read(dummy)
            except Exception:  # noqa: BLE001 - aquecimento e best-effort
                return

    def describe(self) -> dict[str, Any]:
        """Onde e como o motor roda de fato - vai para a proveniencia da sessao."""
        return {"name": self.name, "runtime": "?", "device": "?"}

    # -- visibilidade de falhas -------------------------------------------

    @property
    def falhas_consecutivas(self) -> int:
        return getattr(self, "_falhas", 0)

    @property
    def ultimo_erro(self) -> str:
        return getattr(self, "_ultimo_erro", "")

    def _registrar_falha(self, exc: BaseException) -> None:
        """Contabiliza uma falha e avisa quando ela deixa de ser esporadica."""
        self._falhas = getattr(self, "_falhas", 0) + 1
        self._ultimo_erro = f"{type(exc).__name__}: {exc}"
        log.debug("Motor de OCR '%s' falhou: %s", self.name, self._ultimo_erro)
        if self._falhas == self.FALHAS_ATE_AVISAR:
            log.warning(
                "O motor de OCR '%s' falhou em %d leituras seguidas: %s",
                self.name, self._falhas, self._ultimo_erro,
            )
            log.warning(
                "Nenhuma placa sera reconhecida enquanto isso persistir. "
                "Para contornar agora, use: --ocr template"
            )

    def _sucesso(self) -> None:
        if getattr(self, "_falhas", 0):
            log.info("Motor de OCR '%s' voltou a funcionar.", self.name)
        self._falhas = 0

    def close(self) -> None:
        """Libera o modelo subjacente.

        A implementacao base descarta as referencias mais comuns. Sessoes de
        inferencia seguram recursos nativos; libera-las explicitamente, enquanto
        o interpretador ainda esta vivo, evita a coleta em ordem arbitraria
        durante o encerramento.
        """
        for atributo in ("_model", "_ocr", "_reader", "_session"):
            if not hasattr(self, atributo):
                continue
            # Um motor pode expor o modelo como propriedade somente leitura ou
            # usar __slots__: nesses casos nao ha o que liberar, e insistir seria
            # transformar uma limpeza best-effort em erro.
            with contextlib.suppress(AttributeError):
                setattr(self, atributo, None)  # noqa: B010 - nome vem da lista acima


def empty_read(engine: str = "", raw: str = "") -> PlateRead:
    """Leitura vazia padronizada (falha de segmentacao, recorte degenerado...)."""
    return PlateRead(text="", confidence=0.0, engine=engine, raw_text=raw)


def build_read(
    raw_text: str,
    confidence: float,
    engine: str,
    t0: float,
    char_confs: list[float] | None = None,
    max_corrections: int = plates_br.DEFAULT_MAX_CORRECTIONS,
) -> PlateRead:
    """Normaliza, corrige pela gramatica BR e monta o ``PlateRead``.

    ``raw_text`` e guardado como veio do motor: a diferenca entre ele e ``text``
    mede a contribuicao da gramatica. As confiancas por caractere so sao mantidas
    quando podem ser alinhadas ao texto final - confianca desalinhada faria a
    fusao ponderar a posicao errada.
    """
    janela, inicio = plates_br.salvage_span(raw_text, max_corrections)
    candidate = plates_br.correct(janela, max_corrections=max_corrections)

    alinhadas: list[float] = []
    if char_confs:
        normalizado = plates_br.normalize(raw_text)
        if len(char_confs) == len(normalizado):
            alinhadas = [float(c) for c in char_confs[inicio : inicio + len(janela)]]
        if len(alinhadas) != len(candidate.text):
            alinhadas = []

    return PlateRead(
        text=candidate.text,
        confidence=float(confidence),
        char_confidences=alinhadas,
        layout=candidate.layout,
        valid_syntax=candidate.valid,
        engine=engine,
        ocr_ms=(time.perf_counter() - t0) * 1000.0,
        raw_text=raw_text,
        n_corrections=candidate.corrections,
    )
