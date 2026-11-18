"""Filas de acoplamento entre estagios, com as duas politicas do sistema.

* ``DropOldestQueue`` - descarta o quadro *mais antigo* quando enche. E a peca
  central do comportamento de tempo real: quando a inferencia atrasa, latencia
  importa mais que completude. Conta os descartes, que vao para as metricas.
* ``BlockingQueue`` - bloqueia o produtor quando enche. E a politica correta
  para processamento offline, em que perder um quadro invalidaria a avaliacao.

As duas expoem a mesma interface, entao o pipeline escolhe pela configuracao
sem ramificar a logica.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")

_SENTINEL = object()


class BlockingQueue(Generic[T]):
    """Fila limitada que BLOQUEIA quando cheia, em vez de descartar.

    E a contrapartida da ``DropOldestQueue`` e a politica correta para
    processamento **offline**: ali completude vale mais que latencia, e perder um
    quadro invalidaria a avaliacao. O limite de tamanho continua existindo para
    conter o uso de memoria - um produtor rapido apenas espera o consumidor.

    Expoe a mesma interface da ``DropOldestQueue`` para que o pipeline escolha a
    politica pelo modo de execucao, sem ramificar o resto do codigo.
    """

    def __init__(self, maxsize: int = 8) -> None:
        if maxsize < 1:
            raise ValueError("maxsize deve ser >= 1")
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()

    def put(self, item: T) -> bool:
        """Bloqueia ate haver espaco. Retorna ``False`` se a fila fechou."""
        while not self._closed.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def get(self, timeout: float | None = None) -> T | None:
        try:
            return self._queue.get(timeout=timeout if timeout is not None else 0.1)
        except queue.Empty:
            return None

    def get_latest(self, timeout: float | None = None) -> T | None:
        """Sem descarte: identico a ``get``. Existe para igualar a interface."""
        return self.get(timeout)

    def close(self) -> None:
        self._closed.set()

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def dropped(self) -> int:
        return 0        # por construcao, nada e descartado

    def __len__(self) -> int:
        return self._queue.qsize()


class DropOldestQueue(Generic[T]):
    """Fila FIFO limitada; ao encher, descarta o item mais antigo."""

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize deve ser >= 1")
        self._buf: deque[T] = deque()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dropped = 0
        self._closed = False

    def put(self, item: T) -> bool:
        """Enfileira. Retorna ``False`` se um item antigo teve de ser descartado."""
        with self._not_empty:
            if self._closed:
                return False
            dropped = False
            while len(self._buf) >= self._maxsize:
                self._buf.popleft()
                self._dropped += 1
                dropped = True
            self._buf.append(item)
            self._not_empty.notify()
            return not dropped

    def get(self, timeout: float | None = None) -> T | None:
        """Desenfileira o item mais antigo. ``None`` em timeout ou fila fechada."""
        with self._not_empty:
            if not self._buf and not self._closed:
                self._not_empty.wait(timeout)
            if self._buf:
                return self._buf.popleft()
            return None

    def get_latest(self, timeout: float | None = None) -> T | None:
        """Desenfileira o item *mais recente*, descartando os anteriores.

        Usado pela interface: renderizar quadros velhos so aumenta a latencia
        percebida.
        """
        with self._not_empty:
            if not self._buf and not self._closed:
                self._not_empty.wait(timeout)
            if not self._buf:
                return None
            while len(self._buf) > 1:
                self._buf.popleft()
                self._dropped += 1
            return self._buf.popleft()

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
