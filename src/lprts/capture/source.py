"""Fontes de video: arquivo, webcam e stream de rede (RTSP/HTTP).

Duas politicas de leitura, escolhidas pelo modo de operacao:

* **offline** (``realtime=False``): le todo quadro, na velocidade que o consumidor
  aguentar. E o modo correto para *avaliacao cientifica* - nenhum quadro perdido,
  resultados reproduziveis.
* **tempo real** (``realtime=True``): a thread de captura le continuamente e
  publica na ``DropOldestQueue``; quadros atrasados sao descartados. Para arquivo,
  respeita o FPS nominal para simular uma camera ao vivo.

Base de tempo
-------------
Cada quadro carrega dois relogios:

* ``source_pts`` - **tempo de midia**, a base de toda medida fisica. Em arquivo,
  e o carimbo do proprio container (PTS), ou indice/fps quando ausente. Em
  camera ou stream ao vivo, e o instante de chegada medido desde a abertura da
  fonte: o PTS de RTSP pode comecar em qualquer valor ou saltar numa
  reconexao, e o de webcam varia por driver.
* ``timestamp`` - relogio de processamento (``time.monotonic()``) no instante em
  que o quadro ficou disponivel; serve so para medir latencia.

Uma versao anterior usava ``timestamp`` como base fisica: a mesma trilha saia a
219 km/h numa maquina rapida e a 17 km/h numa lenta.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..core.types import Frame
from ..utils.logging import get_logger
from .ringbuffer import BlockingQueue, DropOldestQueue

log = get_logger(__name__)


@dataclass(slots=True)
class RawFrame:
    """Quadro decodificado, antes da reducao e da rotacao (``VideoSource.prepare``)."""

    index: int
    image: np.ndarray
    arrival: float          # time.monotonic() ao fim da decodificacao (latencia)
    media_s: float          # tempo de midia
    decode_ms: float


# Metadado de rotacao do container -> codigo do cv2.rotate que o aplica.
_ROTACOES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _looks_like_stream(uri: str) -> bool:
    return uri.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


class VideoSource:
    """Envolve ``cv2.VideoCapture`` com metadados, reconexao e modo tempo real."""

    def __init__(
        self,
        uri: str | int | Path,
        realtime: bool = False,
        queue_size: int = 2,
        target_fps: float | None = None,
        resize_width: int | None = None,
        reconnect_attempts: int = 5,
        keep_original: bool = False,
        backend: str = "auto",
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
    ) -> None:
        if backend not in ("auto", "ffmpeg", "avfoundation"):
            raise ValueError(f"Backend de video desconhecido: {backend!r} "
                             "(use auto, ffmpeg ou avfoundation).")
        self.uri = str(uri) if not isinstance(uri, int) else uri
        self.backend = backend
        self.backend_name = ""        # o que o OpenCV de fato usou (gravado na sessao)
        # Video gravado de lado (celular na vertical): reduz o quadro cru e gira so
        # a imagem pequena, em vez de girar o 4K inteiro (ver open).
        self._rotacao: int | None = None
        self.realtime = realtime
        self.target_fps = target_fps
        self.resize_width = resize_width
        # Guarda o quadro original junto do reduzido (Frame.full_image).
        self.keep_original = keep_original
        self.reconnect_attempts = reconnect_attempts
        if open_timeout_ms <= 0 or read_timeout_ms <= 0:
            raise ValueError("Timeouts de captura devem ser > 0")
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms

        self._cap: cv2.VideoCapture | None = None
        self._queue: DropOldestQueue[Frame] | BlockingQueue[Frame] = (
            DropOldestQueue(queue_size) if realtime else BlockingQueue(queue_size)
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._capture_error: Exception | None = None
        self._read_started_at: float | None = None
        self._frame_index = 0
        # Origem do tempo de midia das fontes ao vivo. Fixada na PRIMEIRA abertura
        # e mantida nas reconexoes: a linha do tempo da sessao e continua.
        self._t0: float | None = None
        self.start_utc: datetime | None = None
        self.reconnections = 0

        self.fps: float = 0.0
        self.width: int = 0           # resolucao ORIGINAL da fonte
        self.height: int = 0
        self.frame_count: int = 0
        self.is_stream: bool = isinstance(uri, int) or _looks_like_stream(str(uri))
        # Webcam (indice) e stream: o tempo de midia e o de chegada.
        if isinstance(self.uri, str) and self.uri.isdigit():
            self.is_stream = True

    # -- ciclo de vida -----------------------------------------------------

    def open(self) -> VideoSource:
        src: str | int = self.uri
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, str) and not _looks_like_stream(src) and not Path(src).exists():
            raise FileNotFoundError(f"Fonte de video nao encontrada: {src}")

        cap = self._abrir(src)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {src}")
        self.backend_name = cap.getBackendName()

        # Buffer minimo no driver: evita acumular latencia em cameras ao vivo.
        if self.is_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._preparar_rotacao(cap)
        self.frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if self._t0 is None:
            self._t0 = time.monotonic()
            if self.is_stream:
                # Instante real do inicio da linha do tempo ao vivo: com ele cada
                # evento ganha um horario absoluto (media_time_utc).
                self.start_utc = datetime.now(timezone.utc)
        log.info(
            "Fonte aberta: %s (%dx%d @ %.2f fps, %s quadros, %s, %s)",
            src, self.width, self.height, self.fps,
            self.frame_count or "?", "ao vivo" if self.is_stream else "arquivo",
            self.backend_name,
        )
        return self

    def _preparar_rotacao(self, cap: cv2.VideoCapture) -> None:
        """Video de celular na vertical: o quadro 4K chega deitado e o OpenCV o gira.

        Girar o quadro inteiro e reduzi-lo em retrato custava 5,6 + 7,8 ms por
        quadro; reduzir o quadro cru e girar so a imagem reduzida custa 2,4 ms. So
        quando ha reducao (``resize_width``) - sem ela a imagem inteira e usada
        girada de qualquer forma. As dimensoes da fonte continuam as do video
        girado, que e o que se ve e onde se anota o ground truth.
        """
        self._rotacao = None
        if self.is_stream or not self.resize_width or not hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            return
        meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
        codigo = _ROTACOES.get(meta)
        if codigo is None or self.resize_width >= self.width:
            return
        if not cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0):
            return
        self._rotacao = codigo

    def _abrir(self, src: str | int) -> cv2.VideoCapture:
        """Abre o arquivo pelo decodificador de hardware do Mac quando possivel.

        O FFmpeg do OpenCV decodifica em software: um quadro 4K HEVC de iPhone,
        com a reducao para 1280 px, custava 46-48 ms (o prazo a 30 qps e 33 ms) e
        a captura passava a ser o gargalo do tempo real. Pelo AVFoundation
        (VideoToolbox) o mesmo quadro custa 8 ms, com os mesmos quadros, a mesma
        rotacao e os mesmos tempos de midia (diferenca media de pixel ~1,5/255, da
        conversao de cor). Cameras e streams seguem o padrao do OpenCV.
        """
        if isinstance(src, str) and _looks_like_stream(src):
            # Estes parametros sao de abertura (set apos open nao os aplica).
            # Nunca fazer fallback para um backend sem limites silenciosamente.
            return cv2.VideoCapture(src, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms,
            ])
        arquivo = isinstance(src, str) and not self.is_stream
        if (arquivo and self.backend in ("auto", "avfoundation") and sys.platform == "darwin"
                and hasattr(cv2, "CAP_AVFOUNDATION")):
            cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
            cap.release()
            if self.backend == "avfoundation":
                raise RuntimeError(f"O AVFoundation nao abriu {src}; use backend ffmpeg.")
            log.info("AVFoundation nao abriu %s; usando o FFmpeg.", src)
        if arquivo and self.backend == "ffmpeg":
            return cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(src)

    def cancel(self) -> None:
        """Interrompe consumidores/esperas, sem liberar um handle nativo em uso."""
        self._stop.set()
        self._queue.close()

    def close(self) -> None:
        self.cancel()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise TimeoutError("Captura ainda em leitura nativa; cancelamento solicitado")
        self._release_capture()

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- leitura -----------------------------------------------------------

    @property
    def processing_size(self) -> tuple[int, int]:
        """Dimensoes da imagem entregue ao pipeline (apos ``resize_width``)."""
        if self.resize_width and self.width and self.width != self.resize_width:
            escala = self.resize_width / self.width
            return self.resize_width, int(round(self.height * escala))
        return self.width, self.height

    def _read_one(self) -> Frame | None:
        cru = self._read_raw()
        return None if cru is None else self.prepare(cru)

    def _read_raw(self) -> RawFrame | None:
        """Decodifica o proximo quadro; o preparo (reducao, rotacao) fica em ``prepare``."""
        assert self._cap is not None
        t_ini = time.perf_counter()
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        chegada = time.monotonic()
        if self.is_stream:
            media = chegada - (self._t0 if self._t0 is not None else chegada)
        else:
            pts = float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            media = pts if pts > 0 else self._frame_index / max(self.fps, 1e-6)
        cru = RawFrame(self._frame_index, image, chegada, media,
                       (time.perf_counter() - t_ini) * 1000.0)
        self._frame_index += 1
        return cru

    def prepare(self, cru: RawFrame) -> Frame:
        """Reduz e gira um quadro decodificado. Sem estado: roda em qualquer thread.

        No video 4K vertical e 84% da captura (6,5 de 7,8 ms: a escala 2160 -> 1280
        nao e inteira e tira o INTER_AREA do caminho rapido); o pipeline paralelo
        offline o faz nos workers, com os mesmos pixels.
        """
        t_ini = time.perf_counter()
        image = cru.image
        orig_h, orig_w = image.shape[:2]
        scale = 1.0
        original = None
        rotacao = self._rotacao
        if rotacao is not None:
            # Quadro cru (deitado): as dimensoes "originais" sao as do video girado.
            deitado = image
            if rotacao != cv2.ROTATE_180:
                orig_w, orig_h = orig_h, orig_w
            assert self.resize_width is not None
            scale = self.resize_width / orig_w
            alvo_w, alvo_h = self.resize_width, int(round(orig_h * scale))
            if rotacao != cv2.ROTATE_180:
                alvo_w, alvo_h = alvo_h, alvo_w
            image = cv2.rotate(
                cv2.resize(deitado, (alvo_w, alvo_h), interpolation=cv2.INTER_AREA), rotacao
            )
            if self.keep_original:
                original = deitado
        elif self.resize_width and orig_w != self.resize_width:
            scale = self.resize_width / orig_w
            if self.keep_original and scale < 1.0:
                original = image
            image = cv2.resize(
                image,
                (self.resize_width, int(round(orig_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return Frame(
            index=cru.index,
            image=image,
            timestamp=cru.arrival,
            source_pts=cru.media_s,
            scale=scale,
            orig_width=orig_w,
            orig_height=orig_h,
            # Captura = decodificar + preparar, onde quer que o preparo rode.
            decode_ms=cru.decode_ms + (time.perf_counter() - t_ini) * 1000.0,
            full_image=original,
            full_rotation=rotacao if original is not None else None,
        )

    def _reconnect(self) -> bool:
        """Tenta reabrir a fonte com backoff exponencial (streams caem)."""
        for attempt in range(1, self.reconnect_attempts + 1):
            delay = min(2**attempt, 16)
            log.warning("Stream perdido; reconectando em %ds (tentativa %d/%d)",
                        delay, attempt, self.reconnect_attempts)
            if self._stop.wait(delay):
                return False
            self._release_capture()
            try:
                self.open()
                self.reconnections += 1
                return True
            except Exception as exc:  # noqa: BLE001 - reconexao e best-effort
                log.warning("Reconexao falhou: %s", exc)
        return False

    def _capture_loop(self) -> None:
        """Thread de captura no modo tempo real."""
        frame_interval = 1.0 / self.target_fps if self.target_fps else 0.0
        # Arquivos sao lidos no ritmo nominal para simular uma camera ao vivo.
        if not self.is_stream and frame_interval == 0.0:
            frame_interval = 1.0 / max(self.fps, 1e-6)
        next_deadline = time.monotonic()

        try:
            while not self._stop.is_set():
                self._read_started_at = time.monotonic()
                try:
                    frame = self._read_one()
                finally:
                    self._read_started_at = None
                if frame is None:
                    if self.is_stream and not self._stop.is_set():
                        if self._reconnect():
                            continue
                        if not self._stop.is_set():
                            raise ConnectionError("Fonte ao vivo perdida; reconexoes esgotadas")
                    break
                self._queue.put(frame)

                if frame_interval > 0:
                    next_deadline += frame_interval
                    sleep_for = next_deadline - time.monotonic()
                    if sleep_for > 0:
                        self._stop.wait(sleep_for)
                    else:
                        next_deadline = time.monotonic()
        except Exception as exc:  # a falha e entregue na thread consumidora
            self._capture_error = exc
        finally:
            self._queue.close()
            if self._stop.is_set():
                self._release_capture()

    def start(self) -> VideoSource:
        """Inicia captura desacoplada para tempo real ou fontes ao vivo."""
        if not self.realtime and not self.is_stream:
            return self
        if self._thread is not None:
            return self
        if self._cap is None:
            self.open()
        self._thread = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        self._thread.start()
        return self

    def frames(self):
        """Itera sobre os quadros conforme a politica configurada."""
        if self._cap is None:
            self.open()
        if self.realtime or self.is_stream:
            if self._thread is None:
                self.start()
            while not self._stop.is_set():
                frame = self._queue.get(timeout=0.1)
                if frame is None:
                    if self._capture_error is not None:
                        raise RuntimeError("Falha na captura: " + str(self._capture_error)) from self._capture_error
                    if self._queue.closed or self._stop.is_set():
                        return
                    started = self._read_started_at
                    if (self.is_stream and started is not None
                            and time.monotonic() - started > self.read_timeout_ms / 1000):
                        self.cancel()
                        raise TimeoutError("Captura excedeu read_timeout_ms")
                    continue
                yield frame
        else:
            while not self._stop.is_set():
                frame = self._read_one()
                if frame is None:
                    return
                yield frame

    @property
    def supports_raw_frames(self) -> bool:
        """Arquivo lido offline: o consumidor pode preparar cada quadro (``prepare``)."""
        return not (self.realtime or self.is_stream)

    def raw_frames(self) -> Iterator[RawFrame]:
        """Quadros so decodificados, em ordem (offline); ``prepare`` completa cada um."""
        if not self.supports_raw_frames:
            raise RuntimeError("raw_frames so vale para arquivo offline")
        if self._cap is None:
            self.open()
        while not self._stop.is_set():
            cru = self._read_raw()
            if cru is None:
                return
            yield cru

    @property
    def dropped(self) -> int:
        return self._queue.dropped

    def seek(self, frame_index: int) -> bool:
        """Reposiciona a leitura (somente arquivos).

        A busca do AVFoundation em HEVC para perto de um quadro-chave, nao no
        quadro pedido (medido: pedido 300, entregue ~294): para buscar, a fonte e
        reaberta pelo FFmpeg, que acerta o quadro.
        """
        if self._cap is None or self.is_stream:
            return False
        if self.backend_name == "AVFOUNDATION":
            self._cap.release()
            self._cap = cv2.VideoCapture(str(self.uri), cv2.CAP_FFMPEG)
            self.backend_name = self._cap.getBackendName()
            self._preparar_rotacao(self._cap)
        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        if ok:
            self._frame_index = frame_index
        return bool(ok)


class ImageFolderSource:
    """Le uma pasta de imagens como se fosse video - util para datasets anotados."""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self, folder: str | Path, fps: float = 30.0, resize_width: int | None = None,
        keep_original: bool = False,
    ) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Pasta de imagens nao encontrada: {folder}")
        self.paths = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise ValueError(f"Nenhuma imagem suportada em {folder}")
        self.fps = fps
        self.realtime = False
        self.is_stream = False
        self.resize_width = resize_width
        self.keep_original = keep_original
        self.start_utc = None
        self._stop = threading.Event()
        first = cv2.imread(str(self.paths[0]))
        if first is None:
            raise ValueError(f"Imagem ilegivel: {self.paths[0]}")
        self.height, self.width = first.shape[:2]
        self.frame_count = len(self.paths)
        self.dropped = 0

    @property
    def processing_size(self) -> tuple[int, int]:
        if self.resize_width and self.width != self.resize_width:
            escala = self.resize_width / self.width
            return self.resize_width, int(round(self.height * escala))
        return self.width, self.height

    def open(self) -> ImageFolderSource:
        return self

    def cancel(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.cancel()

    def __enter__(self) -> ImageFolderSource:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def frames(self):
        for i, path in enumerate(self.paths):
            if self._stop.is_set():
                return
            t_ini = time.perf_counter()
            image: np.ndarray | None = cv2.imread(str(path))
            if image is None:
                log.warning("Imagem ilegivel, ignorada: %s", path)
                continue
            orig_h, orig_w = image.shape[:2]
            scale = 1.0
            original = None
            if self.resize_width and orig_w != self.resize_width:
                scale = self.resize_width / orig_w
                if self.keep_original and scale < 1.0:
                    original = image
                image = cv2.resize(
                    image, (self.resize_width, int(round(orig_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            yield Frame(
                index=i, image=image, timestamp=time.monotonic(), source_pts=i / self.fps,
                scale=scale, orig_width=orig_w, orig_height=orig_h,
                decode_ms=(time.perf_counter() - t_ini) * 1000.0, full_image=original,
            )
