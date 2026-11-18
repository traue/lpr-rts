"""Descoberta de fontes de video da maquina (cameras e streams).

Um cuidado guia todo este modulo: **listar sem abrir**. No macOS, abrir um
dispositivo de camera dispara o pedido de permissao do sistema. Fazer isso ao
montar a interface significaria pedir acesso a camera so porque o programa
abriu - intrusivo e desnecessario. Entao a enumeracao usa o catalogo do sistema
operacional (``system_profiler`` no macOS, ``/dev/video*`` no Linux) e so abre o
dispositivo quando o usuario de fato o seleciona.

A sondagem por forca bruta (tentar abrir os indices 0..N) existe como ultimo
recurso, desativada por padrao, para o caso de o catalogo nao estar disponivel.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VideoDevice:
    """Uma fonte de video disponivel."""

    index: int                  # indice aceito por cv2.VideoCapture
    name: str                   # nome legivel, do catalogo do sistema
    kind: str = "camera"        # "camera" | "arquivo" | "stream"
    path: str = ""              # caminho do dispositivo, quando existir
    detail: str = ""            # informacao extra (modelo, resolucao...)

    @property
    def uri(self) -> str:
        """Valor a passar para ``--source``."""
        return self.path if self.kind != "camera" else str(self.index)

    @property
    def label(self) -> str:
        base = f"Camera {self.index} - {self.name}" if self.kind == "camera" else self.name
        return f"{base} ({self.detail})" if self.detail else base


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------

def _macos_cameras() -> list[VideoDevice]:
    """Le o catalogo de cameras do macOS, sem abrir nenhuma delas.

    ``system_profiler SPCameraDataType`` lista as cameras conhecidas pelo
    sistema. A ordem que ele devolve corresponde, na pratica, a ordem dos
    indices do AVFoundation usados pelo OpenCV - o que nao e garantido por
    contrato, mas e o melhor mapeamento disponivel sem abrir os dispositivos.
    """
    if not shutil.which("system_profiler"):
        return []
    try:
        saida = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        dados = json.loads(saida.stdout or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        log.debug("system_profiler falhou: %s", exc)
        return []

    dispositivos: list[VideoDevice] = []
    for i, item in enumerate(dados.get("SPCameraDataType", [])):
        nome = str(item.get("_name") or f"Camera {i}")
        modelo = str(item.get("spcamera_model-id") or "")
        dispositivos.append(
            VideoDevice(index=i, name=nome, kind="camera", detail=modelo[:40])
        )
    return dispositivos


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

def _linux_cameras() -> list[VideoDevice]:
    """Enumera ``/dev/video*``, lendo o nome em ``/sys`` (sem abrir o dispositivo)."""
    dispositivos: list[VideoDevice] = []
    for caminho in sorted(Path("/dev").glob("video*")):
        m = re.match(r"video(\d+)$", caminho.name)
        if not m:
            continue
        indice = int(m.group(1))
        nome = f"video{indice}"
        sysfs = Path(f"/sys/class/video4linux/{caminho.name}/name")
        try:
            if sysfs.exists():
                nome = sysfs.read_text(encoding="utf-8").strip() or nome
        except OSError:
            pass
        # /dev/video1 costuma ser o canal de metadados do mesmo sensor; nao ha
        # como distinguir sem abrir, entao listamos e deixamos a escolha ao usuario.
        dispositivos.append(
            VideoDevice(index=indice, name=nome, kind="camera", path=str(caminho))
        )
    return dispositivos


# ---------------------------------------------------------------------------
# Sondagem (ultimo recurso)
# ---------------------------------------------------------------------------

def _probe_cameras(max_index: int = 5) -> list[VideoDevice]:
    """Tenta abrir os indices 0..max_index. **Abre dispositivos** - use com cautela.

    So faz sentido quando o catalogo do sistema nao esta disponivel. No macOS
    dispara o pedido de permissao; por isso nunca e chamada automaticamente.
    """
    import cv2

    dispositivos: list[VideoDevice] = []
    for indice in range(max_index + 1):
        cap = cv2.VideoCapture(indice)
        try:
            if not cap.isOpened():
                continue
            largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            detalhe = f"{largura}x{altura}" if largura and altura else ""
            dispositivos.append(
                VideoDevice(index=indice, name=f"Camera {indice}", detail=detalhe)
            )
        finally:
            cap.release()
    return dispositivos


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def list_cameras(probe: bool = False, max_index: int = 5) -> list[VideoDevice]:
    """Lista as cameras da maquina.

    Por padrao consulta apenas o catalogo do sistema, sem abrir dispositivo
    algum. Com ``probe=True``, tenta abrir os indices como ultimo recurso -
    o que, no macOS, dispara o pedido de permissao.
    """
    sistema = platform.system()
    if sistema == "Darwin":
        dispositivos = _macos_cameras()
    elif sistema == "Linux":
        dispositivos = _linux_cameras()
    else:
        dispositivos = []

    if not dispositivos and probe:
        log.info("Catalogo do sistema vazio; sondando os indices de camera.")
        dispositivos = _probe_cameras(max_index)
    return dispositivos


def check_camera(index: int, timeout_s: float = 3.0) -> tuple[bool, str]:
    """Abre a camera e tenta ler um quadro. Devolve ``(ok, mensagem)``.

    Chamada apenas sob acao explicita do usuario - e o momento legitimo para o
    macOS pedir permissao, porque ele acabou de escolher aquela camera.
    """
    import cv2

    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return False, (
                f"Nao foi possivel abrir a camera {index}. "
                "No macOS, autorize em Ajustes > Privacidade e Seguranca > Camera."
            )
        ok, quadro = cap.read()
        if not ok or quadro is None:
            return False, f"A camera {index} abriu, mas nao entregou nenhum quadro."
        h, w = quadro.shape[:2]
        return True, f"Camera {index} funcionando ({w}x{h})."
    except Exception as exc:  # noqa: BLE001
        return False, f"Erro ao acessar a camera {index}: {exc}"
    finally:
        cap.release()


def describe_sources() -> list[VideoDevice]:
    """Cameras encontradas mais as entradas genericas de arquivo e stream."""
    fontes = list_cameras()
    fontes.append(
        VideoDevice(index=-1, name="Arquivo de video...", kind="arquivo")
    )
    fontes.append(
        VideoDevice(index=-2, name="Stream de rede (RTSP/HTTP)...", kind="stream")
    )
    return fontes


def is_stream_uri(uri: str) -> bool:
    return str(uri).lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
    )


def classify_source(uri: str) -> str:
    """Classifica uma URI em ``camera``, ``stream`` ou ``arquivo``."""
    texto = str(uri).strip()
    if texto.isdigit():
        return "camera"
    if is_stream_uri(texto):
        return "stream"
    return "arquivo"


def is_synthetic_video(uri: str) -> bool:
    """Video gerado por ``scripts/make_synthetic_dataset.py``.

    O gerador grava ``<video>_meta.json`` ao lado do video. Os veiculos desses
    videos sao formas desenhadas, que um detector treinado em fotos reais nao
    reconhece: para eles a interface usa o modo so-placa. Qualquer outra fonte
    (video real, camera, stream) e tratada como real.
    """
    if classify_source(uri) != "arquivo" or not str(uri).strip():
        return False
    video = Path(str(uri).strip())
    meta = video.with_name(video.stem + "_meta.json")
    try:
        dados = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(dados, dict) and "make_synthetic_dataset" in str(dados.get("generator", ""))
