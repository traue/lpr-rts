"""Escolha do dispositivo de inferencia local (sem servicos externos).

Ordem de preferencia no macOS com Apple Silicon:

1. **CoreML** - executa no Neural Engine (ANE) e na GPU. E o caminho mais rapido
   e mais eficiente em energia num M1/M2/M3/M4, mas exige exportar o modelo
   para ``.mlpackage``.
2. **MPS** - backend Metal do PyTorch. Roda na GPU integrada, sem exportacao;
   usar quando se quer iterar rapido ou treinar.
3. **CUDA** - se o projeto for movido para uma maquina NVIDIA.
4. **CPU** - sempre disponivel; no ONNX Runtime usa multi-thread e, em Intel,
   pode ser acelerado pelo OpenVINO Execution Provider.

Nada aqui faz chamadas de rede nem usa API de inferencia remota: todo o
processamento e local.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Descreve o dispositivo escolhido e como configurar cada runtime."""

    kind: str                # "coreml" | "mps" | "cuda" | "cpu"
    torch_device: str        # string aceita por torch.device / ultralytics
    onnx_providers: tuple[str, ...]
    description: str
    is_apple_silicon: bool = False

    def __str__(self) -> str:  # pragma: no cover - apenas apresentacao
        return f"{self.kind} ({self.description})"


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except ImportError:
        return []


def resolve_device(preference: str = "auto") -> DeviceInfo:
    """Resolve o dispositivo a partir da preferencia do usuario.

    ``preference`` aceita ``auto``, ``coreml``, ``mps``, ``cuda`` ou ``cpu``.
    Uma preferencia indisponivel emite aviso e cai para o melhor disponivel,
    em vez de falhar - importante para que a POC rode em qualquer maquina.
    """
    pref = (preference or "auto").lower().strip()
    available = _onnx_providers()
    apple = is_apple_silicon()

    def cpu_device() -> DeviceInfo:
        providers: tuple[str, ...] = ("CPUExecutionProvider",)
        desc = "CPU"
        if "OpenVINOExecutionProvider" in available:
            providers = ("OpenVINOExecutionProvider", "CPUExecutionProvider")
            desc = "CPU + OpenVINO"
        return DeviceInfo("cpu", "cpu", providers, desc, apple)

    def coreml_device() -> DeviceInfo | None:
        if not apple or "CoreMLExecutionProvider" not in available:
            return None
        return DeviceInfo(
            "coreml", "mps",
            ("CoreMLExecutionProvider", "CPUExecutionProvider"),
            "Apple Neural Engine / GPU via CoreML", True,
        )

    def mps_device() -> DeviceInfo | None:
        if not apple or not _torch_available():
            return None
        import torch

        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            return None
        return DeviceInfo("mps", "mps", ("CPUExecutionProvider",), "Apple GPU via Metal (MPS)", True)

    def cuda_device() -> DeviceInfo | None:
        if "CUDAExecutionProvider" in available:
            return DeviceInfo(
                "cuda", "cuda", ("CUDAExecutionProvider", "CPUExecutionProvider"), "GPU NVIDIA"
            )
        if _torch_available():
            import torch

            if torch.cuda.is_available():
                return DeviceInfo("cuda", "cuda", ("CPUExecutionProvider",), "GPU NVIDIA (torch)")
        return None

    builders = {"coreml": coreml_device, "mps": mps_device, "cuda": cuda_device}

    if pref == "cpu":
        return cpu_device()
    if pref in builders:
        device = builders[pref]()
        if device is not None:
            return device
        log.warning("Backend '%s' indisponivel nesta maquina; usando fallback automatico.", pref)
        pref = "auto"

    if pref == "auto":
        for builder in (coreml_device, mps_device, cuda_device):
            device = builder()
            if device is not None:
                return device
        return cpu_device()

    log.warning("Preferencia de dispositivo desconhecida: %r; usando CPU.", preference)
    return cpu_device()


def is_available(kind: str) -> bool:
    """Diz se um backend esta disponivel, SEM emitir aviso.

    ``resolve_device`` avisa quando cai para o fallback - correto quando o
    usuario pediu aquele backend, ruidoso quando so estamos montando um menu.
    """
    alvo = (kind or "").lower().strip()
    if alvo == "cpu":
        return True
    if alvo == "auto":
        return True
    providers = _onnx_providers()
    if alvo == "coreml":
        return is_apple_silicon() and "CoreMLExecutionProvider" in providers
    if alvo == "cuda":
        if "CUDAExecutionProvider" in providers:
            return True
        if _torch_available():
            import torch

            return bool(torch.cuda.is_available())
        return False
    if alvo == "mps":
        if not (is_apple_silicon() and _torch_available()):
            return False
        import torch

        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    return False


def describe_environment(include_freeze: bool = True) -> dict[str, object]:
    """Coleta metadados do ambiente para gravar junto de cada sessao.

    Sem isso um benchmark nao e reproduzivel: FPS so significa algo se soubermos
    em que maquina e com que versoes ele foi medido. Inclui a proveniencia
    completa (commit, pacotes, hardware, energia) - ver ``lprts.provenance``.
    """
    from ..provenance import collect, hardware

    hw = hardware()
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        # platform.processor() devolve so "arm" no Apple Silicon; o modelo do chip
        # vem do sysctl.
        "processor": hw.get("cpu_model") or platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "apple_silicon": is_apple_silicon(),
        "onnx_providers": _onnx_providers(),
    }
    info.update(collect(include_freeze=include_freeze))
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_mps"] = bool(torch.backends.mps.is_available())
        info["torch_cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        from ..runtime import import_ultralytics

        info["ultralytics"] = import_ultralytics().__version__
    except ImportError:
        pass
    return info
