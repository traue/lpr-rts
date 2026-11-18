"""Proveniencia de cada sessao: codigo, pacotes, hardware e energia.

Um numero de desempenho ou de acuracia so e interpretavel junto com o que o
produziu. Este modulo coleta, uma vez por processo:

* **codigo** - commit do git e se havia alteracoes nao commitadas (um resultado
  gerado com a arvore suja nao e reproduzivel a partir do commit);
* **pacotes** - versao de todos os pacotes instalados (o equivalente a um
  ``pip freeze``), mais os relevantes destacados;
* **hardware** - modelo do processador, nucleos, memoria;
* **energia** - no macOS, se o equipamento estava na bateria ou na tomada e se o
  modo de baixo consumo estava ligado: ambos mudam o desempenho medido.

Tudo e best-effort: a falha em coletar um item vira ``None``, nunca um erro.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# Pacotes cujas versoes mudam resultados ou desempenho - destacados no topo.
PACOTES_RELEVANTES = (
    "lprts", "numpy", "opencv-python", "opencv-python-headless", "onnxruntime",
    "torch", "torchvision", "ultralytics", "fast-plate-ocr", "easyocr", "paddleocr",
    "paddlepaddle", "pytesseract", "PySide6", "scipy", "pandas", "coremltools",
)


def _rodar(cmd: list[str], cwd: Path | None = None, timeout: float = 3.0) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _raiz_do_codigo() -> Path:
    # src/lprts/provenance.py -> raiz do repositorio (instalacao editavel).
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def git_info() -> dict[str, Any] | None:
    """Commit, ramo e estado da arvore de trabalho do codigo em execucao."""
    raiz = _raiz_do_codigo()
    commit = _rodar(["git", "rev-parse", "HEAD"], cwd=raiz)
    if not commit:
        return None
    status = _rodar(["git", "status", "--porcelain", "--untracked-files=no"], cwd=raiz)
    return {
        "commit": commit,
        "branch": _rodar(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=raiz),
        "dirty": bool(status),
        "dirty_files": status.splitlines()[:50] if status else [],
    }


def code_version() -> str:
    """Rotulo curto do codigo: '<commit12>' ou '<commit12>+dirty', ou a versao do pacote."""
    info = git_info()
    if info:
        return info["commit"][:12] + ("+dirty" if info["dirty"] else "")
    return f"lprts-{package_version('lprts')}"


@lru_cache(maxsize=64)
def package_version(nome: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(nome)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def installed_distributions() -> list[str]:
    """Todos os pacotes instalados, como ``nome==versao`` (equivalente a pip freeze)."""
    try:
        from importlib.metadata import distributions
    except ImportError:  # pragma: no cover
        return []
    vistos: dict[str, str] = {}
    for dist in distributions():
        nome = dist.metadata["Name"] if dist.metadata else None
        if nome:
            vistos[nome.lower()] = f"{nome}=={dist.version}"
    return [vistos[k] for k in sorted(vistos)]


def opencv_conflict() -> bool:
    """True se ``opencv-python`` e ``opencv-python-headless`` estao instalados juntos.

    Os dois instalam o mesmo modulo ``cv2``; o que foi instalado por ultimo
    sobrescreve o outro, e a versao headless remove a interface do OpenCV.
    """
    return bool(package_version("opencv-python") and package_version("opencv-python-headless"))


@lru_cache(maxsize=1)
def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": None,
        "physical_cpus": None,
        "memory_bytes": None,
    }
    sistema = platform.system()
    if sistema == "Darwin":
        info["cpu_model"] = _rodar(["sysctl", "-n", "machdep.cpu.brand_string"])
        fisicos = _rodar(["sysctl", "-n", "hw.physicalcpu"])
        memoria = _rodar(["sysctl", "-n", "hw.memsize"])
        info["physical_cpus"] = int(fisicos) if fisicos and fisicos.isdigit() else None
        info["memory_bytes"] = int(memoria) if memoria and memoria.isdigit() else None
        # Nucleos de desempenho x eficiencia mudam o desempenho de threads.
        for nivel, rotulo in (("0", "performance_cpus"), ("1", "efficiency_cpus")):
            valor = _rodar(["sysctl", "-n", f"hw.perflevel{nivel}.physicalcpu"])
            if valor and valor.isdigit():
                info[rotulo] = int(valor)
    elif sistema == "Linux":
        try:
            texto = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"^model name\s*:\s*(.+)$", texto, re.MULTILINE)
            info["cpu_model"] = m.group(1).strip() if m else None
            mem = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"^MemTotal:\s*(\d+)\s*kB", mem, re.MULTILINE)
            info["memory_bytes"] = int(m.group(1)) * 1024 if m else None
        except OSError:
            pass
    return info


def power_state() -> dict[str, Any]:
    """Fonte de energia e modo de baixo consumo (macOS). Nao e cacheado: muda."""
    if platform.system() != "Darwin":
        return {}
    estado: dict[str, Any] = {}
    batt = _rodar(["pmset", "-g", "batt"])
    if batt:
        if "AC Power" in batt:
            estado["power_source"] = "ac"
        elif "Battery Power" in batt:
            estado["power_source"] = "battery"
        m = re.search(r"(\d+)%", batt)
        if m:
            estado["battery_percent"] = int(m.group(1))
    geral = _rodar(["pmset", "-g"])
    if geral:
        m = re.search(r"lowpowermode\s+(\d)", geral)
        if m:
            estado["low_power_mode"] = m.group(1) == "1"
    return estado


def collect(include_freeze: bool = True) -> dict[str, Any]:
    """Proveniencia completa para gravar em ``sessions.environment_json``."""
    dados: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": os.path.realpath(sys.executable),
        "code": git_info(),
        "code_version": code_version(),
        "hardware": hardware(),
        "power": power_state(),
        "packages": {
            nome: v for nome in PACOTES_RELEVANTES if (v := package_version(nome)) is not None
        },
        "opencv_conflict": opencv_conflict(),
    }
    if include_freeze:
        dados["installed"] = installed_distributions()
    return dados
