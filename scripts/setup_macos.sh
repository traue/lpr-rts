#!/usr/bin/env bash
# Instalacao do LPR-RTS no macOS (Intel ou Apple Silicon).
# Cria um ambiente virtual, instala as dependencias e verifica o resultado.
#
#   bash scripts/setup_macos.sh              # instalacao completa
#   bash scripts/setup_macos.sh --minimal    # so o nucleo (roda sem downloads)

set -euo pipefail

MINIMAL=0
[[ "${1:-}" == "--minimal" ]] && MINIMAL=1

echo "=================================================================="
echo "LPR-RTS - instalacao no macOS"
echo "=================================================================="

ARCH="$(uname -m)"
echo "  arquitetura: ${ARCH}"
if [[ "${ARCH}" == "arm64" ]]; then
  echo "  Apple Silicon detectado: CoreML/Neural Engine e MPS disponiveis."
else
  echo "  Intel detectado: a inferencia usara CPU (considere --minimal)."
fi

# --- Python 3.11 ---------------------------------------------------------
PY=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    VER="$(${candidate} -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${VER}" =~ ^3\.(10|11|12)$ ]]; then PY="${candidate}"; break; fi
  fi
done
if [[ -z "${PY}" ]]; then
  echo "ERRO: Python 3.10-3.12 nao encontrado."
  echo "  Instale com:  brew install python@3.11"
  exit 1
fi
echo "  python: ${PY} ($(${PY} --version))"

# --- ambiente virtual ----------------------------------------------------
if [[ ! -d .venv ]]; then
  echo ""
  echo "[1/4] Criando ambiente virtual em .venv"
  "${PY}" -m venv .venv
else
  echo ""
  echo "[1/4] Ambiente virtual .venv ja existe"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel setuptools

# --- dependencias --------------------------------------------------------
echo ""
if [[ "${MINIMAL}" == "1" ]]; then
  echo "[2/4] Instalando o nucleo (numpy, opencv, typer)"
  pip install --quiet -e .
else
  echo "[2/4] Instalando o pacote completo (pode levar alguns minutos)"
  pip install --quiet -e ".[ui,dl,ocr,analysis,dev]"
fi

# --- tesseract opcional --------------------------------------------------
echo ""
echo "[3/4] Verificando o Tesseract (opcional, motor de OCR alternativo)"
if command -v tesseract >/dev/null 2>&1; then
  echo "  presente: $(tesseract --version 2>&1 | head -1)"
else
  echo "  ausente (opcional). Para instalar:  brew install tesseract"
fi

# --- verificacao ---------------------------------------------------------
echo ""
echo "[4/4] Verificando a instalacao"
python - <<'PYCHECK'
import importlib, platform
print(f"  plataforma: {platform.platform()}")
for mod, rotulo in [("numpy","numpy"),("cv2","opencv"),("typer","typer"),
                    ("PySide6","interface Qt"),("torch","pytorch"),
                    ("ultralytics","ultralytics"),("onnxruntime","onnxruntime")]:
    try:
        m = importlib.import_module(mod)
        print(f"  [ok] {rotulo:14s} {getattr(m,'__version__','')}")
    except ImportError:
        print(f"  [--] {rotulo:14s} (nao instalado)")
try:
    import torch
    if torch.backends.mps.is_available():
        print("  [ok] Apple GPU (MPS) disponivel para inferencia e treino")
except Exception:
    pass
PYCHECK

echo ""
echo "=================================================================="
echo "Instalacao concluida."
echo ""
echo "Ative o ambiente e experimente:"
echo "  source .venv/bin/activate"
echo "  lprts doctor    # verifica se esta tudo pronto"
echo "  lprts demo      # video sintetico + pipeline completo (sem downloads)"
echo "  lprts run --source data/videos/sintetico.mp4   # interface grafica"
echo "=================================================================="
