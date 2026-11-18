#!/usr/bin/env bash
# LPR-RTS - inicio rapido.
#
# Faz tudo o que for preciso e abre a interface grafica:
#   1. cria o ambiente virtual, se nao existir;
#   2. instala o que faltar;
#   3. gera um video de demonstracao, se nao houver nenhum;
#   4. abre a janela.
#
# Uso:
#   ./start.sh                     interface grafica com o video de demonstracao
#   ./start.sh video.mp4           interface grafica com o seu video
#   ./start.sh 0                   interface grafica com a webcam
#   ./start.sh --doctor            so verifica o ambiente
#   ./start.sh --demo              demonstracao completa no terminal
#   ./start.sh --setup             so instala, nao abre nada

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
AZUL=$'\033[36m'; VERDE=$'\033[32m'; AMARELO=$'\033[33m'; VERMELHO=$'\033[31m'; FIM=$'\033[0m'

info()  { echo "${AZUL}==>${FIM} $*"; }
ok()    { echo "${VERDE}  ok${FIM} $*"; }
aviso() { echo "${AMARELO}  !${FIM} $*"; }
erro()  { echo "${VERMELHO}ERRO:${FIM} $*" >&2; }

# --- 0. modos que nao exigem o ambiente ----------------------------------
# --help precisa responder na hora, sem provocar uma instalacao de minutos.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

# --- 1. Python -----------------------------------------------------------
achar_python() {
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)' 2>/dev/null; then
        echo "$c"; return 0
      fi
    fi
  done
  return 1
}

if [[ ! -d "$VENV" ]]; then
  info "Primeira execucao: preparando o ambiente (leva alguns minutos)"
  PY="$(achar_python)" || {
    erro "Python 3.10-3.12 nao encontrado."
    echo "  Instale com:  brew install python@3.11"
    exit 1
  }
  ok "usando $PY ($("$PY" --version))"
  "$PY" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --quiet --upgrade pip wheel setuptools
  info "Instalando o LPR-RTS e suas dependencias"
  pip install --quiet -e ".[ui,dl,ocr,analysis]" || {
    aviso "instalacao completa falhou; tentando a versao minima"
    pip install --quiet -e ".[ui]"
  }
  ok "ambiente pronto"
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  # A instalacao editavel some se o pacote foi movido ou o venv reaproveitado.
  python -c "import lprts" 2>/dev/null || {
    info "Reinstalando o pacote no ambiente existente"
    pip install --quiet -e ".[ui]"
  }
fi

# --- 2. modos que nao abrem a interface ----------------------------------
case "${1:-}" in
  --setup)  ok "Ambiente pronto. Rode ./start.sh para abrir a interface."; exit 0 ;;
  --doctor) exec lprts doctor ;;
  --demo)   exec lprts demo ;;
esac

# --- 3. bancos de versoes anteriores -------------------------------------
# A versao atual recusa bancos de esquema antigo (seus dados tem erros
# conhecidos). Aqui eles sao arquivados em runs/legado/ - movidos, nunca apagados.
python - <<'PY'
from lprts.analytics.store import archive_legacy_databases

for origem, destino in archive_legacy_databases("runs"):
    print(f"\033[33m  !\033[0m banco de versao anterior arquivado: {origem} -> {destino}")
PY

# --- 4. fonte de video ---------------------------------------------------
FONTE="${1:-}"
if [[ -z "$FONTE" ]]; then
  FONTE="data/videos/sintetico.mp4"
  if [[ ! -f "$FONTE" ]]; then
    info "Nenhum video encontrado; gerando um de demonstracao"
    python scripts/make_synthetic_dataset.py --out "$FONTE" --frames 400 --vehicles 4 >/dev/null
    ok "gerado: $FONTE"
  fi
  # Os veiculos deste video sao formas sinteticas, que um detector treinado em
  # fotos reais nao reconhece: ja entramos no modo que rastreia placas direto.
  # Vale so para esta fonte: ao abrir um video real ou uma camera na interface,
  # a caixa "Veiculos (YOLO)" e marcada de novo.
  EXTRA=(--no-vehicle-detector --plate-detector classic)
  aviso "usando o video de demonstracao (formas sinteticas: modo so-placa)"
else
  EXTRA=()
fi

# --- 5. abrir a interface ------------------------------------------------
info "Abrindo a interface grafica"
echo "   fonte: $FONTE"
echo "   (para outra fonte: ./start.sh SEU_VIDEO.mp4  ou  ./start.sh 0 para a webcam)"
echo ""
# "${EXTRA[@]+...}" e a forma portatil de expandir um array possivelmente vazio
# sob `set -u`: o bash 3.2 que vem no macOS aborta com "EXTRA[@]: unbound
# variable" na forma simples - e era isso que fazia `./start.sh video.mp4` falhar.
exec lprts run --source "$FONTE" ${EXTRA[@]+"${EXTRA[@]}"}
