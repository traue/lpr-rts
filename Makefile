# LPR-RTS - atalhos de desenvolvimento e uso.
# Rode `make` (ou `make help`) para a lista de alvos.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
VIDEO  ?= data/videos/sintetico.mp4
DB     ?= runs/lprts.db

.DEFAULT_GOAL := help
.PHONY: help venv install install-minimal install-dev demo run run-realtime gui \
        run-parallel batch tune synthetic eval bench export test test-cov lint format \
        typecheck models clean clean-runs info doctor diagnose start lock

help:  ## Mostra esta ajuda
	@echo "LPR-RTS - alvos disponiveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variaveis: VIDEO=$(VIDEO)  DB=$(DB)"

venv:  ## Cria o ambiente virtual
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip wheel setuptools

install: venv  ## Instalacao completa (interface, deep learning, OCR, analise)
	$(BIN)/pip install -e ".[ui,dl,ocr,analysis,dev]"

install-minimal: venv  ## Instalacao minima (roda sem downloads de modelo)
	$(BIN)/pip install -e ".[dev]"

install-dev: venv  ## Nucleo + ferramentas de desenvolvimento
	$(BIN)/pip install -e ".[ui,analysis,dev]"

doctor:  ## Verifica se o ambiente esta pronto para executar
	$(BIN)/lprts doctor

diagnose:  ## Mostra em que estagio o pipeline para, no video de teste
	$(BIN)/lprts diagnose --source $(VIDEO)

info:  ## Ambiente, dispositivo de inferencia e motores disponiveis
	$(BIN)/lprts info

models:  ## Baixa os pesos YOLO (e exporta CoreML no Apple Silicon)
	$(BIN)/python scripts/download_models.py --coreml

synthetic:  ## Gera video sintetico + ground truth
	$(BIN)/python scripts/make_synthetic_dataset.py --out $(VIDEO) --frames 400 --vehicles 4

demo:  ## Demonstracao completa: gera, executa e avalia (sem downloads)
	$(BIN)/lprts demo

start:  ## Atalho: prepara o ambiente e abre a interface (equivale a ./start.sh)
	./start.sh

run:  ## Executa com interface grafica sobre VIDEO
	$(BIN)/lprts run --source $(VIDEO)

run-realtime:  ## Executa em modo tempo real (webcam por padrao)
	$(BIN)/lprts run --source 0 --realtime --config configs/tempo_real.yaml

gui: run  ## Alias de `run`

eval:  ## Avalia contra o ground truth do video sintetico
	$(BIN)/lprts eval --source $(VIDEO) --gt $(basename $(VIDEO))_gt.csv \
	  --config configs/avaliacao.yaml --out runs/report.json

bench:  ## Mede o desempenho por estagio
	$(BIN)/python scripts/benchmark.py --source $(VIDEO) --frames 300

run-parallel:  ## Executa com deteccao paralela (headless)
	$(BIN)/lprts run --source $(VIDEO) --headless --parallel -1

batch:  ## Processa em lote todos os videos de data/videos
	$(BIN)/lprts batch "data/videos/*.mp4" --backend process --config configs/lote.yaml

tune:  ## Otimiza os parametros de fusao a partir do banco (varredura exaustiva)
	$(BIN)/lprts tune --db $(DB) --gt $(basename $(VIDEO))_gt.csv --grid

export:  ## Exporta o banco para CSV
	$(BIN)/lprts export --db $(DB) --out runs/export --format csv

test:  ## Roda a suite de testes
	$(BIN)/python -m pytest tests/ -q

test-cov:  ## Testes com relatorio de cobertura
	$(BIN)/python -m pytest tests/ --cov=lprts --cov-report=term-missing --cov-report=html

lint:  ## Verifica o estilo com ruff
	$(BIN)/ruff check src tests scripts

format:  ## Formata e corrige o que for automatico
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

typecheck:  ## Verificacao de tipos com mypy
	$(BIN)/mypy src/lprts

clean:  ## Remove artefatos de build e cache
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-runs:  ## Apaga bancos, recortes e relatorios gerados
	rm -rf runs/*.db runs/*.db-wal runs/*.db-shm runs/crops runs/export runs/*.json

lock:  ## Trava as versoes exatas do ambiente atual em requirements.lock (reprodutibilidade)
	$(BIN)/python -m pip freeze --exclude-editable | sort > requirements.lock
	@echo "Versoes travadas em requirements.lock - versione junto dos resultados."
	@echo "Recriar o ambiente: pip install -r requirements.lock && pip install -e . --no-deps"
