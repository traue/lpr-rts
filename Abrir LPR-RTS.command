#!/usr/bin/env bash
# Atalho para abrir o LPR-RTS com dois cliques no Finder do macOS.
# O .command faz o Terminal executar o script na pasta do projeto.
cd "$(dirname "${BASH_SOURCE[0]}")"
./start.sh
echo ""
echo "Pressione qualquer tecla para fechar esta janela..."
read -n 1 -s
