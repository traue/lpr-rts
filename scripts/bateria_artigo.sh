#!/bin/bash
# Bateria de experimentos do artigo para UM video (docs/artigo/CONTINUAR.md, secao 6).
#
#   scripts/bateria_artigo.sh VIDEO GT NOME [ESCOPO]
#   scripts/bateria_artigo.sh data/videos/cortes/V1.MOV data/videos/gabaritos/V1_gt.csv V1 50
#
# Grava tudo em runs/artigo/NOME/ (fora do git). ESCOPO e a largura minima da placa, em px
# da imagem original, passada ao --scope-min-width (0 = avaliar tudo; padrao 50); pode ser
# refeita depois com `lprts eval-db`, sem rodar de novo.
#
# Sao 40 rodadas: 5 sem prazo e 35 em tempo real (7 condicoes x 5 repeticoes); cada rodada
# em tempo real dura o video inteiro. Uma rodada cujo relatorio ja existe e pulada: se a bateria for
# interrompida, basta chamar de novo. REFAZER="base_rt o4_ffmpeg_rt" (ou REFAZER=todas) refaz
# essas rodadas: os arquivos antigos vao para runs/artigo/NOME/descartadas/<data>/, sem apagar.
#
# Antes: Mac na tomada, outros programas fechados, arvore do git limpa (o commit vai
# gravado em cada sessao), `make lock`, Power Nap desligado (Ajustes > Bateria > Opcoes). O
# script mantem a tela acesa e o Mac acordado do inicio ao fim (caffeinate -dims): em
# 2026-11-15 o Mac dormiu entre rodadas e, em DarkWake, tarefas de manutencao do macOS
# contaminaram medidas de tempo. FORCAR=1 roda mesmo sem tomada ou com o git sujo (so para
# ensaio: o resultado nao serve para o artigo).

set -euo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
V=$1; G=$2; NOME=$3; ESCOPO=${4:-50}
REPETICOES=${REPETICOES:-5}
cd "$(dirname "$0")/.."
LPRTS=${LPRTS:-.venv/bin/lprts}
PY=${PY:-.venv/bin/python}
R=runs/artigo/$NOME

[ -f "$V" ] || { echo "video nao encontrado: $V" >&2; exit 1; }
[ -f "$G" ] || { echo "gabarito nao encontrado: $G" >&2; exit 1; }
for y in sem_recortes o1_ultralytics o2_todas_placas o3_recorte_reduzido o4_ffmpeg res_960 res_1920; do
  [ -f "configs/artigo/$y.yaml" ] || { echo "falta configs/artigo/$y.yaml" >&2; exit 1; }
done

problema=""
[ -z "$(git status --porcelain)" ] || problema="a arvore do git tem mudancas sem commit"
if command -v pmset >/dev/null && ! pmset -g batt | grep -q "AC Power"; then
  problema="${problema:+$problema; }o computador nao esta na tomada"
fi
if [ -n "$problema" ]; then
  if [ "${FORCAR:-0}" = "1" ]; then
    echo "AVISO: $problema. FORCAR=1: seguindo, mas este resultado nao serve para o artigo." >&2
  else
    echo "Nao rodei: $problema. (FORCAR=1 para ensaiar assim mesmo.)" >&2
    exit 1
  fi
fi

if command -v pmset >/dev/null && pmset -g | grep -qE "^ *powernap +1"; then
  echo "AVISO: Power Nap ligado. Desligue-o (Ajustes > Bateria > Opcoes) para que tarefas de" >&2
  echo "manutencao do macOS nao rodem durante a bateria." >&2
fi

mkdir -p "$R"
B=(-c configs/tempo_real.yaml -c configs/artigo/sem_recortes.yaml)
CAFE=()
command -v caffeinate >/dev/null && CAFE=(caffeinate -i)
# Tela acesa e Mac acordado ate o fim do script, inclusive entre as rodadas.
command -v caffeinate >/dev/null && caffeinate -dims -w $$ &

# Rodadas a refazer: os arquivos antigos saem do caminho (nao sao apagados).
if [ -n "${REFAZER:-}" ]; then
  D="$R/descartadas/$(date '+%Y%m%dT%H%M%S')"
  mkdir -p "$D"
  if [ "$REFAZER" = "todas" ]; then
    for f in "$R"/*; do
      case "$(basename "$f")" in bateria.log|descartadas) ;; *) mv "$f" "$D/" ;; esac
    done
  else
    for n in $REFAZER; do
      for f in "$R/$n".* "$R/$n"_*; do
        [ -e "$f" ] && mv "$f" "$D/"
      done
      # o relatorio da linha de base sai do banco dela
      if [ "$n" = "base_rt" ]; then
        for f in "$R/relatorio_base_rt" "$R/relatorio_base_rt.log"; do
          [ -e "$f" ] && mv "$f" "$D/"
        done
      fi
    done
  fi
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') refazer: ${REFAZER} (antigos em ${D#"$R"/})" >> "$R/bateria.log"
fi

{
  echo "inicio: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "video: $V | gabarito: $G | escopo: $ESCOPO px | repeticoes: $REPETICOES"
  echo "commit: $(git rev-parse HEAD)$([ -z "$(git status --porcelain)" ] || echo ' (com mudancas sem commit)')"
  echo "energia: $(pmset -g 2>/dev/null | awk '/powernap/ {print "powernap=" $2}') $(pmset -g batt 2>/dev/null | head -1 | grep -o "'.*'")"
} >> "$R/bateria.log"

# rodada NOME_DA_SAIDA MODO [-c variante.yaml]
rodada() {
  local saida=$1 modo=$2; shift 2
  if [ -f "$R/$saida.json" ]; then
    echo "[pulada, ja existe] $saida"
    return 0
  fi
  local extra=()
  [ "$modo" = "--realtime" ] && extra=(--repeat "$REPETICOES")
  echo "[$(date '+%H:%M:%S')] $saida ($modo)"
  # O relatorio so recebe o nome final se a rodada terminar: uma interrompida e refeita.
  rm -f "$R/$saida.db" "$R/$saida.db-wal" "$R/$saida.db-shm"
  ${CAFE[@]+"${CAFE[@]}"} "$LPRTS" eval -s "$V" --gt "$G" "${B[@]}" "$@" "$modo" ${extra[@]+"${extra[@]}"} \
    --scope-min-width "$ESCOPO" --db "$R/$saida.db" --out "$R/$saida.parcial.json" >> "$R/$saida.log" 2>&1
  for f in "$R/$saida".parcial_*; do
    [ -e "$f" ] || continue
    mv "$f" "${f/.parcial/}"
  done
  mv "$R/$saida.parcial.json" "$R/$saida.json"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') ok $saida" >> "$R/bateria.log"
}

# Linha de base, tudo ligado: E3 e referencia do E2.
rodada base_off --offline
rodada base_rt  --realtime

# E2: uma otimizacao desligada por vez. O1 e O4 so mudam o custo: tempo real.
for o in o1_ultralytics o4_ffmpeg; do
  rodada "${o}_rt" --realtime -c "configs/artigo/$o.yaml"
done
# O2 e O3 mudam a entrada dos modelos: sem prazo e em tempo real.
for o in o2_todas_placas o3_recorte_reduzido; do
  rodada "${o}_off" --offline  -c "configs/artigo/$o.yaml"
  rodada "${o}_rt"  --realtime -c "configs/artigo/$o.yaml"
done

# E1: resolucao e custo isolado de cada componente. Sem prazo, a resolucao so muda o que o
# OCR alcanca (o teto previsto pela anotacao); em tempo real, tambem o custo e o descarte.
for w in 960 1920; do
  rodada "res${w}_off" --offline  -c "configs/artigo/res_$w.yaml"
  rodada "res${w}_rt"  --realtime -c "configs/artigo/res_$w.yaml"
done
if [ ! -f "$R/bench_componentes.json" ]; then
  echo "[$(date '+%H:%M:%S')] custo por componente"
  ${CAFE[@]+"${CAFE[@]}"} "$PY" scripts/benchmark_componentes.py --source "$V" --width 1280 \
    --out "$R/bench_componentes.json" >> "$R/bench_componentes.log" 2>&1
fi
if [ ! -d "$R/relatorio_base_rt" ]; then
  "$LPRTS" report --db "$R/base_rt.db" --out "$R/relatorio_base_rt" >> "$R/relatorio_base_rt.log" 2>&1 || \
    echo "AVISO: o relatorio de base_rt falhou; veja $R/relatorio_base_rt.log" >&2
fi

echo "fim: $(date '+%Y-%m-%dT%H:%M:%S%z')" >> "$R/bateria.log"
echo "Bateria de $NOME concluida: $R/"
