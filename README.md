# LPR-RTS — Reconhecimento de Placas Veiculares em Tempo Real

Prova de conceito (POC) para pesquisa de doutorado: um sistema completo de **LPR/ALPR**
(*License Plate Recognition*) que detecta veículos, localiza placas, faz OCR, funde
leituras ao longo do tempo e grava dados estruturados para análise científica.

**Roda inteiramente na sua máquina** — CPU, GPU ou Neural Engine. Nenhum serviço de
nuvem, nenhuma API externa, nenhum container. Testado em macOS (Apple Silicon) e Linux.

```bash
git clone https://github.com/traue/lpr-rts.git
cd lpr-rts
./start.sh
```

É só isso. O `start.sh` cria o ambiente, instala o que falta, gera um vídeo de
demonstração se não houver nenhum e abre a interface. No macOS você também pode
dar dois cliques em **Abrir LPR-RTS.command** no Finder.

```bash
./start.sh video.mp4     # com o seu vídeo
./start.sh 0             # com a webcam
./start.sh --doctor      # só verifica o ambiente
./start.sh --demo        # demonstração completa no terminal
```

O comando acima funciona **sem baixar nenhum modelo**: o sistema traz um detector de
placas clássico e um OCR por gabaritos que servem de baseline e garantem a primeira
execução.

> **Versão 0.5.** A 0.2 corrigiu erros que invalidavam os dados gravados pela 0.1
> (base de tempo, trilhas truncadas, descartes não gravados, limiar do rastreador,
> avaliação otimista); bancos da 0.1 são recusados. A 0.3 mede a latência com modelos
> neurais e usa o CoreML de fato no Apple Silicon, acrescenta velocidade e borrão de
> movimento ao gerador sintético e torna a política de publicação configurável
> (bancos da 0.2 são migrados sem perda). A 0.4 corta pela metade o custo da detecção
> (YOLO direto no CoreML), lê as placas no quadro original da câmera, começa cada
> processamento com o banco limpo e gera tabelas e figuras para o artigo
> (`lprts report`). A 0.5 (auditoria de publicação; esquema 4, bancos 2 e 3 migrados)
> faz toda falha de gravação ou de captura interromper a sessão em vez de virar dado
> faltante, associa as trilhas como o ByteTrack de referência, fixa as threads do OCR
> (leituras independentes da máquina) e usa os núcleos: paralelo por padrão offline,
> com resultado idêntico ao serial, 1,6–1,7× no perfil padrão e ~3× no de avaliação.
> Detalhes em [CHANGELOG.md](CHANGELOG.md).

---

## Reproduzir o artigo

Este repositório traz o código, as configurações e os resultados do artigo "LPR-RTS: leitura de
placas veiculares em tempo real brando em hardware de consumo". Os vídeos, os gabaritos e os
bancos das sessões ficam fora do git, porque são grandes e contêm placas reais. Eles estão no
Zenodo, em <https://doi.org/10.5281/zenodo.24351664>, com as condições de uso.

- Resultados: [`docs/artigo/resultados/`](docs/artigo/resultados/) tem o `analise.json`, com
  todos os números do artigo, e os dois diagnósticos.
- Plano de análise: [`docs/artigo/plano_analise_registrado.md`](docs/artigo/plano_analise_registrado.md)
  é o plano como foi registrado, antes da bateria.
- Refazer a análise a partir dos bancos publicados: `python scripts/artigo/analise.py`.
- Refazer a bateria a partir dos vídeos: `scripts/bateria_artigo.sh`. O cabeçalho do script
  explica os argumentos.

Os detalhes, inclusive o que mudou em relação ao plano, estão em
[`docs/artigo/README.md`](docs/artigo/README.md).

---

## Índice

1. [O que o sistema faz](#1-o-que-o-sistema-faz)
2. [Escolha da stack](#2-escolha-da-stack-e-por-quê)
3. [Arquitetura](#3-arquitetura)
4. [Instalação](#4-instalação)
5. [Uso](#5-uso)
6. [Dados extraídos](#6-dados-extraídos-para-análise)
7. [Avaliação científica](#7-avaliação-científica)
8. [Paralelismo e otimização](#8-paralelismo-e-otimização)
9. [Treinar seus próprios modelos](#9-treinar-seus-próprios-modelos)
10. [Desempenho e ajuste fino](#10-desempenho-e-ajuste-fino)
11. [Estrutura do projeto](#11-estrutura-do-projeto)
12. [Solução de problemas](#12-solução-de-problemas)
13. [Limitações](#13-limitações-conhecidas)

---

## 1. O que o sistema faz

| Capacidade | Descrição |
|---|---|
| **Detecção multi-classe** | Pessoas, bicicletas, carros, motos, ônibus, caminhões, animais, mobiliário urbano |
| **Taxonomia de objetos** | Cada classe sabe se porta placa e se é usuário vulnerável da via |
| **Detecção de placas** | YOLO treinável **ou** detector clássico morfológico (baseline sem download) |
| **OCR** | `fast-plate-ocr`, PaddleOCR, EasyOCR, Tesseract **ou** gabaritos clássicos |
| **Gramática brasileira** | Valida e corrige Mercosul (`LLLNLNN`) e antigo (`LLLNNNN`), no máximo 2 trocas |
| **Rastreamento** | ByteTrack próprio (Kalman + húngaro) — mantém a identidade do veículo |
| **Fusão multi-frame** | Votação ponderada por caractere entre todos os quadros de uma trilha |
| **Analítica de tráfego** | Linhas virtuais, contagem por sentido, velocidade via homografia |
| **Telemetria de tempo real** | Latência ponta a ponta, fila, serviço por estágio, descartes e prazos perdidos — por quadro |
| **Avaliação** | Por veículo, pareada por posição, com intervalos de confiança; detecção, OCR, rastreamento (MOTA/IDF1) e desempenho |
| **Proveniência** | Commit do código, versões de pacotes, hardware, energia e dispositivo efetivo em cada sessão |
| **Interfaces** | GUI desktop (Qt6), CLI completa e modo headless |
| **Persistência** | SQLite com esquema documentado coluna a coluna; exporta CSV/Parquet/JSON |
| **Paralelismo** | Detecção multi-thread, lote multiprocesso e MPI para cluster |
| **Otimização** | Busca multiobjetivo NSGA-II com fronteira de Pareto e validação em veículos reservados |

### Resultado do `lprts demo` (vídeo sintético, 400 quadros, Apple M5 Pro)

Baseline clássico (detector morfológico + OCR por gabaritos), modo só-placa:

```
POR VEICULO                                   EFEITO DA FUSAO (9 veiculos lidos)
  Veiculos anotados ....... 9                   Uma leitura ao acaso ...... 0.29
  Revocacao ............... 0.56 [0.22, 0.89]   Leitura mais confiante .... 0.33
  Precisao dos eventos .... 0.56 [0.22, 0.89]   Fusao de todas as leituras  0.33
  Latencia de decisao ..... 2.43 s (p95 2.87)   Publicado (decisao online)  0.56
DETECCAO DE PLACAS (IoU >= 0.50)
  Precisao / Revocacao .... 0.98 / 0.64       DESEMPENHO
  mAP@[.5:.95] ............ 0.44                Vazao ..................... 297 quadros/s
RASTREAMENTO                                    Latencia p50 / p95 / p99 .. 3.1 / 4.6 / 5.0 ms
  MOTA / IDF1 ............. 0.61 / 0.76         Descartes ................. 0
```

Medido na 0.5.0 (a 0.4.0 dá os mesmos números, salvo a leitura ao acaso: 0,33, porque o
OCR por gabaritos passava do limite de correções gramaticais; CHANGELOG 0.5.0).

Como ler: a fusão melhora sobre uma leitura isolada (0,29 → 0,33). Neste vídeo a
decisão publicada (na primeira fusão estável) ficou acima da fusão de todas as
leituras — dois veículos em nove: as primeiras leituras acertaram e as seguintes
acumularam um erro do OCR por gabaritos (6 lido como 5, 154 vezes no vídeo). Com 9
veículos a diferença não é significativa, mas mostra o limite da votação: ela não
corrige um erro sistemático do motor (é também por isso que a política `revise` não
compensa; §3). Os intervalos são largos porque são só 9 veículos. Nenhum desses fatos aparecia na avaliação da versão 0.1,
que pareava leituras por texto parecido e ignorava os veículos perdidos.

> Esses números vêm de vídeo **sintético** com o baseline clássico e servem para
> validar a implementação. Resultados para publicação exigem vídeo real e motores
> neurais (ver [§7](#7-avaliação-científica)).
>
> **O vídeo de referência é o versionado.** O `lprts demo` usa
> `data/videos/sintetico.mp4` quando ele existe (até a 0.3.1 ele o regenerava e
> sobrescrevia). Regenerado em outra máquina, o vídeo sai com o mesmo ground truth e
> diferença média de 0,03 por pixel (codificação do mp4) — e 2 dos 9 veículos mudam de
> resultado: revocação 0,33 em vez de 0,56. Com 9 veículos e o baseline clássico, o número
> é frágil; os intervalos de confiança, largos, já dizem isso.

---

## 2. Escolha da stack (e por quê)

### Python 3.11 + OpenCV + PySide6

| Alternativa | Por que **não** foi escolhida |
|---|---|
| **C++ puro** | Mais rápido no laço de orquestração, mas o gargalo é a inferência das redes, que já roda em código nativo (ONNX Runtime, PyTorch/Metal). O custo de iteração seria alto numa POC. |
| **Rust** | Ecossistema de visão computacional imaturo; sem ultralytics, PaddleOCR ou EasyOCR. |
| **Web (JS/WASM)** | A interface pedida é desktop, e WASM não acessa o Neural Engine. |

O laço quente (detecção, OCR, redimensionamento) executa em código nativo; o
Python orquestra. Para saber quanto cada estágio pesa **na sua configuração**,
rode `lprts bench`.

### Inferência local, sem motores externos

| Backend | Quando é usado | Aceleração |
|---|---|---|
| **CoreML** | Apple Silicon, com pesos exportados para `.mlpackage` | Neural Engine + GPU |
| **MPS** | Apple Silicon, pesos `.pt` via PyTorch | GPU Metal |
| **CUDA** | Máquina NVIDIA | GPU |
| **CPU** | Sempre disponível; OpenVINO se presente | SIMD multi-thread |

`--device auto` resolve a preferência; o que **de fato** roda depende do formato dos
pesos: um `.pt` roda em PyTorch (MPS/CUDA/CPU) e só um `.mlpackage` roda no CoreML.
Cada sessão grava o runtime e o dispositivo **efetivos** de cada componente (coluna
`sessions.device`), por exemplo `objetos=ultralytics/pytorch@mps; placas=opencv@cpu;
ocr=onnxruntime@cpu`. O OCR `fast-plate-ocr` roda em CPU por padrão (ver §12).

### Sem containers

O Docker no macOS roda numa VM Linux **sem acesso ao Neural Engine nem à GPU** —
containerizar custaria justamente a aceleração. A instalação é um ambiente virtual
Python.

---

## 3. Arquitetura

Referência técnica completa em [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md).

### Fluxo de dados

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌─────────┐
│ CAPTURA  │──▶│ DETECÇÃO  │──▶│ RASTREIO │──▶│    OCR    │──▶│  FUSÃO  │
│ arquivo  │   │ veículos  │   │ByteTrack │   │ assíncrono│   │ votação │
│ webcam   │   │  + placas │   │ + Kalman │   │  (pool)   │   │  /char  │
│ RTSP     │   └───────────┘   └──────────┘   └───────────┘   └────┬────┘
└────┬─────┘                                                       │
     │ DropOldestQueue                    ┌──────────────┬─────────┴────┐
     │ (descarta o mais antigo)           ▼              ▼              ▼
     └───────────────────────────▶   ANALYTICS      INTERFACE       SQLITE
                                    linhas/veloc.   Qt6 + HUD    dados p/ análise
```

### É um sistema de tempo real?

É um sistema de **tempo real brando** (*soft real-time*), de melhor esforço. Não há
garantia de prazo — o macOS não é um sistema operacional de tempo real e o custo de
um quadro varia com a cena —, mas o sistema **degrada de forma controlada e mede
quanto degradou**:

**1. Descarte a montante, nunca no meio.** Quando a inferência atrasa, quadros são
descartados **na captura** (`DropOldestQueue`), não no processamento. O total
descartado e os quadros pulados vão para o banco a cada quadro.

**2. OCR assíncrono.** O OCR roda num pool de threads; o laço principal envia só o
recorte da placa e colhe o resultado nos quadros seguintes. Cada leitura é gravada
com o quadro de onde o recorte saiu.

**3. Orçamento por quadro.** `max_ocr_per_frame` limita as leituras e
`max_rois_per_frame` a busca de placas por quadro.

**4. Medida, não promessa.** Cada quadro grava a latência **ponta a ponta** (do quadro
disponível ao resultado), a espera em fila, o tempo de serviço de cada estágio, os
descartes e se o prazo declarado foi perdido.

No modo `--offline` nada é descartado, o OCR é síncrono e o resultado é
determinístico — o modo para medir a capacidade do método. A acurácia **sob
restrição de tempo real** se mede à parte (`lprts eval --realtime --repeat N`, §7).

### Por que rastrear importa tanto

Um quadro isolado é uma amostra ruidosa. Como o rastreador garante que todas as
leituras de um `track_id` vêm **do mesmo veículo**, elas podem ser combinadas:

```
quadro 41: "ABC1D99"  ─┐
quadro 44: "XYZ1D23"  ─┼─▶ votação ponderada por posição ─▶ "ABC1D23" ✓
quadro 47: "ABC1D23"  ─┘     (peso = confiança × nitidez × sintaxe válida)
```

A votação é **por caractere**: aproveita quadros parcialmente corretos. Quanto ela
ajuda de fato é medido em cada avaliação (bloco "Efeito da fusão").

**Quando publicar** é uma decisão à parte (`fusion.publish_policy`): na primeira
fusão estável (`first_stable`, padrão), com revisão quando uma fusão estável
posterior muda o texto (`revise`), ou só quando a trilha termina (`on_exit`). Com
`fusion.exit_min_reads`, a trilha que sai de cena sem decisão — o veículo rápido, com
poucas leituras — ainda pode publicar. Cada evento grava o tipo da decisão e o número
da revisão; a avaliação mede a decisão vigente e a primeira emitida.

### Gramática brasileira como restrição

Placas brasileiras têm 7 caracteres, e sabemos para cada posição se deve ser letra
ou dígito:

```
Mercosul  L L L N L N N     ABC1D23
Antigo    L L L N N N N     ABC1234
```

Isso resolve a ambiguidade visual clássica do OCR (`O/0`, `I/1`, `S/5`, `B/8`) usando a
**posição** como contexto:

| Leitura bruta | Corrigida | Raciocínio |
|---|---|---|
| `4BC1D23` | `ABC1D23` | posição 0 exige letra → `4`→`A` |
| `ABCID23` | `ABC1D23` | posição 3 exige dígito → `I`→`1` |
| `ABC1O23` | `ABC1O23` | posição 4 exige letra → `O` está correto, não mexe |
| `1231231` | *rejeitada* | exigiria 3+ trocas — seria inventar uma placa |

O limite é de **2 trocas** (`ocr.max_grammar_corrections`). O texto antes da
gramática (`raw_text`) e o número de trocas (`n_corrections`) são gravados, para que
a contribuição da gramática possa ser medida.

### Taxonomia de objetos

| Classe | Categoria | Tem placa? | Vulnerável? |
|---|---|---|---|
| pessoa | pessoa | não | **sim** |
| bicicleta | veículo não motorizado | **não** | **sim** |
| carro | veículo motorizado | sim | não |
| motocicleta | veículo motorizado | sim | **sim** |
| ônibus, caminhão | veículo motorizado | sim | não |
| cachorro, cavalo | animal | não | não |
| semáforo, banco | mobiliário urbano | não | não |

1. **Bicicletas e pedestres nunca chegam ao OCR** (no Brasil bicicletas não são
   emplacadas).
2. **Motocicleta é o caso único** — emplacada e vulnerável ao mesmo tempo.

Todo objeto confirmado é gravado em `tracks` quando sai de cena, com ou sem placa —
é o que permite medir a composição do tráfego. Presets de classes: `veiculos`,
`trafego` ou `completo` (`--classes`).

### Quando o detector de objetos não serve ao vídeo

O fluxo padrão procura veículos e depois placas dentro deles. Mas o detector de
objetos é treinado em fotos reais: em vídeo sintético, animação ou imagem térmica
ele não reconhece nada.

Após `fallback_after_frames` quadros sem nenhum objeto **que possa portar placa**,
com placas visíveis, o sistema passa a rastrear as placas diretamente, encerra (e
grava) as trilhas em curso e avisa — no log, na barra de estado e nas notas da
sessão, com o quadro em que ocorreu. Em `configs/avaliacao.yaml` o fallback vem
**desligado**, para que nada mude no meio de uma medição.

### Componentes intercambiáveis

| Etapa | Opções |
|---|---|
| Detecção de placas | `yolo` (treinado) · `classic` (morfológico) · `auto` |
| OCR | `fastplate` · `paddle` · `easyocr` · `tesseract` · `template` · `auto` |
| Fusão | `char` (por caractere) · `string` (placa inteira) |
| Dispositivo | `coreml` · `mps` · `cuda` · `cpu` · `auto` |

Pedir um motor que não está instalado é **erro** (só `auto` escolhe o melhor
disponível, e registra qual). O **baseline clássico** é um ponto de comparação
pré-*deep learning* legítimo de reportar numa tese.

---

## 4. Instalação

### Requisitos

- macOS 12+ (Apple Silicon ou Intel) ou Linux
- Python 3.10 – 3.12
- ~2 GB de disco para o pacote completo

### Opção A — script automático (macOS)

```bash
bash scripts/setup_macos.sh              # completo
bash scripts/setup_macos.sh --minimal    # só o núcleo, roda sem downloads
```

### Opção B — manual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install -e .                         # núcleo (numpy, opencv, typer)
pip install -e ".[ui]"                   # + interface Qt6
pip install -e ".[dl]"                   # + torch, ultralytics, onnxruntime
pip install -e ".[ocr]"                  # + fast-plate-ocr, easyocr
pip install -e ".[analysis]"             # + pandas, pyarrow, duckdb
pip install -e ".[ui,dl,ocr,analysis,dev]"   # tudo
```

### Opção C — Makefile

```bash
make install            # completo
make install-minimal    # núcleo + testes
make lock               # trava as versões exatas em requirements.lock
```

### Verificar

```bash
lprts doctor    # dependências, GPU, modelos, permissões, bancos antigos
lprts info      # processador, energia, commit do código, dispositivo
make test       # suíte de testes (~600)
make lint typecheck   # ruff e mypy
```

O CI (`.github/workflows/checks.yml`) repete testes, ruff, mypy, a construção do
wheel e a demo instalada fora do checkout, em Linux e macOS com Python 3.10 e 3.12.

O `doctor` diz, item a item, o que está **ok**, o que é **aviso** (funciona em modo
degradado) e o que é **erro** (com o comando que resolve). Ele também detecta
`opencv-python` e `opencv-python-headless` instalados juntos (o ultralytics
depende de um, o easyocr do outro) e bancos da versão 0.1 em `runs/`.

### Baixar modelos (opcional)

```bash
python scripts/download_models.py --plate morsetech-n --coreml
```

Baixa o YOLO11n (COCO) e, no Apple Silicon, exporta para `data/models/yolo11n.mlpackage`
(sem NMS embutido, para que `iou` e `conf` da configuração continuem valendo). Com
`device: auto`, o `.mlpackage` ao lado do `.pt` é usado automaticamente — a detecção
cai de 7,3 para 5,1 ms e a latência p99 do pipeline, de ~24 para ~11 ms (§10).

**Não há peso oficial universal para detecção de placas.** `--plate` baixa um YOLO11 de
terceiros (AGPL-3.0, treinado em placas de vários países, métricas publicadas infladas
pelo próprio autor) para `data/models/plate_yolo.pt` e o exporta em 320 px: serve para
testar o pipeline neural, não para resultados da tese — para isso, veja a
[§9](#9-treinar-seus-próprios-modelos). Detalhes em [`data/models/README.md`](data/models/README.md).
Sem peso de placa, o detector clássico é usado.

A exportação CoreML exige `coremltools` (vem com `.[dl]` no macOS) e numpy ≤ 2.3.5; com
outro numpy, o ultralytics troca o do ambiente sozinho e a primeira exportação falha —
rode de novo.

---

## 5. Uso

### Interface gráfica

```bash
lprts run --source data/videos/sintetico.mp4    # arquivo
lprts run --source 0 --realtime                 # webcam, tempo real
lprts run --source rtsp://usuario:senha@camera/stream --realtime
```

Exige `pip install -e ".[ui]"`. Sem PySide6 o comando avisa e segue sem interface;
para forçar, use `--headless`.

A janela mostra o vídeo anotado, telemetria ao vivo (vazão, latência ponta a ponta,
descartes) e a tabela de placas. Leituras **confirmadas** aparecem em verde;
**provisórias**, em azul com `?`.

**Seleção da fonte.** A primeira linha da janela lista as câmeras da máquina, mais
"Arquivo de vídeo..." e "Stream de rede (RTSP/HTTP)...". As câmeras são descobertas
**sem abrir nenhuma delas** (abrir dispararia o pedido de permissão do macOS). O
botão **Testar** abre a fonte e verifica se ela entrega imagem. Escolher câmera ou
stream liga o modo tempo real; escolher arquivo desliga — é sugestão, não imposição.

**Veículos (YOLO).** A caixa liga o detector de veículos: ele encontra carros,
motos, ônibus e caminhões, e a placa é procurada dentro de cada um. Ela acompanha a
fonte: o `./start.sh` abre o vídeo sintético em modo só-placa (o YOLO não reconhece
veículos desenhados), e ao abrir um vídeo real ou uma câmera a caixa é marcada e o
detector de placas volta para `auto`. Desmarcada, as placas são rastreadas direto
no quadro.

**Objetos.** Os tipos de objeto que o YOLO detecta, rastreia e grava: `trafego`
(pessoas, bicicletas, carros, motos, ônibus e caminhões — o padrão), `completo`
(também semáforos, placas de pare, bancos, animais) ou `veiculos` (só os que portam
placa). O painel de telemetria mostra os objetos únicos por tipo e quantos são
usuários vulneráveis da via (pedestres, ciclistas, motociclistas).

**Limpar anteriores** (marcada por padrão): ao **Iniciar**, o banco
(`runs/lprts.db`) e os recortes das execuções anteriores são apagados — os dados
passam a descrever só o vídeo desta execução. A limpeza acontece depois de a fonte
abrir (um arquivo que não abre não custa os dados de antes) e só apaga um banco do
LPR-RTS. Desmarque para acumular execuções no mesmo banco
(`analytics.clear_previous: false`; na CLI, `--keep-previous`).

**Exportar** (`Ctrl+E`) grava **só o vídeo analisado** — a sessão da última execução
na janela —, numa pasta `<vídeo>_<sessão>/` com as tabelas filtradas e o resumo
`sessao.json`. Com execuções acumuladas, `lprts export --all` exporta o banco inteiro.

Atalhos: `Ctrl+R` iniciar · `Espaço` pausar · `Ctrl+.` parar · `Ctrl+D` procurar
câmeras · `Ctrl+E` exportar.

### Linha de comando

```bash
lprts run --source video.mp4 --headless               # sem interface
lprts run --source video.mp4 --save-video anotado.mp4 # grava o resultado
lprts run --source video.mp4 --ocr fastplate --device mps
lprts run --source video.mp4 --plate-detector classic --no-vehicle-detector
lprts run --source pasta_de_imagens/                  # pasta de imagens
lprts run --source gravacao.mp4 --recording-start 2026-03-14T08:00:00-03:00
```

`--recording-start` informa o instante real do primeiro quadro de um vídeo gravado:
cada evento ganha `media_time_utc` e as análises "por hora do dia" passam a ser
possíveis. Em câmera e stream ao vivo isso é automático.

| Comando | Função |
|---|---|
| `lprts doctor` | Verifica se a máquina está pronta para executar |
| `lprts diagnose` | Mostra em que estágio o pipeline para, num vídeo específico |
| `lprts info` | Ambiente, dispositivo e motores disponíveis |
| `lprts demo` | Gera vídeo sintético, executa e avalia — sem downloads |
| `lprts run` | Executa o pipeline (GUI ou headless) |
| `lprts eval` | Executa e avalia contra ground truth (`--realtime --repeat N` para tempo real) |
| `lprts eval-db` | Reavalia uma sessão já gravada, sem reprocessar o vídeo |
| `lprts compare` | Compara dois relatórios nos mesmos veículos (McNemar) |
| `lprts export` | Exporta os dados do vídeo analisado (CSV/Parquet/JSON); `--all` exporta o banco inteiro |
| `lprts sessions` | Lista as sessões gravadas (com o commit de cada uma) |
| `lprts report` | Tabelas e figuras descritivas para o artigo (sem textos de placa) |
| `lprts bench` | Mede o desempenho por estágio |
| `lprts calibrate` | Desenha linhas virtuais e calibra a homografia |
| `lprts batch` | Processa muitos vídeos (multiprocessos ou MPI) |
| `lprts tune` | Otimiza a fusão e emite a fronteira de Pareto |

### Perfis de configuração

```bash
lprts run --source 0 --config configs/tempo_real.yaml        # latência baixa
lprts run --source rtsp://... --config configs/tempo_real_rapido.yaml  # rodovia, carros rápidos
lprts eval --source v.mp4 --gt gt.csv --config configs/avaliacao.yaml
lprts run --source v.mp4 --config configs/avaliacao.yaml --config configs/minha_camera.yaml
```

| Perfil | Prioriza | Escolhas |
|---|---|---|
| `default.yaml` | equilíbrio | OCR a cada 2 quadros, 1280 px |
| `tempo_real.yaml` | latência | descarta quadros, 1280 px, OCR em quadros alternados, teto de ROIs |
| `tempo_real_rapido.yaml` | veículos rápidos | descarta quadros, resolução original, detecção e OCR em todo quadro |
| `avaliacao.yaml` | reprodutibilidade | todos os quadros, OCR síncrono, leituras sem censura, tudo gravado |
| `lote.yaml` | vazão agregada | offline, sem recortes, leituras sem censura |

Vários `--config` se combinam (o posterior sobrepõe o anterior). Opções da linha de
comando só sobrescrevem o YAML quando são passadas. **Chave desconhecida no YAML é
erro** (com sugestão do nome correto). A configuração efetiva e o seu hash são
gravados em cada sessão.

### Calibração (velocidade e contagem)

```bash
lprts calibrate --source video.mp4 --out configs/minha_camera.yaml
lprts run --source video.mp4 --config configs/avaliacao.yaml --config configs/minha_camera.yaml
```

Desenhe as **linhas virtuais** e marque **4 pontos no solo** com dimensões reais
conhecidas (largura de faixa de rodovia: 3,5 m; marca tracejada: 2 m de traço e 4 m de
intervalo). Os pontos ficam em pixels da imagem original e a calibração registra a
resolução em que foi feita: usá-la com uma fonte de outra resolução é erro.

> **Sem calibração, `speed_kmh` fica nulo** — um número em km/h sem base métrica seria
> pior que a ausência do dado. A velocidade usa o ponto de contato com o solo (meio da
> borda inferior da caixa) e o tempo de mídia do vídeo.

---

## 6. Dados extraídos para análise

Tudo vai para um **SQLite** (`runs/lprts.db` por padrão). Referência coluna a coluna
em [`docs/DICIONARIO_DE_DADOS.md`](docs/DICIONARIO_DE_DADOS.md); receitas em
[`docs/DADOS.md`](docs/DADOS.md).

**Cada processamento começa do zero.** `lprts run`, `lprts eval` e o **Iniciar** da
interface apagam o banco configurado e os recortes das sessões dele antes de gravar
(as repetições de um `lprts eval --repeat` ficam juntas). Para acumular vários vídeos
num banco, use `--keep-previous` ou `analytics.clear_previous: false` — ou o
`lprts batch`, que grava cada execução numa pasta própria e consolida os vídeos.

| Tabela | Uma linha por | Para quê |
|---|---|---|
| `sessions` | execução | configuração, commit, pacotes, hardware, energia, dispositivo efetivo, contadores |
| `frames` | quadro processado | latência ponta a ponta, fila, serviço por estágio, descartes, prazos |
| `detections` | detecção | reavaliar a detecção sem reprocessar |
| `track_points` | trilha × quadro | trajetória completa, métricas de rastreamento |
| `tracks` | trilha encerrada | permanência, trajetória, classe, placa final e publicada, velocidade |
| `reads` | tentativa de OCR | texto antes/depois da gramática, confiança por caractere, nitidez, tamanho |
| `events` | decisão publicada | "a placa lida", com o instante e o tipo da decisão (revisões inclusive; `v_decisions` traz só as vigentes) |
| `crossings` | cruzamento de linha | contagem por linha e sentido |

Três convenções valem para tudo: **coordenadas em pixels da imagem original**,
**tempo de mídia** como base física (o relógio de processamento só mede latência) e
`ts` = hora em que a linha foi produzida (o horário real do veículo é
`media_time_utc`).

**Toda classe é gravada**, não só as que têm placa: um pedestre não gera evento, mas
entra em `tracks` — é o que responde "quantos ciclistas passaram por este cruzamento".

```bash
lprts sessions                                    # sessoes gravadas (uma por execucao)
lprts export                                      # so o ultimo video analisado
lprts export --session <id> --format parquet      # um video especifico
lprts export --all                                # o banco inteiro, todas as sessoes
```

Cada exportação de um vídeo vai para `runs/export/<vídeo>_<sessão>/`: as tabelas e
visões **filtradas pela sessão** e o resumo `sessao.json` (configuração, ambiente,
desempenho, placas). Sem o filtro, vídeos diferentes se misturariam nas mesmas
tabelas, com `track_id` repetidos. Em R: `DBI::dbConnect(RSQLite::SQLite(), "runs/lprts.db")`.

> **Vídeo real tem dado pessoal.** Placas identificam pessoas (LGPD). Antes de
> compartilhar uma exportação, troque as placas por pseudônimos (receita em
> [`docs/DADOS.md`](docs/DADOS.md)) e não versione vídeos reais.

### Tabelas e figuras para o artigo

```bash
lprts report --db runs/lprts.db --out runs/relatorio          # um banco
lprts report --db runs/lote/<execucao>/consolidado.db           # todos os videos de um lote
```

Gera, para cada sessão, tabelas em CSV e Markdown (`relatorio.md`) e figuras em PDF
vetorial e PNG:

| Saída | Conteúdo |
|---|---|
| `videos.csv` | vídeo, resolução, duração, modo, detectores e runtime efetivos, processador, energia, commit |
| `objetos.csv` | objetos únicos por classe e categoria, por minuto, permanência mediana |
| `funil.csv` | por classe de veículo: veículos → placa detectada → tentativa de OCR → leitura válida → placa publicada; leituras e tempo até a decisão |
| `ocr_largura.csv` | leituras com texto e com sintaxe válida por largura da placa |
| `desempenho.csv` | vazão, latência p50/p95/p99/máx., prazo perdido, descartes, custo por estágio |
| `fig_composicao`, `fig_latencia`, `fig_estagios`, `fig_ocr_largura` | composição do tráfego, distribuição da latência contra o prazo, custo por estágio, OCR por tamanho de placa |

Objetos são contados por trilha **confirmada** — com permanência mínima
(`--min-dwell`, 0,5 s): uma trilha de poucos quadros é, em geral, o fragmento de um
veículo perdido numa oclusão. As duas contagens saem lado a lado. Nenhum texto de
placa sai nas tabelas (só contagens), e elas descrevem o que o sistema **mediu**:
acurácia exige ground truth (§7).

---

## 7. Avaliação científica

Protocolo completo em [`docs/AVALIACAO.md`](docs/AVALIACAO.md).

### Ground truth

```csv
frame_index,x1,y1,x2,y2,plate_text,object_class,track_id,kind,difficult
120,880,540,1010,585,ABC1D23,carro,3,plate,0
120,700,380,1180,640,,carro,3,object,0
```

`track_id` identifica o veículo ao longo dos quadros (é o que permite medir **por
veículo**); `difficult = 1` marca caixas que não contam nem como acerto nem como erro.
Para filmar e anotar vídeo real (CVAT e o conversor `scripts/cvat_para_gt.py`), veja
[`docs/COLETA.md`](docs/COLETA.md).

### Executar

```bash
lprts eval --source video.mp4 --gt anotacoes.csv \
           --config configs/avaliacao.yaml --out runs/relatorio.json
lprts eval-db --db runs/eval.db --gt anotacoes_corrigidas.csv     # sem reprocessar
```

### O que é reportado

- **Por veículo (métrica principal)**: revocação (veículos com a placa correta
  publicada ÷ veículos anotados — os perdidos contam), precisão dos eventos,
  duplicatas, fragmentação, latência de decisão. Com IC de 95%.
- **Efeito da fusão** nos mesmos veículos: leitura ao acaso × mais confiante × fusão ×
  decisão publicada, com teste de McNemar.
- **Por tentativa de OCR**: pareada por **posição** com a placa anotada, tentativas
  sem texto no denominador, IC agrupado por veículo, confusões e erros por posição.
- **Detecção** (P/R/F1, AP, mAP@[.5:.95]), **rastreamento** (MOTA, IDF1) e
  **desempenho** (vazão, latência p50/p95/p99, descartes, prazos).
- **Tabelas rotuladas** `relatorio_veiculos.csv`, `relatorio_leituras.csv`,
  `relatorio_eventos.csv`, prontas para modelagem estatística.

> Em LPR, uma placa com um caractere errado é **inútil**. A métrica de uso é a placa
> inteira correta, por veículo; o CER é reportado porque a literatura o usa.

### Comparar configurações

```bash
for motor in template easyocr fastplate; do
  lprts eval --source video.mp4 --gt gt.csv --config configs/avaliacao.yaml \
             --ocr $motor --out runs/ocr_$motor.json
done
lprts compare runs/ocr_template.json runs/ocr_fastplate.json
```

### Tempo real

```bash
lprts eval --source video.mp4 --gt gt.csv --config configs/tempo_real_rapido.yaml \
           --realtime --repeat 5 --out runs/tempo_real.json
```

Medido em vídeo sintético com veículos 1×, 2× e 3× mais rápidos (Apple M5 Pro, detector
clássico + `fast-plate-ocr`, em tempo real): revocação por veículo de 0,88 / 0,86 / 0,69 com
`tempo_real_rapido.yaml` (latência p99 < 9 ms, nenhum descarte). Desses, 0,61 → 0,69 em 3×
vêm de `fusion.exit_min_reads: 1`: o veículo rápido com 1–2 leituras, que antes saía de cena
sem decisão, agora publica na saída (com precisão mantida; §4 de
[`docs/AVALIACAO.md`](docs/AVALIACAO.md)). O perfil `tempo_real.yaml` da 0.2.0, que detectava
em quadros alternados e lia a cada 3 quadros, caía para 0,84 / 0,27 / 0,02 — economia de
processamento que a máquina não precisava, paga em acurácia.

O vídeo é lido na cadência nominal, com descarte e OCR assíncrono; como o resultado
depende da carga, as repetições são agregadas (média, desvio, mínimo, máximo). Reporte
lado a lado a acurácia offline (capacidade do método) e a de tempo real (o que o
sistema entrega).

### Dados sintéticos para ablação controlada

```bash
for desfoque in 0 3 5 7 9; do
  python scripts/make_synthetic_dataset.py --out /tmp/v$desfoque.mp4 \
      --frames 300 --blur $desfoque
  lprts eval --source /tmp/v$desfoque.mp4 --gt /tmp/v${desfoque}_gt.csv \
      --config configs/avaliacao.yaml --out runs/desfoque_$desfoque.json
done
```

O gerador anota `track_id`, caixas de objeto e `difficult` (placas pequenas, cortadas
ou ocluídas), desenha as placas numa fonte fora do banco do OCR por gabaritos (evita
vazamento) e grava os parâmetros em `<vídeo>_meta.json`. É determinístico: a mesma
semente gera os mesmos bytes (o `data/videos/sintetico.mp4` versionado é reproduzível).
Use para ablações; **use vídeo real para os resultados da tese**.

**Acurácia × velocidade.** `--speed` multiplica a velocidade dos veículos e
`--shutter-ms` simula a exposição da câmera (borrão de movimento proporcional à
velocidade aparente). Os mesmos veículos aparecem em toda condição (desenho pareado),
e `scripts/curva_velocidade.py` roda a grade inteira e grava tabelas prontas para
modelagem (por condição e por veículo, com velocidade aparente e borrão em pixels) —
ver [`docs/AVALIACAO.md`](docs/AVALIACAO.md#velocidade-e-borrão-de-movimento-vídeo-sintético-desenho-pareado).

---

## 8. Paralelismo e otimização

Documentação completa em [`docs/PARALELISMO.md`](docs/PARALELISMO.md).

| Problema | Ferramenta | Medido (Apple M5 Pro, vídeo real 4K, modelos neurais) |
|---|---|---|
| Um vídeo, núcleos ociosos | threads: preparo do quadro, detecção e (no perfil de avaliação) OCR nos workers | padrão 50,9 → 87,3 q/s (vertical) e 80,5 → 127,1 (horizontal); avaliação 21,5 → 63,4 q/s — resultados idênticos bit a bit |
| Muitos vídeos, uma máquina | processos | 194 → 461 q/s com 3 processos (2,38×; 0.2.0, baseline clássico) |
| Muitos vídeos, um cluster | **MPI** | lógica testada; não validado em cluster real |

**O paralelo é o padrão offline** (`detector.workers: -1`, automático): a interface e o
`lprts run` usam 4 workers (7 quando o OCR é antecipado), e o resultado é o do serial —
detecções, leituras e eventos verificados bit a bit nos dois vídeos, com testes. Em
**tempo real** o automático é serial: a máquina já acompanha a câmera, e a reordenação
só somaria latência. O perfil de avaliação também fica serial, porque em paralelo a
latência por quadro inclui a fila; para só a acurácia, `--parallel -1` dá o mesmo
resultado ~3× mais rápido. Detalhes e medições em [`docs/PARALELISMO.md`](docs/PARALELISMO.md).

```bash
lprts run --source video.mp4 --headless          # paralelo automático
lprts eval --source v.mp4 --gt gt.csv --config configs/avaliacao.yaml --parallel -1
lprts batch "corpus/*.mp4" --backend process --workers 8 --config configs/lote.yaml
```

### Otimização da fusão

```bash
lprts tune --db runs/avaliacao.db --gt anotacoes.csv --holdout 0.3 --trials 2000
```

Como `reads` guarda cada tentativa de OCR, varrer a fusão **não reprocessa vídeo**. A
verdade de cada trilha vem da posição das leituras, a decisão online é replicada, a
revocação é sobre veículos anotados e a fronteira é reavaliada em 30% dos veículos
reservados. Backends: `optuna`, `nsga2` (padrão) e `random` (linha de base honesta).

---

## 9. Treinar seus próprios modelos

O detector de placas é o componente que mais se beneficia de treino com dados
brasileiros.

### Datasets públicos brasileiros

| Dataset | Conteúdo | Acesso |
|---|---|---|
| **UFPR-ALPR** | 4.500 imagens, 150 veículos, placas brasileiras | pedido acadêmico aos autores |
| **RodoSol-ALPR** | 20.000 imagens, carros e motos, Mercosul e antigo | pedido acadêmico aos autores |
| **OpenALPR-BR** | conjunto menor, uso livre | público |

### Formato

```
dataset/
  images/train/*.jpg      labels/train/*.txt
  images/val/*.jpg        labels/val/*.txt
```

Cada `.txt` tem uma linha por placa: `0 <cx> <cy> <w> <h>` (normalizados, classe 0).

```yaml
path: /caminho/absoluto/para/dataset
train: images/train
val: images/val
names: {0: placa}
```

### Treinar

```bash
python scripts/train_plate_detector.py --data data/placas.yaml \
       --epochs 100 --device mps --export-coreml
```

**Espelhamento horizontal é desativado** — inverteria a ordem dos caracteres. O
melhor peso é copiado para `data/models/plate_yolo.pt` e passa a ser usado
automaticamente (`--plate-detector auto`). Separe treino e teste **por vídeo** ou
por local, nunca por quadro.

---

## 10. Desempenho e ajuste fino

### Medir antes de otimizar

```bash
lprts bench --source video.mp4 --frames 300                 # custo por estágio
lprts eval --source video.mp4 --gt gt.csv --realtime --repeat 5   # sob carga real
```

O `bench` mostra média, p50, p95, p99 e a fatia de cada estágio, e julga se o tempo
de relógio por quadro cabe no período de quadro. Reporte sempre **p95/p99**, chip,
estado de energia (tomada ou bateria) e versões — o `environment_json` de cada
sessão já tem tudo isso.

### Custo medido (Apple M5 Pro, na bateria)

`python scripts/benchmark_componentes.py` mede cada componente isolado, em cada
runtime disponível (200 repetições após 20 de aquecimento; JSON com a proveniência).
Medido na 0.5.0, quadro de 1280×720, em ms:

| Componente | p50 | p99 |
|---|---|---|
| Objetos YOLO11n — CoreML direto (`.mlpackage`, padrão no Apple Silicon) | 2,8 | 2,9 |
| Objetos YOLO11n — CoreML pelo ultralytics | 5,5 | 6,2 |
| Objetos YOLO11n — PyTorch/MPS (`.pt`) | 7,4 | 11,3 |
| Objetos YOLO11n — PyTorch/CPU | 15,9 | 17,7 |
| Placas, clássico — quadro inteiro (modo só-placa) | 4,7 | 5,1 |
| Placas, clássico — por ROI de veículo | 0,3 | 0,3 |
| Placas, YOLO 320 px por ROI — CoreML direto / pelo ultralytics | 0,9 / 1,9 | 1,0 / 2,0 |
| Placas, YOLO 320 px por ROI — PyTorch/MPS / CPU | 4,9 / 6,3 | 6,1 / 6,8 |
| OCR `fast-plate-ocr` / `easyocr` / `template`, por leitura | 11,0 / 10,7 / 0,4 | 12,6 / 13,8 / 0,6 |

O `fast-plate-ocr` usa 4 threads fixas, sem espera ativa (`ocr.threads`, 0.5): 1–2 ms a
mais por leitura que com o padrão do onnxruntime (todos os núcleos), gastando menos da
metade da CPU (27,7 contra 65 ms) e com as mesmas leituras.

Em 1920×1080 o YOLO custa o mesmo (a entrada é sempre 640 px); o detector clássico
de quadro inteiro sobe para 11,1 ms. O OCR roda fora do caminho crítico em tempo real.

**CoreML direto × ultralytics (0.4.0, Apple M4 Pro).** Os mesmos `.mlpackage`, os dois
runtimes no mesmo processo, quadro a quadro, sobre 239 quadros reais 4K reduzidos a
1280 px: objetos p50 **5,5** × 12,3 ms (p99 8,3 × 16,4); placa por veículo p50 **1,56** ×
3,65 ms. As caixas coincidem (IoU 1,000 em 1923 objetos e 911 de 912 placas). A inferência
no Neural Engine custa 1,6 e 0,5 ms; o resto é preparo da imagem e pós-processamento.
`scripts/benchmark_componentes.py` mede os dois caminhos.

Pipeline completo em tempo real (`tempo_real_rapido.yaml`, 30 fps, YOLO11n + placas
clássico + `fast-plate-ocr`; vídeo de conteúdo real, 900 quadros, 3 execuções
alternadas, aquecimento excluído):

| Detector de objetos | Latência p50 | p99 | Máximo | Prazos perdidos (33 ms) |
|---|---|---|---|---|
| CoreML | 9,0–9,2 | 11,2–11,7 | 16,2 | 0 |
| PyTorch/MPS | 17,3–18,2 | 23,6–26,6 | 29,1 | 0 |

No Apple Silicon, `device: auto` usa automaticamente o `.mlpackage` exportado ao lado
do `.pt` (`python scripts/download_models.py --coreml`), desde que ele respeite a
configuração: entrada igual a `imgsz` e **sem NMS embutido**. Um `.mlpackage` exportado
com `nms=True` fixa os limiares da exportação e o ultralytics passa a ignorar o `iou`
configurado sem avisar; o detector agora avisa e grava os valores efetivos. Para
forçar o PyTorch, use `device: mps`. O runtime, os pesos e os parâmetros efetivos de
cada componente ficam em `sessions.environment_json`.

**Vídeo real (iPhone, 4K HEVC, 30 qps), 0.3.1 × 0.4.0.** Apple M4 Pro na tomada, perfil
padrão com `--realtime` (o que a interface usa com **Tempo real** marcado), YOLO11n de
objetos + YOLO de placas + `fast-plate-ocr`. As versões rodaram **intercaladas**, porque a
máquina estava sob carga de serviços do macOS (`mediaanalysisd`, que também usa o Neural
Engine; carga média 16–30): os valores absolutos são pessimistas, a comparação vale.

| Vídeo | Versão | Quadros processados | Descartados | Latência p50 / p99 | Prazo de 33 ms perdido | Placas |
|---|---|---|---|---|---|---|
| 92 s, horizontal | 0.3.1 | 1083–1126 de 2763 | 60% | 144 / 407 ms | 99% | 25 |
| 92 s, horizontal | **0.4.0** | 2756–2761 | **0,2%** | **15 / 68 ms** | **2,7%** | 40 |
| 27 s, vertical | 0.3.1 | 675–707 de 809 | 15% | 163 / 316 ms | 98% | 9 |
| 27 s, vertical | **0.4.0** | 809 | **0%** | 44 / 206 ms | 84% | 14 |

O que mudou (CHANGELOG 0.4.0): o YOLO roda direto no CoreML (detecção p50 de ~60–70 para
13–19 ms), a placa não é procurada em veículo distante demais para ser lida, e o arquivo é
decodificado em hardware (AVFoundation; a captura de um quadro 4K HEVC caiu de ~46 para
~8 ms com a máquina livre). O vídeo **vertical** continua acima do prazo: o quadro retrato
de 1280×2276 px tem 3,2× os pixels do horizontal, e reduzi-lo custa caro com a máquina
ocupada. Para coletar dados, filme na horizontal; em vídeo vertical, `--width 960` alivia.

**Vídeo real, 0.4 × 0.5.** Apple M5 Pro na bateria, mesmos dois vídeos do iPhone,
YOLO11n de objetos + YOLO de placas no CoreML + `fast-plate-ocr`. Vazão = quadros ÷
duração da sessão. Leituras e eventos idênticos entre as versões em todas as linhas
(a 0.5 só mudou onde cada coisa roda):

| Perfil | Vídeo | 0.4 | 0.5 |
|---|---|---|---|
| padrão, offline (interface) | 27 s, vertical | 50,9 q/s | **87,3 q/s** (4 workers) |
| padrão, offline (interface) | 92 s, horizontal | 80,5 q/s | **127,1 q/s** (4 workers) |
| avaliação, `--parallel -1` | 27 s, vertical | 21,5 q/s | **63,4 q/s** (7 workers) |
| tempo real (serial) | 27 s, vertical | p50/p99 16,9/20,1 ms, 0 descarte | 16,8/21,4 ms, 0 descarte |
| tempo real (serial) | 92 s, horizontal | 13,1/19,5 ms, 0 descarte | 12,7/18,2 ms, 0 descarte, **−36% de CPU** |

Nesta máquina e nesta carga os dois vídeos já ficavam dentro do prazo em tempo real
com a 0.4 (a tabela anterior, do M4 Pro sob carga, mostra o caso pior). O OCR do
`fast-plate-ocr` passou a usar 4 threads fixas e sem espera ativa (`ocr.threads`): cada
leitura leva ~10,7 ms em vez de ~9,4, mas gasta 27,7 ms de CPU em vez de 65 — a folga
que a detecção e a captura disputavam.

### Se estiver lento

| Ajuste | Efeito |
|---|---|
| `--width 960` | menos pixels na detecção (coordenadas gravadas continuam originais) |
| `detector.imgsz: 512` | entrada menor no YOLO |
| `detector.detect_every_n: 2` | detecção alternada; o Kalman só prevê nos intermediários |
| `detector.max_rois_per_frame: 6` | teto de busca de placas por quadro |
| `--no-vehicle-detector` | pula a busca por veículos; rastreia placas direto |
| `ocr.ocr_every_n: 4` | menos OCR por trilha |
| `ocr.max_ocr_per_frame: 2` | teto de OCR por quadro |
| `python scripts/download_models.py --coreml` | CoreML no Apple Silicon: latência p99 ~2× menor (§10) |
| `analytics.store_detections: false` | menos escrita em execução 24/7 |

### Se a precisão estiver baixa

| Sintoma | Ajuste |
|---|---|
| Placas não detectadas | `--plate-detector yolo` com modelo treinado; baixe `plate_conf` |
| Texto errado | motor de OCR profundo; aumente `min_plate_width` |
| Placas trocadas entre veículos | aumente `tracker.match_iou` e `min_hits` |
| Decisões erradas publicadas cedo | aumente `fusion.min_reads` e `min_agreement` (meça com `lprts tune`) |
| Placas fantasma | `fusion.require_valid_syntax: true` |

---

## 11. Estrutura do projeto

```
src/lprts/
├── cli.py              CLI (typer)
├── doctor.py           diagnóstico do ambiente
├── provenance.py       commit, pacotes, hardware e energia de cada sessão
├── runtime.py          configuração de runtimes nativos (telemetria ONNX)
├── diagnose.py         diagnóstico do pipeline (onde o fluxo para)
├── config.py           configuração tipada + YAML (chaves desconhecidas = erro)
├── factory.py          construção dos componentes e registro da sessão
├── core/
│   ├── types.py        estruturas compartilhadas (tempo de mídia, escala)
│   ├── taxonomy.py     classes de objeto, categorias, porte de placa
│   ├── pipeline.py     etapa de detecção + etapa sequencial; ciclo de vida das trilhas
│   ├── parallel.py     detecção paralela + reordenação por sequência
│   ├── batch.py        lote de vídeos (processos ou MPI)
│   └── clock.py        FPS, percentis de latência
├── capture/            fontes (arquivo, webcam, RTSP, pasta), filas, descoberta de câmeras
├── detect/             CoreML/MPS/CUDA/CPU, YOLO (ultralytics/ONNX), detector clássico
├── track/              ByteTrack, Kalman com Δt, húngaro
├── ocr/                gramática brasileira, gabaritos, adaptadores dos motores
├── fusion/voting.py    votação multi-frame e critério de publicação
├── optimize/           Pareto/NSGA-II, espaços de busca, tuner, refusão offline
├── analytics/          store (SQLite, esquema 2), tráfego, métricas, exportação
├── eval/               ground truth, métricas, avaliação da sessão, estatística, relatório
├── ui/                 app Qt, widgets, overlay, calibração
└── utils/              geometria, processamento de imagem, logging

start.sh    ponto de entrada único (cria ambiente, instala, abre a interface)
scripts/    setup_macos.sh · download_models.py · make_synthetic_dataset.py
            train_plate_detector.py · benchmark.py · benchmark_componentes.py · curva_velocidade.py
            cvat_para_gt.py · refinar_caixas.py · bateria_artigo.sh
            artigo/ (análise, figuras e diagnósticos do artigo)
configs/    default.yaml · tempo_real.yaml · tempo_real_rapido.yaml · avaliacao.yaml · lote.yaml
            artigo/ (as variantes da bateria do artigo)
tests/      ~670 testes, incluindo as regressões da auditoria de dados
docs/       ARQUITETURA · DADOS · DICIONARIO_DE_DADOS · AVALIACAO · COLETA · PARALELISMO
            modelos/registro_de_campo.csv
            artigo/ (resultados e plano de análise registrado)
CHANGELOG.md · CITATION.cff
```

---

## 12. Solução de problemas

**Não sei se está tudo certo para rodar**
`lprts doctor` responde item a item, com o comando que resolve cada pendência.

**"O banco ... foi criado por outra versão do LPR-RTS"**
Bancos da 0.1 têm erros conhecidos e não podem ser misturados com os novos. Mova o
arquivo (ex.: `mkdir -p runs/legado && mv runs/lprts.db* runs/legado/`) ou use outro
banco com `--db`.

**"Opção de configuração desconhecida" / "foi renomeada"**
O YAML tem uma chave que não existe (o erro sugere a mais parecida) ou uma chave
antiga: `tracker.match_thresh` virou `tracker.match_iou` (IoU mínimo; padrão 0,2).

**"O motor de OCR '...' foi pedido explicitamente mas não pode ser carregado"**
Instale o motor ou use `--ocr auto` (escolhe o melhor disponível e registra qual).

**`ModuleNotFoundError: No module named 'lprts'`**
O ambiente virtual não está ativo: `source .venv/bin/activate`.

**A interface gráfica não abre**
`lprts doctor` diz se o PySide6 está instalado. Se faltar: `pip install -e ".[ui]"`.
Sem ela, use `--headless`.

**A janela de calibração não abre**
Provavelmente `opencv-python-headless` sobrescreveu o `opencv-python` (o `doctor`
avisa). Reinstale: `pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python`.

**FPS baixo**
Rode `lprts bench` primeiro: ele diz qual estágio domina. Depois, a [§10](#10-desempenho-e-ajuste-fino).

**A tabela "Placas reconhecidas" fica vazia**

```bash
lprts diagnose --source data/videos/sintetico.mp4
```

Ele processa alguns quadros contando o que sai de cada etapa —
`quadros → objetos → objetos com placa → placas → trilhas → leituras → eventos` — e
nomeia a primeira que zera. A causa mais comum: com os pesos YOLO instalados, o sistema
procura **veículos** primeiro, e o detector treinado em fotos reais não reconhece os
veículos de vídeo sintético. O fallback automático resolve sozinho; para fixar:

```bash
lprts run --source video.mp4 --no-vehicle-detector --plate-detector classic
```

Outras causas: placa estreita demais (`ocr.min_plate_width`, em pixels da imagem
processada), poucas leituras concordantes (`fusion.min_reads`), ou leitura ainda
provisória (azul com `?`).

**"Python encerrou-se inesperadamente" no macOS**

Causa identificada por relatório de crash: **a telemetria do ONNX Runtime**, que
tentava enviar dados por HTTP durante o encerramento. O sistema a desliga no arranque.
Se ainda ocorrer: `lprts run --source video.mp4 --device cpu`.

O modelo de OCR roda em CPU por padrão mesmo no Apple Silicon: o grafo tem 384 nós, dos
quais o CoreML aceita 239, em **56 partições** — o custo das trocas supera o ganho. Para
medir: `ocr.force_coreml: true`.

**O OCR não devolve nenhuma leitura**
`lprts diagnose --source video.mp4`. O sistema avisa após 5 falhas seguidas do motor,
com o erro real. Para isolar: `lprts run --source video.mp4 --ocr template`.

**Stream RTSP travando**
Use `--realtime` e reduza com `--width 960`. O sistema reconecta sozinho com backoff
exponencial; a linha do tempo da sessão continua contínua.

**`sqlite3.OperationalError: database is locked`**
Feche outros processos lendo o banco, ou use uma cópia. O modo WAL já permite leitura
concorrente.

---

## 13. Limitações conhecidas

- **Tempo real brando, não rígido.** Não há garantia de prazo; o sistema descarta,
  mede e reporta. A latência medida começa quando o quadro fica disponível ao
  programa (não inclui exposição do sensor nem buffer do driver).
- **Decisão por votação.** Revisar a decisão (`publish_policy: revise`) corrige a
  trilha que começou na placa errada, mas também segue um erro **sistemático** do OCR
  (ex.: Q lido como O quando a placa se aproxima): a votação não distingue os dois.
  Meça antes de adotar (§7).
- **O detector clássico é um baseline, não um resultado.** Sensível à iluminação,
  falha com placas inclinadas ou menores que ~25 px de altura. Para a tese, treine o
  detector.
- **O OCR por gabaritos** usa fontes Hershey, não a fonte oficial das placas, e sua
  confiança não é calibrada (a leitura "mais confiante" pode ser pior que uma ao acaso).
- **Confianças não são comparáveis entre motores.** Calibre por motor antes de usá-las
  como covariável ou limiar.
- **Sem re-identificação entre câmeras.** O `track_id` é local à sessão.
- **Velocidade.** A homografia assume o ponto sobre o plano do solo; usa-se o contato
  pneu-solo. No modo só-placa o ponto está na placa (~0,5 m acima do solo): há viés.
- **Placas de moto** (duas linhas) são segmentadas pelo OCR clássico, mas os motores
  profundos testados assumem linha única.
- **Não validado sob chuva, neblina ou noite.**
- **A detecção multi-classe depende dos pesos YOLO.** Sem eles o sistema opera em modo
  só-placa e a composição do tráfego não é medida.
- **A busca de hiperparâmetros automatizada cobre só a fusão.** Os espaços `realtime` e
  `tracker` exigem reprocessar o vídeo e devem ser conduzidos por script.
- **MPI foi implementado mas não validado em cluster real.**
- **Vídeo sintético** não tem sujeira, reflexo, deformação nem compressão de câmera.
- **Validação de produção não feita.** Falta avaliar os modelos em vídeos anotados
  separados do treino (noite, reflexo, chuva, oclusão, placas pequenas), medir
  precisão, revocação, tempo até a placa correta, descartes e memória sob carga
  contínua, repetir em cada hardware de destino (Mac, NVIDIA, dispositivo de borda)
  com o número de câmeras, quadros/s e prazo exigidos, e testar reconexão,
  armazenamento lento, reinício, execuções longas e paridade dos modelos exportados.
- **Chamada nativa travada.** Uma thread Python não encerra com segurança uma
  inferência ou leitura de câmera presa no código nativo: o sistema limita a espera,
  reporta a falha e não libera o modelo em uso. Recuperação automática exige
  supervisão de processo na implantação.

### Direções naturais de continuação

- Detector de placas orientado (*oriented bounding boxes*) para corrigir perspectiva
  antes do OCR
- Revisão da decisão ponderada pela qualidade da leitura (tamanho e nitidez da placa),
  para não seguir erros sistemáticos do OCR
- Re-identificação multi-câmera com descritores de aparência
- Quantização INT8 dos modelos para ganho adicional no Neural Engine
- Calibração das confianças, para separar "não sei" de "tenho certeza"

---

## Licença

O código está sob a **GNU Affero General Public License v3.0 ou posterior** (AGPL-3.0-or-later);
o texto está em [LICENSE](LICENSE). Quem distribuir o programa, ou uma versão modificada, ou
oferecê-lo como serviço em rede, precisa publicar o código-fonte sob a mesma licença.

Os modelos de terceiros não estão neste repositório: o `scripts/download_models.py` os baixa,
cada um com a sua licença. O YOLO11 da Ultralytics e o detector de placas do morsetechlab estão
sob AGPL-3.0; o `fast-plate-ocr`, sob MIT. Os dados do artigo têm licença própria, indicada no
Zenodo.

## Como citar

```bibtex
@software{lprts2026,
  author  = {Traue, Thiago G.},
  title   = {LPR-RTS: Sistema de Reconhecimento de Placas Veiculares em Tempo Real},
  year    = {2026},
  version = {0.5.0},
  note    = {Prova de conceito para pesquisa de doutorado},
  url     = {https://github.com/traue/lpr-rts}
}
```

Ao publicar resultados, cite também o commit gravado em `sessions.code_version`.
