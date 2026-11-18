# Arquitetura

Documento técnico de referência. Para instalação e uso, veja o
[README](../README.md); para os dados gravados, o
[dicionário de dados](DICIONARIO_DE_DADOS.md).

---

## 1. Visão geral

O pipeline troca dados entre estágios por meio das dataclasses de
`core/types.py`; nenhum estágio conhece a implementação do seguinte, o que
permite trocar peças e medir o efeito.

```
Frame ──▶ [Detection] ──▶ [TrackState] ──▶ [PlateRead] ──▶ FusedPlate ──▶ FrameResult
```

O processamento de um quadro tem duas etapas de naturezas opostas:

| Etapa | Função | Estado | Onde roda |
|---|---|---|---|
| `VideoSource.prepare` | redução e rotação do quadro decodificado | **sem estado** | leitura (serial) ou workers (paralelo, arquivo offline) |
| `detect_stage` | detecção de objetos e de placas | **sem estado** | thread principal (serial) ou workers (paralelo) |
| `prefetch_ocr` | OCR das placas detectadas, só no perfil que lê toda placa em todo quadro | **sem estado** | workers (paralelo offline, com veículos) |
| `sequential_stage` | rastreamento, OCR (ou a leitura pronta), fusão, analítica, gravação | **com estado, em ordem** | sempre em série |

Serial e paralelo executam literalmente o mesmo `sequential_stage` — é o que
garante resultados idênticos (verificados bit a bit em vídeo real; ver
[PARALELISMO.md](PARALELISMO.md)). Offline, o paralelo é o padrão
(`detector.workers: -1`); em tempo real e no perfil de avaliação, o serial.

### Duas convenções que tudo respeita

- **Tempo.** Cada `Frame` carrega o tempo de mídia (`source_pts`: PTS do
  arquivo, ou chegada desde a abertura da fonte ao vivo) e o relógio de
  processamento (`timestamp`, `time.monotonic()`). Toda medida física —
  velocidade, permanência, trajetória — usa o tempo de mídia; o relógio de
  processamento só mede latência. *(Na 0.1 a base física era o relógio de
  processamento: a mesma trilha saía a 219 km/h numa máquina rápida e a
  17 km/h numa lenta.)*
- **Coordenadas.** O pipeline trabalha na resolução processada
  (`resize_width`); `Frame.scale` guarda a razão processada/original, e tudo o
  que é gravado volta para pixels da imagem original — o espaço do ground truth
  e da calibração. Em vídeo de celular gravado na vertical, a imagem original é a
  **girada** (como o player a exibe). Para não girar o quadro 4K inteiro, a
  captura reduz o quadro cru e gira só a imagem reduzida, e o recorte do OCR
  converte as coordenadas e gira só o recorte (`crop_rotated`).

---

## 2. Modelo de tempo real

O sistema é de **tempo real brando** (*soft real-time*), de melhor esforço: não
há garantia de prazo — o macOS não é um sistema operacional de tempo real, o
coletor de lixo do Python pode pausar, e o custo de um quadro varia com a cena.
O que o sistema faz é **degradar de forma controlada e medir quanto degradou**.

### 2.1 Descarte na captura

A captura roda numa thread própria e, em arquivos no macOS, decodifica pelo
AVFoundation (VideoToolbox, em hardware): um quadro 4K HEVC custa 8 ms, contra
46–48 ms pelo FFmpeg em software — acima do período de quadro de 33 ms, o que fazia
da captura o gargalo. O backend efetivo fica em `environment_json.video_backend`.

`DropOldestQueue` mantém uma fila limitada e, ao encher, descarta o item **mais
antigo**: exibir um quadro de 2 segundos atrás é pior que nunca tê-lo
processado. A fila de captura tem 2 posições, o que limita a espera a cerca de
dois quadros.

### 2.2 OCR assíncrono

O OCR roda num `ThreadPoolExecutor`; o laço principal submete o **recorte** da
placa (uma cópia pequena, não o quadro inteiro) e colhe o resultado em quadros
seguintes. Cada leitura é gravada com o quadro de onde o recorte saiu
(`frame_index`) e com o quadro em que entrou na fusão (`collected_frame`).

**Detecta pequeno, lê grande.** A detecção roda na imagem reduzida
(`source.resize_width`, 1280 px), mas o recorte do OCR sai do quadro **original**
(`ocr.full_resolution`): a captura guarda uma referência ao quadro decodificado, sem
cópia nem custo de detecção. Em vídeo 4K, placas de 35–45 px (a 1280) passaram de
43% para 83% de leituras iguais à decisão final do veículo; a partir de 45 px os
dois recortes empatam.

### 2.3 Detecção sem sobrecarga de framework

Com pesos `.mlpackage`, o YOLO roda pelo `coremltools` no Neural Engine, com
letterbox, decodificação e NMS próprios (`detector.runtime: auto`). Com
`device: auto` no Apple Silicon, o `.mlpackage` exportado ao lado do `.pt`
(`scripts/download_models.py --coreml`) é usado no lugar dele, desde que tenha a
entrada igual a `imgsz` e nenhum NMS embutido. Pelo ultralytics, o preparo da imagem e o pós-processamento em PyTorch custavam 3–4× a
própria inferência; em quadros reais as caixas dos dois caminhos coincidem (IoU
1,000). `detector.runtime: ultralytics` volta ao caminho de referência.

### 2.4 Orçamento por quadro

- `ocr.max_ocr_per_frame` limita as leituras por quadro, priorizando
  `área_da_placa × (2 se a trilha ainda não publicou, senão 1)`;
- `detector.max_rois_per_frame` limita a busca de placas aos N maiores
  veículos (padrão 0 = sem teto; o perfil de tempo real usa 6);
- `detector.plate_width_ratio` pula a busca de placa em veículos estreitos demais
  para que a placa chegue a `ocr.min_plate_width` (a placa ocupa no máximo ~38% da
  largura do veículo); cada busca custa uma inferência do YOLO de placas;
- `detector.detect_every_n` detecta a cada N quadros; nos intermediários o
  rastreador só **prevê** (sem medida, sem OCR). Reduz o custo médio, não a
  latência de um quadro. *(Na 0.1 os quadros intermediários reenviavam as
  detecções antigas: a velocidade do Kalman ia a zero e, com n = 3, um veículo
  virava 20 trilhas.)*

### 2.5 O que é medido, por quadro

| Medida | Significado |
|---|---|
| `latency_ms` | ponta a ponta: do quadro disponível ao resultado pronto |
| `queue_ms` | espera na fila antes do processamento |
| `total_ms` | tempo de serviço (soma dos estágios) |
| `dropped_frames`, `frames_skipped` | descartes acumulados e desde o quadro anterior |
| `deadline_miss` | latência acima do prazo declarado (padrão: um período de quadro) |
| `ocr_in_flight` | leituras assíncronas pendentes |

A latência não inclui exposição do sensor nem buffer do driver da câmera — o
OpenCV não expõe esse instante de forma confiável.

### 2.6 Modo offline

Com `realtime: false`: nenhum descarte, OCR síncrono, todos os quadros
processados. É o modo para medir a **capacidade** do método de forma
determinística; a acurácia **sob restrição de tempo real** se mede com
`lprts eval --realtime --repeat N` (ver [AVALIACAO.md](AVALIACAO.md), §5).

---

## 3. Detecção

### 3.1 Dois estágios

Com detector de objetos, a busca por placas acontece **dentro da ROI de cada
veículo** que pode portar placa (a taxonomia exclui pedestres e bicicletas).
É mais rápido *e* mais preciso: a placa ocupa fração maior da entrada do
detector, e falsos positivos fora de veículos desaparecem por construção.

### 3.2 Associação placa → veículo

Cada placa detectada vai para a trilha de veículo que a contém (≥ 50% da área da
placa dentro da caixa do veículo). Três regras evitam placas duplicadas ou trocadas:

- **NMS entre classes** (`detector.vehicle_merge_iou`, 0,7; desde a 0.3.1). O NMS
  do YOLO é por classe, e o mesmo veículo saía como `car` e `truck` (IoU
  0,98–0,99): duas trilhas, a placa publicada duas vezes. Fica a caixa de maior
  score, e só entre veículos que portam placa (piloto e moto continuam separados).
- **Só quem porta placa recebe placa** (desde a 0.4.0). A caixa do piloto de uma
  moto, ou de um pedestre na frente do carro, ficava com a placa, e a trilha da
  pessoa nunca passa pelo OCR.
- **Placa contida em dois veículos** (caixas vizinhas se sobrepõem em
  perspectiva; desde a 0.3.1): fica com o veículo cujo centro horizontal está mais
  perto dela, relativo à largura dele — a placa traseira é centrada. Antes o empate
  dependia da ordem da lista, que muda de quadro a quadro.

Limite conhecido: um caminhão próximo, com a caixa cobrindo meio quadro, contém o
carro à frente e pode herdar a placa dele (CHANGELOG 0.4.0).

### 3.3 Modo só-placa

Sem detector de objetos (ou após o fallback automático), as **placas** são os
objetos rastreados. Tratar o quadro inteiro como um objeto faria todas as placas
votarem juntas numa placa quimera (`test_modo_so_placa_rastreia_placas`).

O fallback automático troca para este modo quando o detector de objetos não vê
nada que porte placa por `fallback_after_frames` quadros mas há placas
visíveis. As trilhas em curso são encerradas (motivo `reset`) e gravadas, e a
troca é anotada na sessão com o quadro em que ocorreu. Desligado no perfil de
avaliação.

### 3.4 Detector clássico

Com `plate_detector: auto`, o YOLO de placas é usado quando os pesos
(`detector.plate_weights`) existem; sem eles, este baseline — o que garante que o
sistema roda sem downloads. Baseline morfológico: Sobel em X, fechamento com núcleo largo e baixo,
contornos filtrados por proporção (1,9–5,5), altura e preenchimento, e um score
heurístico em [0, 1] que **não é probabilidade**.

---

## 4. Rastreamento

### 4.1 ByteTrack em duas passadas

1. Detecções de **alta** confiança contra todas as trilhas, IoU ≥ `match_iou`;
2. detecções de **baixa** confiança contra as trilhas livres vistas
   recentemente, IoU ≥ `second_match_iou`.

| Perfil | `match_iou` | `second_match_iou` | `high_thresh` |
|---|---|---|---|
| Veículo | 0,2 | 0,5 | 0,5 |
| Placa | 0,3 | 0,15 | 0,3 |

Os valores de veículo são os **efetivos do ByteTrack de referência**, onde
`match_thresh = 0.8` é um custo máximo (1 − IoU), isto é, IoU ≥ 0,2. *(A 0.1
lia o 0,8 como IoU mínimo: um carro de 200 px a 25 px/quadro — ~60 km/h em escala
típica — nunca era confirmado e, portanto, nunca era lido.)*

### 4.2 Filtro de Kalman

Estado `[cx, cy, a, h, vcx, vcy, va, vh]` (formulação SORT), atualização na
forma de Joseph. A predição avança o **número real de quadros** decorridos desde
a última atualização (descarte, `detect_every_n`), e `max_age` é contado em
quadros da fonte — a tolerância a oclusão é a mesma em segundos qualquer que
seja a carga.

### 4.3 Atribuição

Em cada passada, o emparelhamento maximiza a soma de `IoU − limiar` sobre os
pares acima do limiar (`max_weight_assignment`): é o critério do ByteTrack de
referência, `lap.lapjv(1 − IoU, extend_cost=True, cost_limit=1 − limiar)`, em
que linhas e colunas podem ficar sem par. Os testes conferem o ótimo contra
força bruta e a escolha contra a matriz estendida do `lap`.

Maximizar o **número** de pares — o critério do MOTA no py-motmetrics, usado só
na avaliação (`gated_assignment`) — troca um par forte por dois fracos. A
revisão anterior à 0.5 fazia isso no rastreador: no vídeo real de 92 s, uma
trilha de carro seguiu num pedestre por 118 quadros e a placa saiu duplicada.
Uma trilha é associável enquanto `time_since_update ≤ max_age`; o código de
referência aceita um quadro a mais, porque expira as trilhas depois de associar.

O solver é uma implementação própria em numpy (Jonker-Volgenant, sem depender
do scipy), com o laço por coluna vetorizado — o mesmo resultado, bit a bit, do
laço escalar (verificado nos testes, inclusive com empates). Linhas e colunas
sem nenhum par admissível saem da conta antes, o que encolhe as matrizes
esparsas de cena.

### 4.4 Ciclo de vida

- Uma trilha é **confirmada** com `min_hits` associações; só trilhas
  confirmadas chegam ao resto do pipeline.
- Quando expira (`max_age`), o rastreador a devolve (`pop_removed`); o pipeline
  espera o OCR em voo dela, faz a última fusão, publica se cabível e grava o
  estado final em `tracks`. O mesmo acontece com todas as trilhas vivas no fim
  do vídeo.
- Os IDs são por rastreador (recomeçam a cada sessão) e não se repetem dentro
  da sessão, mesmo após o fallback.

---

## 5. OCR e gramática

### 5.1 Motores

Todos devolvem o mesmo `PlateRead`. Pedir um motor ausente é **erro**; só
`auto` escolhe o melhor disponível (e a escolha fica na sessão). Cada motor
declara o runtime e o dispositivo efetivos (`describe()`).

A **confiança tem escala própria por motor** e não é calibrada: média das
probabilidades dos caracteres (`fastplate`, só sobre os caracteres reais —
o modelo global tem 9 posições, 2 de preenchimento), score dos trechos
(`easyocr`, `paddle`), conf/100 (`tesseract`), correlação reescalada para
[0, 1] (`template`, em que 0,5 = nenhuma correlação).

O `fast-plate-ocr` recebe a imagem em RGB (o modelo é RGB; o OpenCV entrega BGR).

### 5.2 Gramática brasileira

`plates_br.correct` encaixa o texto em Mercosul (`LLLNLNN`) ou antigo
(`LLLNNNN`), trocando confusões visuais (`O/0`, `I/1`, `B/8`...) pela posição. No
máximo **2 trocas** (`ocr.max_grammar_corrections`): com 3, lixo virava placa
(`1231231` → `IZB1231`). O número de trocas é gravado (`reads.n_corrections`) e
`raw_text` guarda a saída **antes** da gramática — no OCR por gabaritos, a
decodificação livre, sem máscara.

### 5.3 OCR por gabaritos

36 caracteres × 6 variantes (Hershey SIMPLEX, DUPLEX, TRIPLEX, duas espessuras),
32 × 48, letterbox preservando a proporção, correlação normalizada como um
produto matriz-vetor, decodificação restrita às máscaras brasileiras. O hash do
banco de gabaritos vai para a sessão. Por usar fontes Hershey, **vídeos
sintéticos com placas Hershey favorecem este motor** — o gerador do projeto usa
DejaVu Sans Bold.

---

## 6. Fusão multi-frame

### 6.1 Peso do voto

```
peso = confiança × (√(nitidez_relativa) + 0,5) × (1,35 se sintaxe válida)
```

e, quando o motor dá confiança por caractere alinhada ao texto, cada posição é
ponderada também por ela.

### 6.2 Votação por caractere

Cada posição é votada independentemente — aproveita quadros parcialmente
corretos. Sem leituras de 7 caracteres, cai para a votação por string.

### 6.3 Publicação

A fusão diz qual é a melhor estimativa a cada momento; a **política de
publicação** (`fusion.publish_policy`, em `lprts/fusion/publication.py`) diz quando
ela vira um evento. O pipeline e a réplica offline do `lprts tune` usam a mesma
classe.

| Política | Decide | Custo |
|---|---|---|
| `first_stable` (padrão) | na primeira fusão que atende a `min_reads`, `min_agreement`, `min_confidence` e, se exigido, sintaxe válida | decisão errada cedo fica errada |
| `revise` | idem, e publica uma revisão quando uma fusão estável posterior muda o texto | segue também um viés sistemático do OCR (ver AVALIACAO.md) |
| `on_exit` | uma vez, quando a trilha termina | latência de decisão = permanência em cena |

`fusion.exit_min_reads > 0` deixa a trilha que termina sem decisão publicar com
menos leituras (`decision = 'exit_relaxed'`). O critério vale igual no laço
principal, quando a trilha termina e no fim do vídeo. Antes da primeira decisão,
o texto provisório acompanha a fusão corrente. A diferença entre a decisão
vigente e a fusão final (`tracks.published_text` × `tracks.plate_text`) mede o
custo da política.

`fusion.max_reads_per_track` (0 = todas) mantém só as leituras de maior
confiança; altera a votação, por isso o perfil de avaliação usa 0.

---

## 7. Persistência

### 7.1 Por que SQLite

Roda num Mac sem containers nem serviços, é um único arquivo, e o DuckDB o lê
diretamente. Em modo WAL sustenta com folga a taxa de escrita do pipeline.

### 7.2 Escrita assíncrona e integridade

O pipeline só enfileira; uma thread grava em lotes dentro de transações.

- **Resultados nunca são descartados**: eventos, trilhas, leituras e
  cruzamentos esperam espaço na fila.
- **Telemetria volumosa** (quadros, detecções, pontos de trilha) pode ser
  descartada se o disco não acompanhar — nunca se bloqueia o pipeline por
  telemetria —, mas cada perda é contada por tabela e gravada na sessão
  (`stats_json.rows_lost`).
- Um lote com erro é regravado linha a linha; só as linhas inválidas se perdem
  (e são contadas).
- Bancos do esquema 2 (0.2) são migrados no lugar, sem perda; os do esquema 1
  (0.1), com erros conhecidos, são recusados.
- Cada processamento começa de um banco vazio (`analytics.clear_previous`; ver
  [DADOS.md](DADOS.md) §1). Só bancos do LPR-RTS são apagados, e só depois de a
  fonte abrir: um caminho de vídeo errado não custa os dados anteriores.

### 7.3 Sincronização do flush

Esvaziar a fila não basta: a escritora já retirou itens para um lote ainda não
commitado. Um marcador `_FlushRequest` atravessa a fila e só libera quem espera
depois do commit (`test_flush_sincroniza_antes_da_leitura`).

### 7.4 Proveniência

`factory.open_session` é o único caminho para abrir uma sessão (CLI, interface,
lote e avaliação): grava configuração, hash da configuração, commit do código,
versões de pacotes, hardware, energia e os componentes efetivos.

---

## 8. Interface

O pipeline **nunca** roda na thread da interface. `PipelineWorker` é uma
`QThread` que emite **cópias** dos resultados (o pipeline continua alterando as
trilhas enquanto a interface desenha). A interface repinta em cadência própria
(~30 Hz) e mostra a latência ponta a ponta. Leituras **confirmadas** e
**provisórias** são visualmente distintas (cor e sufixo `?`).

---

## 9. Decisões que poderiam ter sido outras

| Decisão | Alternativa | Por que a atual |
|---|---|---|
| Rastrear veículos, não placas | rastrear placas sempre | a placa some em curvas; o veículo persiste |
| Votação por caractere | por string | aproveita quadros parcialmente corretos |
| Publicar na primeira fusão estável (padrão de `fusion.publish_policy`) | revisar (`revise`) ou esperar a trilha terminar (`on_exit`) | latência de decisão baixa; no sintético pareado, `revise` e `on_exit` não ganharam (+7/−10 veículos, AVALIACAO §3) |
| SQLite | DuckDB nativo, Postgres | zero dependências; DuckDB lê o arquivo |
| Húngaro próprio | scipy | evita uma dependência grande por uma função |
| ByteTrack próprio | supervision, boxmot | integração direta com `TrackState` |
| Velocidade nula sem calibração | estimativa aproximada | número sem base métrica é pior que ausência de dado |
| Gravar tudo (detecções, pontos, leituras) | só o consolidado | reavaliar e reotimizar sem reprocessar vídeo |
