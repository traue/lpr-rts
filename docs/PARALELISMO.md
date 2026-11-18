# Paralelismo e otimização

Como o sistema usa vários núcleos — e por que cada caso pede uma ferramenta
diferente.

---

## 1. Três problemas, três respostas

| Problema | Natureza | Ferramenta |
|---|---|---|
| Um vídeo, núcleos ociosos | estágios acoplados, estado compartilhado | **threads** (`core.parallel`) |
| Muitos vídeos, uma máquina | tarefas independentes | **processos** (`core.batch`) |
| Muitos vídeos, um cluster | tarefas independentes, memória distribuída | **MPI** (`core.batch`) |

MPI dentro de um vídeo adicionaria serialização de quadros de dezenas de MB;
threads entre vídeos disputariam o GIL na orquestração e não escalariam além de
um nó. MPI só compensa com workers em **máquinas diferentes**.

---

## 2. Dentro de um vídeo: etapa paralela, etapa serial

### O raciocínio

- **Preparar o quadro, detectar e ler um recorte são sem estado.** Reduzir e
  girar o quadro (`VideoSource.prepare`), detectar objetos e placas
  (`LPRPipeline.detect_stage`) e o OCR de uma placa (função só do quadro e da
  caixa) não dependem do quadro anterior.
- **Rastreamento, fusão, publicação e gravação são com estado e ordenados**
  (`LPRPipeline.sequential_stage`).

Daí o desenho: **preparar, detectar e (no perfil que lê tudo) ler em paralelo;
reordenar; processar em série**. A etapa sequencial é literalmente o mesmo
código do pipeline serial — é o que garante resultados idênticos.

```
            só decodifica      ┌──▶ worker 1: prepara, detecta, lê ──┐
   leitura ───────────▶ fila ├──▶ worker 2: prepara, detecta, lê ──┤──▶ reordenação ──▶ rastreio ──▶ fusão ──▶ banco
   (VideoToolbox)              └──▶ worker N: prepara, detecta, lê ──┘   (por sequência)   (serial: incorpora as leituras)
```

- **Preparo nos workers** (arquivo offline): a thread de leitura só decodifica
  (`raw_frames`); cada worker reduz e gira o próprio quadro, com a mesma função.
  No vídeo 4K vertical, reduzir 2160 → 1280 px (escala não inteira) e girar era
  84% da captura (6,5 de 7,8 ms): com tudo na leitura, o paralelo ficava **mais
  lento** que o serial.
- **OCR antecipado** (offline, com veículos, e só quando o agendamento lê toda
  placa em todo quadro — `ocr_every_n: 1`, `max_reads_published: 0`, o perfil de
  avaliação): cada worker lê, com motor próprio, as placas acima de
  `ocr.min_plate_width` que acabou de detectar; a etapa serial usa a leitura
  pronta da **mesma** detecção. Desperdício medido: 6–15%. No perfil padrão, o
  agendamento seletivo descartaria 7× o que aproveitasse, e nada é antecipado;
  no modo só-placa o recorte sai da caixa da trilha (estado) e também não.

### Uso

`detector.workers` no YAML (a interface, `lprts run` e `lprts eval` seguem o
perfil; `--parallel` sobrepõe):

| Valor | Efeito |
|---|---|
| `-1` (padrão) | automático: offline, 4 workers (metade dos núcleos, até 8, com OCR antecipado); em tempo real, serial |
| `0` | pipeline serial (perfis `avaliacao`, `tempo_real*`, `lote`) |
| `N` | N workers |

```bash
lprts run --source video.mp4 --headless              # automático
lprts eval --source v.mp4 --gt gt.csv --config configs/avaliacao.yaml --parallel -1
```

### Resultado medido (0.5)

Apple M5 Pro (15 núcleos, 24 GB), **na bateria**; vídeos reais do iPhone (4K
HEVC, 30 qps), objetos YOLO11n e placas YOLO no CoreML, `fast-plate-ocr` na CPU.
Vazão = quadros ÷ duração da sessão (sem a carga dos modelos). Em **todas** as
linhas, detecções, leituras (texto e confianças por caractere) e eventos são
idênticos, bit a bit, aos do serial.

| Perfil e vídeo | Serial 0.4 | Paralelo 0.4 | **0.5 (automático)** |
|---|---|---|---|
| Padrão, 27 s vertical | 50,9 q/s | 50,1 (3 workers) | **87,3** (4 workers) |
| Padrão, 92 s horizontal | 80,5 q/s | 121,5 (2) | **127,1** (4) |
| Avaliação, 27 s vertical | 21,5 q/s | 30,6 (2) | **63,4** (7, OCR antecipado) |
| Avaliação, 92 s horizontal | 23,2 q/s | — | **58,2** (7, OCR antecipado) |

Workers × vazão (perfil de avaliação, vídeo vertical): 2 → 32,9; 3 → 42,2;
4 → 50,5; 6 → 59,1; 8 → 62,6; 10 → 52,7 q/s. Perfil padrão: o vídeo horizontal
satura em 2–3 workers (126 q/s), limitado pelo Neural Engine, que os workers
compartilham (a detecção de um quadro vai de 8,4 ms sozinha a 16 ms com 2 em
paralelo); o vertical ganha até 4 (78/83/87 q/s com 2/3/4).

Em tempo real o automático é serial: nos dois vídeos a máquina acompanha os
30 q/s sem descarte (p50 12,7–16,8 ms, p99 18–21 ms), e com 2 workers a
latência não melhora.

### Determinismo

O paralelo só serve à avaliação se não muda o resultado; cada mudança acima foi
verificada contra o serial nos dois vídeos (hash das detecções, das leituras
com as confianças por caractere e dos eventos) e tem teste
(`TestEquivalencia`, `TestPreparoNosWorkers`).

O onnxruntime **não** dá o mesmo resultado com qualquer número de threads: com
1 ou 2 threads por sessão, as confianças do OCR diferem em ~3·10⁻⁷ das obtidas
com 4 ou mais (os textos coincidiram em 400 de 400 recortes). Por isso
`ocr.threads` é fixo (4) e igual no serial, no paralelo e em cada worker — e a
leitura deixa de depender dos núcleos da máquina, como acontecia com o padrão
da biblioteca. Neste Apple M5 Pro, 4 threads reproduzem o padrão bit a bit.

### Latência offline em paralelo

Em paralelo, a latência por quadro offline inclui a fila da reordenação e a
disputa pelos núcleos (p50 0,1–0,35 s nas linhas acima): não é o custo de
processar um quadro. O perfil de avaliação fica serial porque a condição "sem
prazo" reporta esse custo; para medir só a acurácia, `--parallel -1` dá o mesmo
resultado ~3× mais rápido, e o relatório avisa que a latência daquela sessão não
é comparável.

### Histórico: 0.2.0, baseline clássico

Medido na 0.2.0, vídeo sintético 1280×720 de 400 quadros, detector clássico +
OCR por gabaritos, offline: serial 198 q/s; 2 workers 277 (1,40×); 3 workers
242; 4 workers 266 — eventos idênticos.

### Reordenação sem buracos

Cada quadro recebe um número de sequência **no momento em que um worker o
retira da fila**, sob uma trava; todo quadro retirado produz uma saída, ou a
falha do worker interrompe a sessão com erro explícito (desde a 0.5; antes,
virava um quadro sem detecções). A sequência não tem buracos por construção.

> **Defeito corrigido na 0.2.** A versão anterior reordenava pelo índice do
> quadro na fonte. No modo tempo real, cada quadro descartado deixava um buraco
> que o buffer esperava até acumular 32 quadros: com detecção mais lenta que a
> fonte, a latência real chegou a ~1,6 s enquanto o banco registrava ~90 ms, e
> os descartes eram contados em dobro. Coberto por
> `test_reordenacao_nao_trava_com_descarte`.

### Política das filas

- **Entrada**: *drop-oldest* em tempo real (latência acima de completude);
  **bloqueante** em offline (perder um quadro invalidaria a avaliação).
- **Saída**: sempre **bloqueante**. Descartar depois da detecção jogaria fora
  trabalho pronto; sob carga, a contrapressão chega à fila de entrada — o lugar
  certo de descartar.

### Oversubscription

Cada worker tentaria usar todos os núcleos nas operações do OpenCV;
`tune_opencv_threads` reparte os núcleos entre os workers.

O onnxruntime fazia pior: com o padrão, cada sessão de OCR usava todos os
núcleos e **girava** esperando trabalho — uma leitura de 9,4 ms gastava 65 ms de
CPU, tirados da leitura do vídeo e da detecção. Com `ocr.threads: 4` e sem
espera ativa, 10,7 ms e 27,7 ms de CPU (CPU de usuário do processo: −19% no
perfil padrão, −61% no de avaliação, −36% em tempo real no vídeo horizontal).

O inverso também acontece. Ao ser importado, o ultralytics executa
`cv2.setNumThreads(0)` (pensado para o DataLoader de treino), e todo o OpenCV do
processo passava a rodar numa thread: no vídeo 4K vertical, redimensionar o quadro
custava 70 ms e o processamento caiu para 10 quadros/s. Desde a 0.3.1 o ultralytics
é importado por `lprts.runtime.import_ultralytics`, que restaura as threads (29,5
quadros/s no mesmo vídeo, com as mesmas placas).

### O compromisso

A etapa paralela aumenta a **vazão**, não reduz a **latência** de um quadro.
Numa câmera ao vivo, fique no serial enquanto a máquina acompanhar (descarte
zero); se descartar, meça `--parallel 2` antes de adotar. Para esvaziar um
arquivo, o automático.

---

## 3. Entre vídeos: lote

### Local (processos)

```bash
lprts batch "corpus/*.mp4" --backend process --workers 8 --config configs/lote.yaml
```

Cada worker carrega o próprio modelo; o `ProcessPoolExecutor` entrega a próxima
tarefa a quem terminar primeiro (vídeos têm durações muito diferentes).

Medido na 0.2.0 (mesma máquina, 6 cópias do vídeo acima, 2.400 quadros):

| Backend | Vazão agregada | Sessões | Quadros no consolidado |
|---|---|---|---|
| sequential | 194 q/s | 6 | 2.400 |
| **process (3)** | **461 q/s** (2,38×) | 6 | 2.400 |

### Cluster (MPI)

```bash
pip install mpi4py
mpirun -n 64 python -m lprts.core.batch --videos "corpus/*.mp4" \
       --backend mpi --config configs/lote.yaml --out runs/corpus
```

Mestre-trabalhador **dinâmico**: o rank 0 entrega uma tarefa por vez a quem
pedir. Com partição estática, um vídeo longo deixaria núcleos ociosos no fim.
O backend MPI foi testado na lógica de despacho e consolidação, **não** em
cluster real.

### Consolidação dos bancos

Cada execução grava numa pasta própria (`runs/lote/<data-hora>-<pid>/`), cada
worker num fragmento (`shard_<id>.db`), e ao final os fragmentos são
consolidados em `consolidado.db` na mesma pasta.

> **Dois detalhes críticos.** (1) Cada fragmento numera suas linhas a partir de
> 1: a junção copia só as colunas que não são a chave primária, senão as linhas
> colidiriam e seriam descartadas em silêncio
> (`test_nao_perde_linhas_com_ids_colidentes`). (2) Na 0.1 os fragmentos ficavam
> direto em `runs/lote/` e eram reaproveitados entre execuções (PID repetido,
> `shard_0.db` do modo sequencial): o consolidado misturava sessões antigas.

---

## 4. Otimização de hiperparâmetros

### Por que multiobjetivo

Baixar `ocr_every_n` aumenta a exatidão e derruba a vazão; subir `min_reads`
aumenta a precisão e reduz a revocação. Reduzir tudo a um número esconde o
compromisso atrás de um peso arbitrário. A resposta honesta é a **fronteira de
Pareto**.

### Reotimizar a fusão sem reprocessar vídeo

A tabela `reads` guarda cada tentativa de OCR, com confiança por caractere e a
caixa da placa. Varrer os parâmetros de fusão custa milissegundos por
configuração:

```bash
lprts tune --db runs/eval.db --gt anotacoes.csv --holdout 0.3 --trials 2000
lprts tune --db runs/eval.db --gt anotacoes.csv --grid
```

(`runs/eval.db` é o banco padrão do `lprts eval`, que o apaga a cada execução: ver
[AVALIACAO.md](AVALIACAO.md) §2.)

O que mudou na 0.2, e por quê:

- **Verdade por posição.** Cada trilha recebe a placa do veículo anotado que ela
  cobre (IoU das caixas no mesmo quadro). A 0.1 usava a placa anotada mais
  parecida com a própria leitura — o otimizador era recompensado por erros que
  por acaso lembravam outra placa.
- **Decisão online replicada.** As leituras são reapresentadas em ordem e a
  trilha publica como no pipeline — desde a 0.3, com a mesma `PublicationPolicy`
  (`fusion.publish_policy`, `fusion.exit_min_reads`).
- **Revocação sobre veículos anotados**, não sobre trilhas.
- **Validação em veículos reservados** (`--holdout`): a fronteira é escolhida
  numa parte dos veículos e reavaliada na outra.
- **Hipervolume com referência fixa** (pior valor possível de cada objetivo),
  comparável entre estudos; o ponto usado é gravado no estudo.
- **Aviso de censura**: se a sessão foi gravada com
  `ocr.max_reads_published > 0`, trilhas publicadas pararam de ser lidas e o
  `tune` avisa.

Os números de "+59% de F1" citados na 0.1 foram obtidos com a atribuição por
texto e sem validação: não os use.

### Backends

| Backend | Quando usar |
|---|---|
| `optuna` | se instalado; amostrador NSGA-II |
| `nsga2` | padrão; implementação própria |
| `random` | **linha de base honesta** — todo otimizador deve se provar contra ela |

### Espaços de busca

| Espaço | Parâmetros | Custo por tentativa |
|---|---|---|
| `fusion` | modo, min_reads, min_agreement, min_confidence, sintaxe, publish_policy, exit_min_reads | ~ms |
| `realtime` | imgsz, conf, detect_every_n, ocr_every_n, max_ocr_per_frame, min_plate_width, resize_width | uma passada de vídeo |
| `tracker` | IoU mínimos (veículo e placa), max_age, min_hits | uma passada de vídeo |

Só o espaço `fusion` está ligado ao `lprts tune`; os outros exigem reprocessar
o vídeo e devem ser conduzidos por script, avaliando cada tentativa com
`lprts eval` e um conjunto de validação separado do de teste.

---

## 5. Como escolher

```
Um vídeo, ao vivo, latência importa
    └─▶ lprts run --config configs/tempo_real.yaml        (serial; se descartar, meça --parallel 2)

Um vídeo, offline, quero terminar rápido
    └─▶ lprts run --offline --headless                    (paralelo automático, mesmo resultado)

Dados do artigo: acurácia e custo sem prazo
    └─▶ lprts eval --config configs/avaliacao.yaml        (serial: a latência mede o custo)
    └─▶ ... --parallel -1                                 (só a acurácia: a mesma, ~3x mais rápido)

Muitos vídeos, uma máquina
    └─▶ lprts batch "*.mp4" --backend process

Muitos vídeos, um cluster
    └─▶ mpirun -n N python -m lprts.core.batch --backend mpi

Já processei; quero os melhores parâmetros de fusão
    └─▶ lprts tune --db runs/x.db --gt gt.csv --holdout 0.3
```
