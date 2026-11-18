# Protocolo de avaliação

Como produzir números defensáveis com este sistema.

---

## 1. Anotar o ground truth

### Formato CSV

```csv
frame_index,x1,y1,x2,y2,plate_text,object_class,track_id,kind,difficult
120,880,540,1010,585,ABC1D23,carro,3,plate,0
120,700,380,1180,640,,carro,3,object,0
121,884,541,1016,587,ABC1D23,carro,3,plate,0
```

- Uma linha por **caixa** por quadro. `kind = plate` é a caixa da placa
  (`plate_text` obrigatório); `kind = object`, a do objeto (veículo, pedestre,
  bicicleta). Sem a coluna `kind`, linha com placa é `plate` e sem placa é `object`.
- **`track_id` identifica o objeto ao longo dos quadros** — a placa e o veículo
  que a porta têm o mesmo id. É o que permite medir **por veículo** e avaliar o
  rastreamento. Sem ele, o veículo é identificado pelo texto da placa (inválido
  se uma placa aparece em mais de um veículo; a avaliação avisa).
- Coordenadas em **pixels da imagem original**, `(x1, y1)` no canto superior
  esquerdo. Em vídeo de celular gravado na vertical (metadado de rotação no
  contêiner), a imagem original é a **girada**, como o player a exibe: anote nessa
  orientação (`sessions.source_width/height` são as dimensões dela).
- `plate_text` é normalizado na leitura (maiúsculas, sem hífen).
- `difficult = 1`: caixa que não conta nem como acerto nem como erro (placa
  ilegível a olho nu ou de texto incerto para o anotador, oclusão severa, cortada pela
  borda) — protocolo PASCAL VOC.
  Um veículo com todas as aparições difíceis sai da métrica por veículo.
  **Declare no artigo** quantas caixas e veículos foram marcados e o critério.

### Quanto anotar

| Dimensão | Mínimo | Recomendado |
|---|---|---|
| Veículos com placa | 50 | 200+ |
| Quadros anotados | 500 | 2.000+ |
| Condições distintas | 2 (dia/noite) | 4+ (dia, noite, chuva, contraluz) |
| Ângulos de câmera | 1 | 3+ |

A **unidade estatística é o veículo**: 200 veículos em três condições ensinam
mais que 2.000 quadros de 20 veículos. Com 50 veículos, o intervalo de confiança
de uma revocação de 80% vai de ~67% a ~89% — planeje o tamanho da amostra pela
largura de intervalo que a comparação exige.

### Ferramentas

Para vídeo real, anote no CVAT, instalado localmente. O passo a passo da filmagem e da
anotação está em [COLETA.md](COLETA.md). `scripts/cvat_para_gt.py` converte a exportação
"CVAT for video 1.1" neste formato: interpola os quadros entre os quadros-chave e recusa
trilha sem texto. Antes dele, `scripts/refinar_caixas.py` põe de volta em cima da placa as
caixas interpoladas, seguindo-a no vídeo entre os quadros-chave do anotador, sem usar o
detector do sistema. `scripts/make_synthetic_dataset.py` gera ground truth exato (com
`track_id`, caixas de objeto e `difficult`) para validar o pipeline de avaliação
antes de investir em anotação manual.

---

## 2. Executar

```bash
lprts eval --source video.mp4 --gt anotacoes.csv \
           --config configs/avaliacao.yaml --out runs/relatorio.json
```

O perfil `avaliacao.yaml` garante o essencial:

- `realtime: false` — nenhum quadro descartado, OCR síncrono, determinístico;
- `ocr_every_n: 1` e `max_reads_published: 0` — toda placa elegível é lida em
  todo quadro, **inclusive depois de publicada**: a amostra de leituras não
  depende do acerto do próprio motor;
- `max_reads_per_track: 0` — todas as leituras votam (a refusão offline reproduz
  o online);
- `resize_width: null` — resolução original;
- `auto_plate_fallback: false` — nada muda no meio do experimento;
- `plate_width_ratio: 0` — a placa é procurada em todo veículo, mesmo distante: mede
  a detecção de placas inteira (o padrão, 0,5, pula veículos estreitos demais para a
  placa ser lida);
- tudo gravado: `reads`, `detections`, `track_points`, `frames`;
- `detector.workers: 0` — pipeline serial: a latência por quadro "sem prazo" mede o
  custo de processar, sem fila;
- `ocr.threads: 4` — as leituras do OCR não dependem dos núcleos da máquina (com 1
  ou 2 threads o onnxruntime muda as confianças no último bit).

**Só a acurácia, mais rápido.** `--parallel -1` lê as placas nos workers do pipeline
paralelo e dá o **mesmo** resultado bit a bit (detecções, leituras e eventos,
verificados em vídeo real) ~3× mais rápido: 21,5 → 63,4 quadros/s no vídeo 4K
vertical, no Apple M5 Pro. A latência por quadro dessa sessão inclui a fila do
paralelo e não é comparável — o relatório avisa. Para as tabelas de tempo, use a
execução serial.

A sessão fica no banco (`--db`, padrão `runs/eval.db`; o `analytics.db_path` do
perfil não vale para o `lprts eval`) e o relatório é calculado **a partir do
banco**. Para reavaliar depois — com outro limiar, ou um ground truth corrigido —
sem reprocessar o vídeo:

```bash
lprts eval-db --db runs/eval.db --gt anotacoes_v2.csv --session <id> --out runs/r2.json
```

> **Cada `lprts eval` apaga o banco `--db`** e os recortes das sessões dele antes de
> começar (`analytics.clear_previous`, padrão desde a 0.4; as repetições de um mesmo
> `--repeat` ficam juntas). O relatório JSON e as tabelas CSV sobrevivem, mas a
> sessão da execução anterior não: para reavaliá-la depois (`eval-db`, `tune`), dê a
> cada execução o seu banco (`--db runs/<nome>.db`) ou acumule com `--keep-previous`.

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `relatorio.json` | Todas as métricas, com intervalos de confiança, configuração, ambiente e componentes efetivos. |
| `relatorio_veiculos.csv` | Uma linha por veículo anotado (desfecho e covariáveis). |
| `relatorio_leituras.csv` | Uma linha por tentativa de OCR, rotulada (acerto, CER) e com covariáveis. |
| `relatorio_eventos.csv` | Uma linha por evento publicado, rotulado. |

---

## 3. Métricas

### Pareamento: sempre por posição

Leituras, trilhas e eventos são associados ao veículo anotado pela
**sobreposição espacial no mesmo quadro** (IoU entre a caixa da placa do sistema
e a anotada; 0,5 para leituras e detecções, 0,3 para associar trilhas). Nunca
pelo texto mais parecido: isso compara uma leitura errada com a placa de outro
veículo que ela por acaso lembre, e o viés cresce com o número de placas do
corpus.

### Escopo: largura mínima da placa

Num vídeo de rua aparecem placas longe demais para qualquer leitura: a pista oposta, os
carros estacionados ao fundo. Sem anotação, tudo o que o sistema fizer com elas conta
como erro; anotá-las uma a uma custa horas. `--scope-min-width N` (em `lprts eval` e
`lprts eval-db`) restringe a avaliação às placas com pelo menos N pixels de largura na
imagem original, como o tamanho mínimo de objeto do KITTI:

- a caixa **anotada** mais estreita que N conta como `difficult` naquele quadro (um
  veículo que nunca chega a N sai da métrica por veículo);
- a saída do sistema mais estreita que N **que não cai sobre nenhuma placa anotada** é
  ignorada: detecção, leitura, caixa de trilha e o evento de uma trilha sem veículo
  anotado cuja placa tem largura mediana abaixo de N;
- placa de N pixels ou mais sem anotação **continua erro**. O gabarito ainda tem de
  estar completo para o que dá para ler.

```bash
lprts eval-db --db runs/eval.db --gt gt.csv --scope-min-width 50 --out runs/escopo50.json
```

Como escolher N: pelo histograma de largura das detecções sem anotação (as placas de
fundo formam um pico, e depois dele a contagem despenca) e pela menor placa que o
anotador leu com segurança. Escolha **antes** de olhar as métricas, use o mesmo N em
todos os vídeos de mesma resolução e declare-o no artigo. A opção não altera a sessão
gravada: dá para reavaliar com outro N e reportar a sensibilidade.

No piloto de 2026-11 (4K, avenida com a pista oposta no quadro, 26 veículos), 93% das
detecções sem anotação tinham menos de 50 px. Com N = 50 os 26 veículos continuaram na
métrica, os eventos sem veículo anotado foram de 53 para 5 (os 5 eram veículos que
faltava anotar) e a precisão dos eventos, de 0,28 para 0,68.

**No artigo, N = 50 px** (vídeos 4K), confirmado pelo autor em 2026-11-15.

O `lprts tune` ainda não aplica o escopo: lá, trilha sem veículo anotado segue contando
como erro.

### Por veículo — a métrica principal

| Métrica | Definição |
|---|---|
| **Revocação** | veículos com a placa correta na decisão **vigente** ÷ veículos anotados (os não lidos contam) |
| Revocação (legíveis) | idem, só veículos cuja placa atinge a largura mínima de OCR em algum quadro |
| **Precisão** | decisões vigentes corretas ÷ decisões vigentes (sem veículo anotado = erro) |
| F1 | média harmônica |
| Duplicatas | decisões vigentes além da primeira no mesmo veículo (fragmentação de trilha; revisão não é duplicata) |
| Latência de decisão | tempo de vídeo do primeiro avistamento à primeira publicação correta (inclui a revisão que corrigiu) |
| Até decidir, no relógio | relógio do host, da primeira imagem da trilha disponível à mesma decisão (`events.decision_elapsed_ms`, esquema 4) |

A **latência de decisão** é a medida do artigo: não depende da velocidade da
máquina, inclui o atraso de detecção e se compara entre o modo sem prazo e o
tempo real. O **relógio do host** só é latência em tempo real (a fonte chega no
ritmo da câmera); offline ele mede a velocidade do processamento, e por isso o
relatório em texto só o mostra em sessões de tempo real (o JSON e os CSVs trazem
sempre). Bancos anteriores ao esquema 4 não têm o relógio: esses veículos ficam
fora dele, com aviso, e nunca entram como tempo zero.

Não há "revocação dentro do prazo": o prazo da sessão (`sessions.deadline_ms`) é
de **um quadro**, e uma decisão exige `fusion.min_reads` leituras — compará-los
daria 100% offline e 0% em tempo real por construção. Para um requisito de
decisão do seu caso (ex.: placa em até 2 s), filtre `time_to_correct_s` em
`relatorio_veiculos.csv`.

Com revisões ou decisões relaxadas na saída (`fusion.publish_policy: revise`,
`fusion.exit_min_reads > 0`), o relatório traz também o que um consumidor que agiu
na **primeira** decisão recebeu: revocação da 1ª decisão (`first_decision_recall`)
e precisão de **tudo o que foi emitido**, inclusive as decisões depois revisadas
(`precision_emitted`). Reporte as duas leituras quando a política não for a padrão.

Em LPR, uma placa com um caractere errado é **inútil**: não consulta banco nem
identifica veículo. A métrica de uso é a placa inteira correta, por veículo.

### Efeito da fusão multi-frame

Medido nos **mesmos veículos** (os que receberam ao menos uma leitura com texto),
sobre a trilha que mais os leu:

| Linha | Pergunta |
|---|---|
| Uma leitura ao acaso | exatidão esperada de uma única tentativa |
| Leitura mais confiante | e se escolhermos a de maior confiança? |
| Fusão de todas as leituras | o que a votação produz com tudo o que foi lido |
| Publicado (decisão online) | o que o sistema efetivamente entregou |

A diferença fusão − melhor leitura vem com intervalo de confiança e teste de
McNemar exato (pares discordantes). A diferença entre **fusão** e **publicado**
mede o custo da política de publicação. Ela tem duas origens, que pedem remédios
diferentes:

* **decisão errada cedo** — a trilha publicou nas primeiras leituras e depois leu
  outra coisa (tipicamente começou na placa de outro veículo). Remédio:
  `publish_policy: revise`;
* **nenhuma decisão** — a trilha terminou com menos de `min_reads` leituras
  (veículo rápido, poucos quadros legíveis), embora a fusão estivesse certa.
  Remédio: `exit_min_reads`, que troca precisão por revocação.

Para saber qual domina, cruze `fused_correct = 1`, `dominant_published_correct = 0`
e `published` em `_veiculos.csv`. Para medir o efeito de cada política sem
reprocessar o vídeo, use `lprts tune` (ela replica a política exata do pipeline);
para os números finais, rode `lprts eval` com cada uma.

Medido na 0.3.0 no vídeo sintético pareado (60 veículos, `tempo_real_rapido.yaml` com
detector clássico e `fast-plate-ocr`, offline; comparação por veículo com a
política padrão, somando 1×–4× e obturador 0 e 4 ms):

| Variante | Veículos ganhos / perdidos | McNemar |
|---|---|---|
| `exit_min_reads: 1` | +38 / −0 | p < 0,001 |
| `exit_min_reads: 2` | +11 / −0 | p < 0,001 |
| `publish_policy: revise` | +7 / −10 | p = 0,63 |
| `publish_policy: on_exit` | +7 / −10 | p = 0,63 |

Revocação por veículo (decisão vigente), padrão → `exit_min_reads: 1`:

| Velocidade | 1× | 2× | 3× | 4× |
|---|---|---|---|---|
| obturador instantâneo | 0,83 → 0,83 | 0,82 → 0,82 | 0,47 → 0,57 | 0,15 → 0,37 |
| obturador de 4 ms | 0,85 → 0,85 | 0,82 → 0,83 | 0,58 → 0,68 | 0,28 → 0,48 |

Precisão nas mesmas condições: 0,80–0,89 com `exit_min_reads: 1` contra 0,80–0,86
no padrão (as decisões na saída, com uma leitura, passaram pelos mesmos filtros de
sintaxe, concordância e confiança).

Em velocidade alta a lacuna é quase toda de veículos **nunca publicados**:
`exit_min_reads: 1` a fecha em 3× (publicado = fusão) e a reduz de 0,40 para 0,10
em 4×, sem perda de precisão. A revisão corrige trilhas que começaram na placa de
outro veículo, mas também segue erros **sistemáticos** do OCR — no caso medido, o
Q das primeiras leituras (placa distante) virou O quando a placa se aproximou, com
59 leituras e concordância 0,93 —, e o saldo é nulo. Votar mais não corrige um
viés do motor; uma revisão útil precisaria ponderar a qualidade da leitura.

> **Observação em aberto.** Em 3× e 4×, o vídeo com obturador de 4 ms teve mais
> veículos rastreados que o de obturador instantâneo (51 × 43 de 60 em 3×), com o
> mesmo número de detecções de placa e caixas do mesmo tamanho; a causa não foi
> isolada. Compare **políticas** no mesmo vídeo (como acima) com confiança;
> compare **obturadores** neste vídeo sintético com cautela.

### Por tentativa de OCR

Cada tentativa, pareada por posição com a placa anotada do quadro. **Tentativas
sem texto entram no denominador** (taxa de rejeição reportada à parte). O
intervalo é por bootstrap **agrupado por veículo**: leituras do mesmo veículo
não são independentes. Inclui matriz de confusão e erros por posição.

### Detecção (por quadro)

Protocolo PASCAL VOC/COCO sobre os quadros que passaram pelo detector:
precisão, revocação, F1, AP (101 pontos), mAP@[.5:.95], IoU médio. Caixas
`difficult` ignoradas. Zero detecção com o detector ligado é erro medido
(revocação 0), não seção omitida.

A detecção de **objetos** só é avaliada quando o detector de objetos rodou a
sessão inteira (`detector.detect_vehicles` ligado e sem o fallback automático para
só-placa) **e** o ground truth anota objetos (`kind=object`). Fora disso o
relatório avisa e omite a seção: com o detector desligado não há o que medir, e
um objeto não anotado não é falso positivo.

### Rastreamento

MOTA (CLEAR-MOT), IDF1 e trocas de identidade, sobre as caixas de placa. A
fragmentação (trilhas por veículo) aparece também na métrica por veículo.

### Desempenho

Vazão (quadros ÷ tempo), latência **ponta a ponta** p50/p95/p99/máx, tempo de
serviço, fila, descartes e taxa de prazo perdido. Reporte **p95/p99, não a
média**: a média esconde os picos que quebram o tempo real.

---

## 4. Ablações

Toda configuração é gravada na sessão, e cada relatório traz os componentes
**efetivos** (runtime, dispositivo, hash dos pesos). Pedir um motor ausente é
erro — um relatório nunca é rotulado com um motor que não rodou.

### Motores de OCR e comparação pareada

```bash
for motor in template easyocr fastplate; do
  lprts eval --source video.mp4 --gt gt.csv --config configs/avaliacao.yaml \
             --ocr $motor --db runs/ocr_$motor.db --out runs/ocr_$motor.json
done
lprts compare runs/ocr_template.json runs/ocr_fastplate.json
```

`lprts compare` pareia os relatórios **por veículo** e devolve a diferença de
revocação com intervalo de confiança e o p-valor de McNemar exato. Comparar duas
proporções como se fossem amostras independentes desperdiça o pareamento e
superestima a incerteza. Um banco por execução (`--db`) mantém cada sessão
disponível para `eval-db` e `tune` (ver §2).

### Fusão (offline, sem reprocessar o vídeo)

```bash
lprts tune --db runs/eval.db --gt gt.csv --holdout 0.3 --trials 2000
```

- A verdade de cada trilha vem da **posição** das leituras.
- A decisão online é **replicada** com a mesma `PublicationPolicy` do pipeline
  (`fusion.publish_policy` e `fusion.exit_min_reads`, que também fazem parte do
  espaço de busca). A varredura `--grid` cobre só modo, `min_reads` e
  `min_agreement`, com a política padrão (`first_stable`).
- A revocação é sobre **veículos anotados**.
- `--holdout 0.3` sorteia 30% dos **veículos** (nunca leituras) para validação:
  a fronteira é escolhida nos outros 70% e reavaliada nos reservados. Ajustar e
  reportar no mesmo conjunto superestima o desempenho.
- O hipervolume usa ponto de referência fixo, comparável entre estudos.

Limite: a refusão só vê as leituras que o sistema fez. Grave com
`max_reads_published: 0` (o perfil de avaliação já faz isso); o `tune` avisa se a
sessão foi censurada.

### Degradação controlada (vídeo sintético)

```bash
for desfoque in 0 3 5 7 9 11; do
  python scripts/make_synthetic_dataset.py --out /tmp/v$desfoque.mp4 \
      --frames 400 --blur $desfoque --seed 1
  lprts eval --source /tmp/v$desfoque.mp4 --gt /tmp/v${desfoque}_gt.csv \
      --config configs/avaliacao.yaml --db runs/desfoque_$desfoque.db \
      --out runs/desfoque_$desfoque.json
done
```

O mesmo vale para `--noise` e `--brightness`. Os parâmetros da geração ficam em
`<vídeo>_meta.json`.

### Velocidade e borrão de movimento (vídeo sintético, desenho pareado)

```bash
python scripts/curva_velocidade.py --out runs/curva_velocidade \
    --speeds 1 1.5 2 3 4 --shutters 0 4 16.7 --total-vehicles 60 \
    --modes offline realtime --repeat 3 \
    --config configs/tempo_real_rapido.yaml \
    -- --no-vehicle-detector --plate-detector classic --ocr fastplate
```

- `--speed S` multiplica a velocidade dos veículos; `--shutter-ms E` faz de cada
  quadro a média de subquadros ao longo de E ms de exposição, centrados no
  instante anotado — o borrão cresce com a velocidade aparente, como numa câmera
  real (1 ms ≈ câmera de LPR dedicada; 4 ms = 1/250 s; 16,7 ms = 1/60 s com pouca
  luz).
- **Pareado por veículo**: cada veículo tem gerador aleatório próprio, e
  `--total-vehicles N` encerra o vídeo quando o N-ésimo sai de cena. Os mesmos N
  veículos (placa, cor, faixa, ordem e configurações relativas) aparecem em toda
  condição; só o tempo em cena e o borrão mudam. Compare condições com testes
  pareados (McNemar, `lprts compare`) e modele com o veículo como efeito aleatório.
- Saídas: `resumo.csv` (uma linha por condição × modo × repetição, com IC),
  `veiculos.csv` (uma linha por veículo × condição, com as covariáveis da geração:
  velocidade aparente da placa em px/quadro, extensão do borrão em px, quadros
  legíveis), `experimento.json` (comando, commit, ambiente) e `curva.png`. Cada
  condição × modo tem o próprio banco (`avaliacoes/*.db`).
- `--variant nome=secao.chave:valor,...` (repetível) roda a mesma grade com
  sobreposições da configuração, nos mesmos vídeos — ex.: `--variant padrao=
  --variant saida1=fusion.exit_min_reads:1` compara duas políticas veículo a
  veículo.

Exemplo de modelo: `glmer(fused_correct ~ log(plate_px_per_frame) + motion_blur_px
+ (1 | vehicle), family = binomial, data = veiculos)`. A velocidade aparente e o
borrão são as variáveis físicas; o fator `speed` só identifica a condição.

A curva sintética isola o efeito da velocidade com tudo o mais fixo; ela **não**
estima a acurácia em rodovia real (placas sem sujeira, reflexo ou compressão).

---

## 5. Avaliação sob restrição de tempo real

O perfil de avaliação mede a acurácia **sem** restrição de tempo: todo quadro é
processado, o OCR é síncrono. Isso **não** é a acurácia que o sistema entrega ao
vivo, onde quadros são descartados e o OCR chega atrasado. Para medi-la:

```bash
lprts eval --source video.mp4 --gt gt.csv --config configs/tempo_real_rapido.yaml \
           --realtime --repeat 5 --db runs/tempo_real.db --out runs/tempo_real.json
```

O vídeo é lido na cadência nominal (simulando a câmera), e o resultado depende
da carga da máquina — por isso **repita** (`--repeat`): o relatório agregado traz
média, desvio, mínimo e máximo de cada métrica entre as repetições, e cada
repetição tem seu relatório completo (`_rep1.json`, ...).

Reporte as duas condições, lado a lado: acurácia offline (capacidade do método)
e acurácia em tempo real (o que o sistema entrega), com a latência p95/p99 e a
taxa de descarte da segunda.

Para isolar o efeito da restrição de tempo real, rode o **mesmo** perfil nos dois
modos (`--offline` e `--realtime`) e compare por veículo com `lprts compare`: entre
`avaliacao.yaml` e um perfil de tempo real mudam também a resolução, os limiares e o
orçamento de OCR. Nos perfis com `ocr.max_reads_published > 0` (`tempo_real.yaml`),
a trilha publicada para de ser lida: a decisão publicada não muda (com
`first_stable`), mas a exatidão por tentativa e a fusão final ficam censuradas.

### Classificação do sistema

O LPR-RTS é um sistema de **tempo real brando** (*soft real-time*), de melhor
esforço: não há garantia de prazo (o macOS não é um sistema operacional de
tempo real e o custo por quadro varia com a cena), mas o sistema descarta
quadros a montante em vez de acumular atraso, e **mede** latência, descartes e
prazos perdidos por quadro. Afirme "tempo real brando" com os números medidos —
vazão ≥ taxa da fonte, latência ponta a ponta p95/p99, taxa de descarte e de
prazo perdido para um prazo declarado — e nunca "tempo real" sem qualificação.

---

## 6. Inferência e modelagem

- **Intervalos de confiança**: todo número do relatório por veículo e por
  tentativa vem com IC de 95% por bootstrap (semente fixa, reproduzível).
- **Comparações**: pareadas por veículo (`lprts compare`), McNemar exato.
- **Modelagem** (ex.: probabilidade de leitura correta em função da largura da
  placa, nitidez, iluminação): use `relatorio_leituras.csv` com **efeitos
  aleatórios por veículo** (`gt_vehicle`) — regressão logística mista, por
  exemplo `glmer(correct ~ log(plate_w) + sharpness + (1 | gt_vehicle))` — ou
  erros-padrão agrupados por veículo. Tratar leituras como independentes
  produz intervalos estreitos demais.
- **Confiança do motor como covariável**: é específica de cada motor e não
  calibrada. Ajuste por motor (ou calibre antes, com diagrama de confiabilidade).
- **Múltiplas comparações**: ao comparar muitas configurações, corrija (Holm)
  ou separe exploração (tune) e confirmação (conjunto reservado).

---

## 7. Ameaças à validade

| Ameaça | Como mitigar |
|---|---|
| **Vídeo sintético** não representa condições reais | Use só para validar a implementação e para ablações controladas; reporte em separado. A fonte das placas (DejaVu) não está no banco do OCR por gabaritos, mas também não é a fonte oficial. |
| **Uma câmera/ângulo** limita a generalização | Avalie em ≥ 3 posicionamentos. |
| **Vazamento de dados** com detector treinado no mesmo vídeo | Separe treino/validação/teste **por vídeo** (ou por local), nunca por quadro. |
| **Anotação enviesada** | Anotação cega ou dupla, com concordância entre anotadores. |
| **Exclusões (`difficult`)** inflam a métrica | Declare quantas e o critério; reporte também sem excluir. |
| **Escopo por largura** escolhido depois de ver o resultado | Fixe N antes, pelo histograma de larguras; reporte a sensibilidade a N. |
| **Hiperparâmetros ajustados no teste** | `lprts tune --holdout`, ou um conjunto de validação separado. |
| **Acurácia offline reportada como tempo real** | Seção 5: avalie também com `--realtime --repeat`. |
| **Desempenho dependente da máquina** | Reporte chip, energia (tomada/bateria), versões e commit — tudo em `environment_json`. Feche outros programas; repita as medidas. |

> A separação treino/teste **por vídeo** é o erro mais fácil de cometer em LPR:
> quadros consecutivos do mesmo veículo são quase idênticos. Dividir por quadro
> faz o mesmo veículo aparecer no treino e no teste, e a métrica resultante é ficção.

---

## 8. Reprodutibilidade

Cada sessão grava a configuração completa, o hash da configuração, o **commit do
código** (com `+dirty` se havia alterações não commitadas), as versões de todos
os pacotes, o hardware e o estado de energia.

```bash
make lock                                         # trava as versoes em requirements.lock
lprts sessions --db runs/eval.db                  # localizar a sessao (com o commit)
lprts export --db runs/eval.db --session <id> --out runs/export   # tabelas + sessao.json
lprts report --db runs/eval.db --out runs/relatorio               # tabelas e figuras (DADOS.md §4)
```

As tabelas do `lprts report` não têm textos de placa e podem acompanhar o artigo;
`sessions.device` distingue as sessões por runtime (ver o
[dicionário de dados](DICIONARIO_DE_DADOS.md)).

Gere resultados para publicação a partir de uma árvore **sem** alterações não
commitadas e versione `requirements.lock` e os relatórios JSON junto do artigo.
