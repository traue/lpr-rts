# Histórico de mudanças

## 0.5.0 — auditoria de publicação: dados à prova de falha, métricas corrigidas e paralelismo (2026-10)

**Esquema 4.** Bancos dos esquemas 2 e 3 são migrados no lugar, numa transação; a 0.4
não abre um banco migrado, e perfis da 0.5 (chaves novas) não carregam na 0.4. Esta
versão reúne a auditoria de publicação feita sobre a 0.4.0 — robustez e integridade dos
dados — e a revisão dela, que corrigiu métricas e o rastreador e acrescentou o
paralelismo. Medido em dois vídeos reais (iPhone, 4K HEVC: 27 s vertical e 92 s
horizontal), Apple M5 Pro na bateria, e no vídeo sintético versionado.

### Dados e integridade

- **Falha de gravação é falha da sessão.** Sessões, eventos, trilhas, leituras e
  cruzamentos que não são gravados (erro do SQLite, ou fila cheia por mais de 100 ms)
  interrompem a sessão com `StorageWriteError` e `end_reason = erro_persistencia`. Antes, a
  linha era contada como perdida depois de 10 s e a sessão terminava como normal. A
  telemetria (quadros, detecções, pontos de trilha) continua descartável e contada.
  `flush` confirma o lote gravado e propaga o erro da escritora.
- **Migração atômica de verdade.** O `ALTER TABLE` rodava fora da transação: uma falha no
  meio deixava o banco meio migrado. `events.decision_elapsed_ms` (esquema 4) fica nulo
  nas decisões antigas.
- O banco ganha `application_id`; abrir ou limpar recusa o banco de outro programa
  mesmo que ele tenha uma tabela `sessions` (antes bastava o nome da tabela).
- **Nada é apagado antes de tudo carregar.** Fonte, calibração e modelos sobem antes de
  `clear_previous` apagar o banco anterior (CLI e interface): um modelo ausente não custa
  mais os dados de antes.
- **Pesos achados de qualquer pasta.** Os perfis usam `data/models/...` relativo; aberto
  de outra pasta, o app caía, só com um aviso, no modo só-placa e no detector clássico —
  outro pipeline, outro resultado. Agora o caminho também é procurado na raiz do
  projeto, e a sessão grava o caminho efetivo.
- O hash dos pesos não fica em cache: trocar o arquivo no mesmo caminho atualiza a
  proveniência (os metadados do CoreML também são relidos).

### Falhas que pareciam resultado

| Antes | Agora |
|---|---|
| Falha do detector num worker paralelo virava um quadro **sem detecções** | a sessão falha com erro explícito |
| Stream que esgotava as reconexões terminava como `fim_da_fonte` | falha da sessão |
| Falha da thread de captura ou do alimentador paralelo podia travar ou sumir | chega ao consumidor |
| Stream travado esperava sem limite | `source.open_timeout_ms`/`read_timeout_ms` (5 s), nativos do FFmpeg |
| OCR em tempo real sem teto | `ocr.max_pending_jobs` (32) e `ocr.job_timeout_s` (5 s); contadores `ocr_backpressure`, `ocr_expired` |
| A janela fechava com a thread do pipeline viva (perda de dados; no macOS, queda) | recusa fechar; o encerramento fecha cada recurso mesmo se outro falha |
| `lprts demo`/`bench` dependiam de `scripts/` (falhavam fora do checkout) | vêm no wheel (`lprts.synthetic`, `lprts.benchmark`), com o gerador idêntico pixel a pixel |
| Python 3.10, o mínimo declarado, não importava (`datetime.UTC`) | importa |

### Avaliação

| Mudança | Efeito medido |
|---|---|
| A latência de decisão (`decision_delay_*`, `time_to_correct_s`) segue em **tempo de vídeo**, do primeiro quadro anotado; o relógio do host vai para campos próprios (`decision_elapsed_*`, `elapsed_to_correct_s`, da primeira imagem da trilha), que o texto só mostra em tempo real | Sintético, offline: 2,43 / 2,86 s, igual à 0.4.0. A auditoria tinha trocado o significado sem trocar o nome: 0,02 s, a velocidade da máquina |
| Sai a "revocação dentro do prazo" da auditoria | Comparava uma decisão de `min_reads` leituras com o prazo de **um** quadro: 0,444 offline e 0,0 em tempo real, por construção |
| Zero detecção com o detector ligado é erro medido | Antes a seção sumia, com o aviso falso "store_detections desligado" |
| Detecção de objetos só é avaliada com o detector ligado a sessão inteira e objetos no ground truth | Com `--no-vehicle-detector`, a auditoria mostrava revocação 0 de um detector desligado |
| MOTA: maximiza os pares acima do limiar e depois o IoU (o critério do py-motmetrics) | Sintético: MOTA e IDF1 iguais |
| `lprts tune` tira veículos inteiramente `difficult` dos dois lados da revocação | — |
| Placas únicas na execução descontam as revisões; `mAP` sai na seção de placas | — |

O relógio da decisão inclui o OCR que termina depois do fim do vídeo; não mede
exposição, rede da câmera, gravação nem entrega a um consumidor externo.

### Rastreamento

| Mudança | Efeito medido |
|---|---|
| Associação de **peso máximo com limiar**, a do ByteTrack de referência (`lapjv` com `cost_limit`). A auditoria maximizava o número de pares — o critério do MOTA —, que troca um par forte por dois fracos | IMG_0834 (perfil padrão): some a trilha de carro que seguiu um pedestre por 118 quadros na auditoria. Trilhas 244 (0.4.0) / 240 (auditoria) / 246; eventos 43 / 44 / 44; as mesmas 40 placas. IMG_0833 e sintético: idênticos nas três |
| Trilha acima de `max_age` sai **antes** de associar (com `detect_every_n` > 1 ela era associável até `max_age + n`); o relógio da fonte voltando zera as identidades; leitura pronta no quadro em que a trilha termina entra nela | Sem efeito nos dois vídeos (`detect_every_n` 1) |
| Húngaro com o laço por coluna vetorizado, sem pares inadmissíveis na conta | Mesmo resultado bit a bit (6000 matrizes, com empates); 40×80 de 1,7 para 0,35 ms |

### OCR

| Mudança | Efeito medido |
|---|---|
| Gabaritos respeitam `max_grammar_corrections` | Sintético: 59 leituras gravavam 3 correções com limite 2. Exatidão por leitura 0,294 → 0,253, CER 0,146 → 0,185; os mesmos 9 eventos |
| `auto` passa a cada motor só as opções dele | O `auto` nunca escolhia EasyOCR nem Paddle (recebiam `model_name` e falhavam), e um fallback descartava o `model_name` configurado sem avisar |
| `ocr.threads` (4) por sessão do onnxruntime, sem espera ativa | Leituras **idênticas bit a bit** às da 0.4 nos dois vídeos (8430 no de 92 s); com 1–2 threads as confianças mudariam ~3·10⁻⁷, e com o padrão da biblioteca dependiam dos núcleos da máquina. Cada leitura: 9,4 → 10,7 ms, mas 65 → 27,7 ms de CPU |

### Paralelismo

Leituras, detecções e eventos **idênticos bit a bit ao serial** em todas as linhas. Vazão
= quadros ÷ duração da sessão.

| Perfil, vídeo | 0.4 serial | 0.4 paralelo | 0.5 |
|---|---|---|---|
| padrão, 27 s vertical | 50,9 q/s | 50,1 (3 workers) | **87,3** (automático: 4) |
| padrão, 92 s horizontal | 80,5 | 121,5 (2) | **127,1** (4) |
| avaliação, 27 s vertical | 21,5 | 30,6 (2) | **63,4** (`--parallel -1`: 7, OCR antecipado) |
| avaliação, 92 s horizontal | 23,2 | — | **58,2** (7) |
| tempo real, os dois | p50 13–17 ms, p99 ≈ 20 ms, 0 descarte | — | igual, serial; CPU −19% (vertical) e −36% (horizontal) |

- **OCR antecipado nos workers.** O OCR de um recorte só depende do quadro e da caixa: no
  perfil que lê toda placa em todo quadro (o de avaliação, onde o OCR era 73% do tempo),
  cada worker lê o que detectou, com motor próprio, e a etapa serial incorpora a leitura
  da mesma detecção. Desperdício 6–15%. Fora desse perfil a antecipação jogaria fora 7× o
  que aproveitasse e fica desligada (também no só-placa e em tempo real).
- **Preparo do quadro nos workers.** No vídeo vertical, reduzir 2160 → 1280 px e girar
  era 84% da captura (6,5 de 7,8 ms); com tudo na thread de leitura, o paralelo ficava
  **mais lento** que o serial. Offline a leitura só decodifica.
- **Paralelo por padrão offline** (`detector.workers: -1`): 4 workers, ou metade dos
  núcleos (até 8) com OCR antecipado; **serial em tempo real**, onde a máquina já
  acompanha a câmera. A interface e `lprts run`/`eval` seguem o perfil; `--parallel`
  sobrepõe. A sessão grava `pipeline_workers` e `ocr_prefetched`.
- **O perfil de avaliação fica serial**: em paralelo a latência por quadro offline inclui
  a fila (p50 0,1–0,35 s) e não mede o custo de processar, que a condição "sem prazo"
  reporta. `--parallel -1` dá a mesma acurácia ~3× mais rápido; o relatório avisa que a
  latência dessa sessão não é comparável. A `lprts demo` também é serial (400 quadros;
  o desempenho que ela mostra é o custo por quadro).

### Modelos e scripts

- Downloads, exportações e treino terminam com código ≠ 0 quando o artefato pedido não
  sai, e preparam o novo à parte antes de substituir o que funcionava (exceção
  documentada: com `--force`, um peso baixado substitui o anterior mesmo se a exportação
  seguinte falhar).
- Entrada ONNX validada na partida (forma `[1, 3, H, W]`, `float`/`float16`).

### Verificação

615 testes (nenhum pulado com os extras neurais), ruff e mypy (66 módulos, também com o
extra `ocr`); wheel construído, com `demo` e `bench` executados fora do checkout. O CI
(`.github/workflows/checks.yml`: Linux e macOS, Python 3.10 e 3.12) foi adicionado e
ainda não rodou no GitHub. Ambientes registrados em `requirements/`: o da medição
(Python 3.11, com os extras neurais) e um de UI/análise/dev (Python 3.12).

### Comparabilidade dos dados

- Não misture números da 0.4 e da 0.5 na exatidão por leitura do OCR por gabaritos nem no
  rastreamento de cenas densas. Detecções e leituras do `fast-plate-ocr` não mudaram.
- A latência de decisão manteve o significado da 0.4.

### Encontrado e não corrigido

- **Caixa aninhada** (já na 0.4.0): no vídeo de 92 s, quadros 1798–1858, um carro dentro
  de outra caixa de carro, numa cena com 12 objetos; os três critérios de associação
  (0.4.0, auditoria, 0.5) erram ali de formas diferentes, cada um com 3–4 duplicatas no
  vídeo. Escolher entre eles exige ground truth do vídeo (MOTA/IDF1); a 0.5 fica com o
  critério do ByteTrack, que o artigo cita.
- Uma chamada nativa travada (inferência, leitura de câmera) não pode ser encerrada com
  segurança por uma thread Python: o sistema limita a espera, reporta e não libera o
  modelo em uso; recuperação automática exige supervisão de processo.

## 0.4.0 — tempo real mais leve, leitura no quadro original e dados para o artigo (2026-10)

O esquema do banco não muda. Medido em dois vídeos reais de rua (iPhone, 4K HEVC,
27 s e 92 s, Apple M4 Pro na tomada), em modo offline e comparando placa a placa
com a configuração da 0.3.1.

### Desempenho

| Mudança | Efeito medido |
|---|---|
| YOLO `.mlpackage` pelo `coremltools`, com letterbox, decodificação e NMS próprios (`detector.runtime: auto`) | Caixas idênticas às do ultralytics em 1923 de 1923 objetos e 911 de 912 placas (IoU 1,000); objetos 12,3 → 5,5 ms por quadro, placa 3,65 → 1,56 ms por veículo. O ultralytics gastava 3–4× a inferência em preparo; o `fromarray` de uma visão `[:, :, ::-1]` sozinho custava 3,3 ms |
| Placa só é procurada em veículo largo o bastante para ela ser legível (`detector.plate_width_ratio: 0,5`) | No vídeo de rua, 35% das buscas eram em veículos distantes; mesmas placas publicadas |
| Detecção por quadro (as duas mudanças) | p50 26–30 → 12–13 ms, p95 36–41 → 20 ms, as mesmas 10 + 32 placas |
| Vídeo de celular na vertical: o quadro cru é reduzido e só a imagem pequena é girada; o recorte do OCR converte as coordenadas e gira só o recorte (`crop_rotated`) | Captura 14,4 → 9,9 ms com a máquina livre; antes o quadro 4K inteiro era girado (5,6 ms) |
| Arquivos decodificados pelo AVFoundation (VideoToolbox, em hardware) no macOS (`source.backend: auto`) | Um quadro 4K HEVC com a redução para 1280 px: 46–48 → 8 ms. Mesmos quadros, rotação e tempos de mídia; diferença média de pixel ~1,5/255 (conversão de cor). Era o gargalo do tempo real: acima do prazo de 33 ms, a captura não acompanhava o vídeo. A busca (`start_frame`) reabre pelo FFmpeg, que acerta o quadro em HEVC |

**Tempo real** (perfil padrão com `--realtime`, versões intercaladas sob carga de serviços
do macOS): no vídeo horizontal de 92 s, de 60% para 0,2% de quadros descartados, latência
p50/p99 de 144/407 para 15/68 ms e prazo perdido de 99% para 2,7%; no vertical, nenhum
descarte (antes 15%), p50 de 163 para 44 ms — ainda acima do prazo (README §10).

### Leitura

| Mudança | Efeito medido |
|---|---|
| O recorte do OCR sai do quadro **original** (`ocr.full_resolution`), a detecção continua na imagem reduzida | Placas de 35–45 px (a 1280): leituras iguais à decisão final do veículo de 43% para 83%; de 25–35 px, de 0% para 42% |
| `ocr.min_plate_width` 60 → 45 px (`tempo_real.yaml`: 50 → 45) | Na faixa de 45–60 px, 95% das leituras iguais à decisão final, e o limiar antigo a descartava. Placas publicadas: 10 → 14 e 32 → 39; das 12 novas, todas conferem na inspeção visual dos recortes; motos com placa publicada 1 → 3 de 11 |
| A placa só é atribuída a veículo que porta placa | A caixa do piloto (ou de um pedestre na frente do carro) podia ficar com a placa, e a trilha da pessoa nunca passa pelo OCR |

Com 35 px saem 16 + 52 placas, mas a inspeção já mostra falsos positivos (um adesivo
"Disque-Denúncia 181" lido como placa): fica como opção, não como padrão.

### Dados

- **Cada processamento começa do zero**: `lprts run`, `lprts eval` e o **Iniciar** da
  interface apagam o banco configurado e os recortes das sessões dele
  (`analytics.clear_previous`, padrão `true`; `--keep-previous` acumula). Só apaga
  banco do LPR-RTS, e só depois de a fonte abrir. As repetições de `eval --repeat`
  ficam juntas.
- **Tipos de objeto**: a interface ganhou o seletor **Objetos** (`trafego`, `completo`,
  `veiculos`) e o painel mostra os objetos únicos por tipo e os usuários vulneráveis.
  Os perfis `tempo_real*.yaml` passam de `veiculos` para `trafego`: pessoas e
  bicicletas entram nos dados também em tempo real.
- **`lprts report`**: tabelas (CSV + Markdown) e figuras (PDF + PNG) para o artigo —
  vídeos e condições, objetos por classe, funil das placas, OCR por largura da placa,
  desempenho e tempo real —, sem nenhum texto de placa.
- `lprts demo` usa o vídeo sintético versionado em vez de regenerá-lo: regenerado em
  outra máquina, o mp4 diferia 0,03 por pixel (codificação), o arquivo do git ficava
  modificado e a revocação do demo caía de 0,56 para 0,33 (2 de 9 veículos).
- `n_rois_skipped` passa a contar também os veículos estreitos demais;
  `reads.sharpness` é medida no recorte que o OCR recebe (o original).

### Encontrado e não corrigido

- **Fragmentação de trilhas**: no vídeo de 92 s, 45% das trilhas de carro duram menos
  de 1 s — o mesmo veículo perdido numa oclusão vira objeto novo. O `lprts report`
  conta objetos por trilha confirmada (≥ 0,5 s) e mostra as duas contagens; ajustar o
  rastreador exige ground truth (MOTA/IDF1).
- **Caixa aninhada**: um caminhão próximo, com a caixa cobrindo meio quadro, contém o
  carro à frente; a placa do carro pode ir para o caminhão, que a publica (e não
  revisa, com `first_stable`). Um termo de escala placa/veículo resolveu o caso
  isolado mas criou outro erro no mesmo vídeo, e foi descartado.
- PyTorch/MPS é mais lento que a CPU neste pipeline (detecção p50 66 × 62 ms; CoreML
  19 ms): cada busca de placa por veículo paga a sincronização com a GPU.

## 0.3.1 — vídeo real com os detectores neurais (2026-10)

Correções encontradas ao rodar vídeos reais de celular (iPhone, 4K, HEVC). O
esquema do banco não muda.

| Defeito | Efeito | Correção |
|---|---|---|
| O `./start.sh` abria a demo com `--no-vehicle-detector`, e a escolha valia para a interface inteira | Vídeo real ou câmera aberto depois na janela rodava **sem o YOLO** (todas as sessões com `detect_vehicles: false`) | Caixa **Veículos (YOLO)** na janela, ajustada pela fonte: só-placa para o vídeo sintético, YOLO para vídeo real e câmera |
| NMS do YOLO é por classe | O mesmo veículo saía como "car" e "truck" (IoU 0,98–0,99), virava duas trilhas e a placa era publicada duas vezes: 8 duplicatas em 40 decisões num vídeo de 92 s | `detector.vehicle_merge_iou` (0,7): NMS entre classes, só entre veículos que portam placa (piloto e moto continuam separados). No mesmo vídeo: 1 duplicata em 33, as mesmas 32 placas |
| Placa contida em dois veículos sobrepostos: empate decidido pela ordem da lista | Leituras da mesma placa nas duas trilhas (duplicata) ou atribuídas ao carro vizinho | A placa vai para o veículo cujo centro horizontal está mais perto dela |
| O **Exportar** da interface e o `lprts export` sem `--session` levavam o banco inteiro | Todos os vídeos já processados misturados nas mesmas tabelas, com `track_id` repetidos | Exporta só o vídeo analisado (a última sessão), em `<vídeo>_<sessão>/`, com `sessao.json`; `--all` para o banco inteiro |
| Importar o ultralytics executa `cv2.setNumThreads(0)` (pensado para o DataLoader de treino) | Com o YOLO ligado, todo o OpenCV do processo ficava numa thread: no vídeo 4K vertical, redimensionar o quadro passou a custar 70 ms e o processamento caiu para 10 quadros/s | `lprts.runtime.import_ultralytics` restaura as threads: 29,5 quadros/s no mesmo vídeo, com as mesmas placas |

No vídeo real, o detector clássico de placas encontrou placa em 20% dos veículos
detectados e nenhuma em motos (a placa de moto é quase quadrada; o clássico procura
retângulos alongados); o YOLO de placas (`--plate morsetech-n`), em 51%, com leitura
válida em 34% (9% no clássico), motos incluídas. A resolução não limitou o OCR:
placas a partir de ~40 px a 1280 px de largura foram lidas igual ao recorte 4K.

Também nesta versão:

- `scripts/download_models.py --plate morsetech-n|s|m` baixa um detector YOLO11 de placas
  de terceiros (revisão fixada, sha256 conferido) para `data/models/plate_yolo.pt`, sem
  sobrescrever um modelo treinado (`--force`). `--coreml`/`--onnx` passam a exportar
  também o detector de placas, na entrada `--plate-imgsz` (320, a de `detector.plate_imgsz`).
- `coremltools` entra no extra `dl` no macOS, com numpy ≤ 2.3.5 (exigência do ultralytics
  na exportação).

Limitação que permanece: troca de identidade entre dois veículos sobrepostos na
mesma faixa (a trilha de um passa para o outro). Ela gera uma decisão repetida;
com ground truth, o avaliador a conta como duplicata e troca de identidade.

## 0.3.0 — tempo real com modelos neurais, velocidade e política de publicação (2026-10)

Esquema do banco **3**. Bancos da 0.2 (esquema 2) são **migrados ao abrir, sem
perda** (só ganham colunas, preenchidas com os valores verdadeiros para a 0.2);
bancos da 0.1 continuam recusados. Com a configuração padrão, os resultados de
detecção e leitura não mudam, **exceto no Apple Silicon com o `.mlpackage`
exportado**, onde o detector de objetos passa a rodar no CoreML (FP16): as caixas
diferem das do PyTorch em IoU médio 0,993 (mínimo 0,982 nas medidas).

### Correções

| Defeito | Efeito | Correção |
|---|---|---|
| Detector `.mlpackage` recebia `device="mps"` | Toda detecção falhava (`float64` no MPS) | Pós-processamento do CoreML na CPU |
| `.mlpackage` exportado com NMS embutido | `iou` e `imgsz` da configuração ignorados sem aviso; o banco gravava os pedidos | Exportação sem NMS (mesma latência); modelo com NMS embutido gera aviso e os parâmetros **efetivos** são gravados |
| `device: auto` resolvia "coreml" mas o `.pt` rodava em MPS | Caminho mais lento e cauda pior que o disponível | O `.mlpackage` exportado ao lado do `.pt` é usado quando respeita a configuração |
| Vídeo sintético de demonstração era o da 0.1 | Placas na fonte do OCR por gabaritos; GT sem `track_id`/`difficult` | Regerado pela versão atual e versionado (`data/videos/`) |

### Latência medida (Apple M5 Pro, na bateria)

`scripts/benchmark_componentes.py` (cada componente, cada runtime) e pipeline completo em
tempo real a 30 fps com YOLO11n: latência p99 de 11–12 ms em CoreML e 24–27 ms em
PyTorch/MPS, nenhum prazo perdido. Tabelas no README (§10).

### Velocidade e borrão de movimento

O gerador sintético ganhou `--speed`, `--shutter-ms` (borrão por integração de
subquadros ao longo da exposição, centrado no instante anotado) e
`--total-vehicles`, e um gerador aleatório por objeto: **os mesmos veículos em
qualquer velocidade** (desenho pareado). `scripts/curva_velocidade.py` roda a
grade velocidade × obturador × modo × variante e grava tabelas por condição e por
veículo, com velocidade aparente e borrão em pixels.

### Política de publicação

`fusion.publish_policy` (`first_stable` — padrão, comportamento da 0.2 —,
`revise`, `on_exit`) e `fusion.exit_min_reads`. Cada evento grava `decision` e
`revision`; a visão `v_decisions` traz as decisões vigentes; a avaliação mede a
decisão vigente, a primeira decisão emitida e a precisão de tudo o que foi
emitido; o `lprts tune` replica a política exata do pipeline e a inclui no espaço
de busca.

Medido no vídeo sintético pareado (60 veículos, 1×–4×, obturador 0 e 4 ms,
`tempo_real_rapido.yaml`, offline), em veículos ganhos/perdidos contra o padrão:
`exit_min_reads: 1` **+38/−0** (McNemar p < 0,001; revocação em 4× de 0,15 para
0,37 e em 3× de 0,47 para 0,57, precisão mantida); `revise` e `on_exit` +7/−10
(p = 0,63). A diferença entre fusão e publicação em alta velocidade era quase toda
de veículos com 1–2 leituras que nunca chegavam a `min_reads`; a revisão segue
também erros sistemáticos do OCR (Q → O quando a placa se aproxima). Por isso
`tempo_real_rapido.yaml` passa a usar `exit_min_reads: 1`, e o padrão continua
`first_stable` sem decisão relaxada. Detalhes em `docs/AVALIACAO.md`.

## 0.2.0 — auditoria de integridade dos dados (2026-10)

Esta versão corrige erros que **invalidavam dados gravados pela 0.1.0**. Bancos
da versão anterior (esquema 1) são recusados — não misture resultados das duas
versões. Toda sessão agora registra o commit do código (`sessions.code_version`):
cite-o junto dos resultados.

### Erros de dados corrigidos

| # | Defeito na 0.1.0 | Efeito medido | Correção |
|---|---|---|---|
| 1 | Base de tempo física era o relógio de processamento (`time.monotonic()`) | Mesmo veículo a 219 km/h numa máquina rápida e 17 km/h numa lenta; `dwell_s` 150× menor; `first_seen_s` em segundos desde o boot | Tempo de mídia (`Frame.source_pts`) em toda medida física; relógio de processamento só para latência |
| 2 | Linha de `tracks` gravada na publicação e nunca atualizada; trilhas encerradas sem placa nunca gravadas | Veículos com 30–80 quadros gravados com 5–9; de 6 pedestres, 0 ou 1 no banco; composição de tráfego vazia | Trilhas gravadas ao terminar (saída de cena, fim do vídeo, troca de perfil), com estado completo |
| 3 | `frames.dropped_frames` atribuído depois da gravação | 39% de descarte registrado como 0 | Descartes acumulados e `frames_skipped` gravados por quadro |
| 4 | Eventos do fim do vídeo sem critério de estabilidade, com `video_time_s = 0` e o primeiro fragmento lido como texto | Eventos `"III"`, `"1"`, `"CI74I"` com 1–2 leituras | Mesmo critério de publicação em todo lugar; texto e instante corretos |
| 5 | `match_thresh = 0.8` usado como IoU mínimo (no ByteTrack é custo máximo, IoU ≥ 0,2) | Carro de 200 px a ≥ 25 px/quadro nunca confirmado (logo, nunca lido) | `tracker.match_iou = 0.2`, `second_match_iou = 0.5` (valores efetivos do ByteTrack) |
| 6 | `detect_every_n > 1` reenviava detecções antigas ao rastreador | Velocidade do Kalman → 0; com n = 3, um veículo → 20 trilhas | Quadros sem detecção só fazem predição |
| 7 | Caixas na escala de `resize_width`; ground truth e calibração na original | `lprts eval` em 1080p: detecção com P = R = 0, sem aviso | Banco sempre em pixels da imagem original; calibração registra e valida a resolução |
| 8 | Leitura assíncrona gravada no quadro em que foi colhida | Deslocamento de 2–3 quadros entre `frame_index` e a caixa | Leitura gravada com o quadro do recorte (`frame_index`) e o da coleta (`collected_frame`) |
| 9 | Pipeline paralelo em tempo real reordenava por índice da fonte | Latência real ~1,6 s com o banco registrando ~90 ms; descartes contados em dobro | Reordenação por número de sequência de retirada (sem buracos) |
| 10 | Kalman previa 1 passo mesmo com quadros pulados | Associação falha para veículos rápidos sob carga | Predição com o número real de quadros decorridos; `max_age` em quadros da fonte |
| 11 | Dispositivo gravado era a preferência resolvida | "CoreML/Neural Engine" registrado com tudo em CPU/MPS | Runtime e dispositivo **efetivos** de cada componente |
| 12 | Motor de OCR pedido e ausente caía em silêncio para o baseline | Relatório "paddle" com números do `template` | Pedido explícito ausente é erro; só `auto` escolhe (e registra) |
| 13 | `fast-plate-ocr` recebia BGR (modelo é RGB) e a confiança incluía 2 posições de preenchimento | Confiança inflada; ponderação por caractere nunca usada | Conversão para RGB; confiança só sobre os caracteres reais |
| 14 | Gramática aceitava 3 correções | `1231231` → `IZB1231` "válida" | Padrão de 2 correções (`ocr.max_grammar_corrections`); número de correções gravado |
| 15 | `raw_text` do OCR por gabaritos já era a saída restrita pela gramática | Ablação da gramática impossível nesse motor | `raw_text` = decodificação livre |
| 16 | Fila do banco descartava qualquer linha quando cheia | Eventos podiam sumir só com aviso no log | Resultados nunca são descartados; perda de telemetria contada por tabela na sessão |
| 17 | Chaves desconhecidas no YAML eram ignoradas | Configuração gravada diferente da pretendida | Chave desconhecida ou renomeada é erro, com sugestão |
| 18 | Flags da CLI com padrão fixo sobrescreviam o YAML | `configs/tempo_real.yaml` rodava em modo offline | Flags ausentes não sobrescrevem a configuração |
| 19 | Contador de `track_id` global ao processo | Mesmo vídeo, IDs diferentes no mesmo processo | Contador por rastreador |
| 20 | Fragmentos do lote reaproveitados entre execuções | Banco consolidado misturava sessões antigas | Pasta própria por execução |

### Erros de metodologia de avaliação corrigidos

- **Pareamento por posição.** Leituras, trilhas e eventos são associados ao
  veículo anotado por sobreposição espacial no mesmo quadro. A versão anterior
  pareava pelo texto mais parecido (otimista: em corpora de 1.000–5.000 placas,
  leituras com 3–4 erros caem na placa errada em 11% a 89% dos casos).
- **Métrica por veículo** como métrica principal: veículos perdidos contam,
  duplicatas são contadas. A "exatidão por trilha" anterior só considerava os
  eventos publicados (relatava 27/27 onde o real era 26/27 com uma duplicata).
- **Efeito da fusão** medido na mesma população de veículos (leitura ao acaso ×
  leitura mais confiante × fusão × decisão publicada), com teste de McNemar. O
  "ganho da fusão" anterior comparava denominadores diferentes.
- **Tentativas sem texto** entram no denominador da exatidão por tentativa.
- **Amostra de leituras sem censura** no perfil de avaliação
  (`ocr.max_reads_published: 0`); antes as trilhas paravam de ser lidas após a
  publicação, e a amostra dependia do acerto do próprio motor.
- **`difficult`** segue o protocolo PASCAL VOC (nem acerto nem erro).
- **Intervalos de confiança** (bootstrap; por agrupamento nas leituras).
- **`lprts tune`**: verdade por posição, revocação sobre veículos anotados,
  réplica exata da decisão online, validação em veículos reservados
  (`--holdout`) e hipervolume com referência fixa.
- **Vídeo sintético**: placas em DejaVu Sans Bold (fora do banco de gabaritos
  do OCR clássico, eliminando o vazamento); ground truth com `track_id`, caixas
  de objeto e `difficult` (placas pequenas, cortadas ou ocluídas); metadados da
  geração em `<saida>_meta.json`.

### Novos dados gravados

Tabelas `detections` e `track_points`; colunas de latência ponta a ponta, fila,
prazo, descartes por quadro, correções da gramática, confiança por caractere,
brilho/contraste, tempo absoluto (`media_time_utc`), proveniência (commit,
pacotes, hardware, energia) e contadores da sessão. Ver
[`docs/DICIONARIO_DE_DADOS.md`](docs/DICIONARIO_DE_DADOS.md).

### Perfis de tempo real

`configs/tempo_real.yaml` foi corrigido e há um perfil novo,
`configs/tempo_real_rapido.yaml`. Medido em tempo real (vídeo sintético com
veículos 1×, 2× e 3× mais rápidos; revocação por veículo):

| Perfil | 1× | 2× | 3× | Latência p99 |
|---|---|---|---|---|
| `tempo_real.yaml` anterior (detecção alternada, OCR a cada 3 quadros, 70 px em 960) | 0,84 | 0,27 | 0,02 | 13 ms |
| `tempo_real.yaml` corrigido (detecção em todo quadro, OCR alternado, 50 px em 1280) | 0,86 | 0,84 | 0,53 | 8,5 ms |
| `tempo_real_rapido.yaml` (tudo em todo quadro, resolução original) | 0,86 | 0,86 | 0,61 | 9,1 ms |

### Novos comandos

`lprts eval-db` (reavalia uma sessão gravada), `lprts compare` (comparação
pareada de dois relatórios), `lprts eval --realtime --repeat N`, `make lock`.
