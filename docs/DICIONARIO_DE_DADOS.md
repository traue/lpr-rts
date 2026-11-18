# Dicionário de dados (esquema 4)

Referência de cada coluna gravada pelo LPR-RTS 0.5. O esquema executável está em
[`src/lprts/analytics/schema.sql`](../src/lprts/analytics/schema.sql); receitas
de análise em [DADOS.md](DADOS.md).

**Esquema 4 = esquema 3 + relógio da decisão.** A migração dos esquemas 2 e 3
é transacional. `events.decision_elapsed_ms` fica nulo para decisões antigas:
não é possível reconstruir essa medida usando apenas o tempo de mídia.

**Esquema 3 = esquema 2 + decisões revisáveis.** Ao ser aberto pela 0.3 ou posterior, um banco
da 0.2 é migrado no lugar, sem perda: ganha `events.decision`, `events.revision`,
`tracks.first_published_text`, `tracks.n_revisions`, `tracks.published_decision` e a
visão `v_decisions`, preenchidos com os valores verdadeiros para a 0.2 (que só
conhecia a primeira decisão estável: `decision = 'stable'`, `revision = 0`). Bancos
da 0.1 (esquema 1) continuam recusados.

## Convenções que valem para todas as tabelas

| Convenção | Regra |
|---|---|
| **Coordenadas** | Pixels da imagem **original** da fonte, origem no canto superior esquerdo, `(x1, y1)` canto superior esquerdo da caixa. Valem para qualquer `source.resize_width`: o pipeline processa na resolução reduzida e converte de volta antes de gravar. É o mesmo espaço do ground truth e da calibração. Em vídeo de celular gravado na vertical, a imagem original é a **girada** (como o player a exibe), e `source_width/height` são as dimensões dela. |
| **Tempo de mídia** | Colunas `video_time_s`, `*_seen_s`, `published_time_s`, `dwell_s`, o `t` da trajetória: segundos desde o início do vídeo (arquivo: carimbo PTS do contêiner) ou desde a abertura da fonte (câmera/stream ao vivo: instante de chegada). É a base de **toda medida física** (velocidade, permanência). |
| **Relógio de processamento** | Colunas `*_t_s` de `frames`: segundos desde o primeiro quadro processado na sessão, medidos com `time.monotonic()`. Servem só para vazão e latência. |
| **Horários absolutos** | `ts`: quando a linha foi **produzida** (UTC). `media_time_utc`: instante **real** do quadro, = `sessions.recording_start_utc` + `video_time_s`; nulo se o início da gravação for desconhecido (vídeo gravado sem `--recording-start`). Para análises "por hora do dia" use `media_time_utc`, nunca `ts`. |
| **Identificadores** | `track_id` é único **dentro da sessão** (recomeça em 1 a cada sessão). Junte sempre por `(session_id, track_id)`. |
| **Confiança** | Escala própria de cada motor de OCR, **não calibrada** e **não comparável** entre motores (ver `reads.confidence`). |
| **Booleanos** | Inteiros 0/1. |

## `sessions` — uma execução

| Coluna | Tipo | Descrição |
|---|---|---|
| `session_id` | TEXT | Identificador (12 hex). |
| `schema_version` | INT | Versão do esquema com que a sessão foi gravada (4; sessões migradas mantêm a versão original). |
| `started_at`, `finished_at` | TEXT | Início e fim (ISO-8601 UTC, relógio de parede). |
| `end_reason` | TEXT | `fim_da_fonte`, `interrompido`, `erro`, `erro_persistencia`. |
| `source_uri`, `source_kind` | TEXT | Fonte; tipo `arquivo`, `ao_vivo` ou `pasta`. |
| `source_fps` | REAL | Taxa nominal da fonte. |
| `source_width/height` | INT | Resolução **original**. |
| `processing_width/height` | INT | Resolução processada (após `resize_width`). |
| `frame_count` | INT | Quadros declarados pelo contêiner (pode ser aproximado). |
| `recording_start_utc` | TEXT | Instante real do primeiro quadro (ao vivo: automático; gravado: `--recording-start`). |
| `realtime` | INT | 1 = modo tempo real (descarta quadros, OCR assíncrono). |
| `deadline_ms` | REAL | Prazo usado em `frames.deadline_miss` (padrão: um período de quadro). |
| `detector`, `plate_detector`, `ocr_engine` | TEXT | Componentes efetivamente carregados. |
| `device` | TEXT | Runtime e dispositivo **efetivos** por componente, ex. `objetos=coremltools@coreml; placas=coremltools@coreml; ocr=onnxruntime@cpu` (0.4 no Apple Silicon, com os `.mlpackage` exportados). Outros valores: `ultralytics/coreml@coreml` (o mesmo `.mlpackage` pelo ultralytics: o caminho da 0.3, ou `detector.runtime: ultralytics`), `ultralytics/pytorch@mps`/`@cpu`/`@cuda` (pesos `.pt`), `onnxruntime@<provedores>` (`.onnx`), `opencv@cpu` (detector clássico). Separe as sessões por esta coluna ao comparar desempenho entre versões. |
| `code_version` | TEXT | Commit do código (`+dirty` se havia alterações não commitadas). |
| `config_hash` | TEXT | Hash da configuração que afeta resultados (agrupa execuções equivalentes). |
| `config_json` | TEXT | Configuração completa. |
| `environment_json` | TEXT | Plataforma, hardware (modelo do chip, núcleos, memória), energia (tomada/bateria, modo de baixo consumo), versões de todos os pacotes, componentes (runtime, dispositivo, hash dos pesos) e `video_backend`, o decodificador efetivo da fonte (`AVFOUNDATION`, `FFMPEG`...). |
| `stats_json` | TEXT | Contadores finais: trilhas encerradas, hipóteses descartadas, falhas de OCR, leituras perdidas no encerramento, descartes da fonte, reconexões, linhas não gravadas por tabela (`rows_lost`); desde a 0.5, `pipeline_workers` (0 = serial), `ocr_prefetched` e `ocr_prefetch_used` (leituras feitas antes pelos workers e as aproveitadas), `ocr_backpressure` e `ocr_expired` (OCR em tempo real recusado pelo teto ou vencido) e, em `erro_persistencia`, `storage_error`. |
| `notes` | TEXT | Decisões em tempo de execução (ex.: fallback para modo só-placa, com o quadro em que ocorreu). |

## `frames` — telemetria por quadro processado

| Coluna | Unidade | Descrição |
|---|---|---|
| `frame_index` | — | Índice do quadro na fonte (com descartes, há saltos). |
| `video_time_s` | s (mídia) | Tempo do quadro. |
| `capture_t_s`, `start_t_s`, `done_t_s` | s (processamento) | Quadro disponível, início e fim do processamento. |
| `detection_ran` | 0/1 | 0 nos quadros intermediários de `detect_every_n` (só predição). |
| `n_object_dets`, `n_plate_dets` | contagem | Detecções brutas neste quadro. |
| `n_tracks`, `n_vehicle_tracks`, `n_person_tracks` | contagem | Trilhas reportadas neste quadro (e quantas são veículos motorizados / pessoas). |
| `n_rois_skipped` | contagem | Veículos que portam placa mas não tiveram a placa procurada neste quadro: estreitos demais para uma placa legível (`detector.plate_width_ratio`) ou acima do teto `max_rois_per_frame`. A largura de cada veículo está em `detections`, o que separa os dois motivos. |
| `capture_ms` | ms | Leitura/decodificação (e redimensionamento). |
| `queue_ms` | ms | Espera entre o quadro ficar disponível e o processamento começar. |
| `detect_ms`, `track_ms`, `ocr_ms`, `analytics_ms` | ms | Tempo de cada estágio. `ocr_ms`: offline = OCR síncrono; tempo real = só agendar e colher (o OCR roda em paralelo — ver `reads.ocr_ms`). |
| `total_ms` | ms | **Tempo de serviço** (soma dos estágios). |
| `latency_ms` | ms | **Latência ponta a ponta**: do quadro disponível ao resultado pronto (fila + serviço). Não inclui exposição do sensor nem buffer do driver da câmera. |
| `deadline_miss` | 0/1 | `latency_ms > sessions.deadline_ms`. Recalcule para outro prazo a partir de `latency_ms`. |
| `ocr_in_flight` | contagem | Leituras assíncronas em execução ao fim do quadro. |
| `fps` | quadros/s | Vazão na janela móvel dos últimos 60 quadros (0 nos primeiros). Para vazão da sessão use `v_session_performance.throughput_fps`. |
| `dropped_frames` | contagem | Descartes **acumulados** na sessão até este quadro. |
| `frames_skipped` | contagem | Quadros da fonte pulados desde o quadro processado anterior. |

## `detections` — cada detecção

| Coluna | Descrição |
|---|---|
| `frame_index`, `video_time_s` | Quadro e tempo. Só quadros com `detection_ran = 1`. |
| `kind` | `object` (detector de objetos) ou `plate` (detector de placas). No modo só-placa só há `plate`. |
| `label`, `class_id`, `score` | Rótulo, classe e score do detector (o detector clássico produz score heurístico, **não** probabilidade). |
| `x1, y1, x2, y2` | Caixa (pixels originais). |
| `track_id` | Trilha que esta detecção alimentou neste quadro (NULL = nenhuma). |

## `track_points` — estado de cada trilha em cada quadro reportado

| Coluna | Descrição |
|---|---|
| `track_id`, `frame_index`, `video_time_s` | Trilha, quadro, tempo. |
| `predicted` | 1 = caixa só prevista pelo Kalman (quadro sem detecção em `detect_every_n`). |
| `score` | Score da última detecção associada. |
| `x1..y2` | Caixa da trilha após o filtro de Kalman. No modo só-placa é a caixa da placa. |
| `plate_x1..plate_y2` | Placa associada à trilha neste quadro (NULL = nenhuma). |

É a trajetória completa, quadro a quadro, e a base das métricas de rastreamento
(MOTA, IDF1).

## `tracks` — resumo final de cada trilha

Uma linha por trilha confirmada, gravada quando a trilha **termina**. Hipóteses do
rastreador que nunca chegaram a ser confirmadas (`min_hits`) não são gravadas.

| Coluna | Descrição |
|---|---|
| `object_class`, `object_category`, `is_vulnerable` | Classe (taxonomia em português), categoria e se é usuário vulnerável. No modo só-placa a classe é `placa_veicular`. |
| `first_frame`, `last_frame` | Primeiro e último quadro **com detecção associada**. |
| `first_seen_s`, `last_seen_s`, `dwell_s` | Tempos de mídia correspondentes e permanência. |
| `n_frames` | Quadros com detecção associada (não confundir com o número de pontos em `trajectory_json`). |
| `n_reads` | Leituras de OCR com texto, todas (mesmo as descartadas por `fusion.max_reads_per_track`). |
| `plate_text`, `plate_layout`, `plate_confidence`, `plate_agreement`, `plate_valid_syntax` | Fusão **final** sobre as leituras retidas. |
| `plate_stable` | A fusão final atende ao critério de publicação. |
| `published`, `published_text`, `published_frame`, `published_time_s` | Se houve decisão; o texto da decisão **vigente**; quadro e instante da **primeira** decisão. Pode diferir de `plate_text` (fusão final): com `fusion.publish_policy: first_stable` a decisão não é revista. |
| `first_published_text` | Texto da primeira decisão (igual a `published_text` sem revisões). |
| `n_revisions` | Revisões publicadas (`publish_policy: revise`). |
| `published_decision` | Tipo da decisão vigente: `stable`, `revision` ou `exit_relaxed` (ver `events.decision`). |
| `speed_kmh` | Mediana das velocidades instantâneas na janela recente, a partir do ponto de contato com o solo (meio da borda inferior). NULL sem calibração. No modo só-placa o ponto está na placa (~0,5 m acima do solo): viés sistemático. |
| `direction` | Direção **na imagem**: `direita`, `esquerda`, `baixo`, `cima`, `parado`. Não são pontos cardeais. |
| `trajectory_json` | `[[x, y, t_s, frame], ...]` — centro da caixa (pixels originais), tempo de mídia e quadro; dizimada (um ponto a cada dois, preservando o primeiro e o último) acima de 2.000 pontos. A versão completa está em `track_points`. |
| `crop_path` | Recorte mais nítido salvo (eventos e trilhas lidas mas não publicadas). |
| `end_reason` | `lost` (saiu de cena), `end_of_stream` (vivo no fim), `reset` (troca de perfil no fallback). Vazio = registro provisório de uma trilha publicada cuja gravação final não ocorreu (ex.: processo interrompido). |

## `reads` — cada tentativa de OCR

| Coluna | Descrição |
|---|---|
| `track_id` | Trilha lida. |
| `frame_index`, `video_time_s` | Quadro **de onde saiu o recorte**. |
| `collected_frame` | Quadro em que o resultado entrou na fusão (= `frame_index` offline; maior no tempo real). |
| `raw_text` | Saída do motor **antes** da gramática (no OCR por gabaritos: decodificação livre, sem máscara). |
| `text` | Normalizado e corrigido pela gramática. **Vazio = tentativa sem texto** — conte no denominador. |
| `n_corrections` | Caracteres trocados pela gramática. |
| `confidence` | Confiança do motor. `fastplate`: média das probabilidades dos caracteres reais; `template`: correlação média reescalada para [0,1] (0,5 = nenhuma correlação); `easyocr`/`paddle`: média dos scores dos trechos; `tesseract`: conf/100. Calibre por motor antes de modelar. |
| `char_confidences` | JSON com a confiança por caractere, alinhada a `text` (vazio quando o motor não fornece ou não é possível alinhar). |
| `layout`, `valid_syntax` | Layout (`mercosul`/`legacy`/`unknown`) e se fecha a gramática após a correção. |
| `engine` | Motor. |
| `ocr_ms` | Tempo dentro do motor. |
| `wait_ms` | Espera na fila do pool de OCR (tempo real). |
| `sharpness` | Variância do Laplaciano do recorte entregue ao OCR — o do quadro **original** com `ocr.full_resolution` (padrão desde a 0.4), o da imagem processada sem ela. Depende da resolução: compare dentro da mesma configuração. |
| `brightness`, `contrast` | Média e desvio-padrão da luminância do recorte (0–255). |
| `plate_w`, `plate_h` | Tamanho da placa em pixels originais. Na imagem processada (a da detecção e do limiar `ocr.min_plate_width`), `plate_w × processing_width / source_width`; com `ocr.full_resolution` o OCR recebe o recorte original, de largura `plate_w`. |
| `bbox_x1..y2` | Caixa da placa (pixels originais). |

## `events` — decisões publicadas

Cada decisão sobre a placa de uma trilha, conforme `fusion.publish_policy`
(ver [`lprts.fusion.publication`](../src/lprts/fusion/publication.py)):

* `first_stable` (padrão): um evento por trilha, na primeira fusão que atinge o
  critério de publicação (`fusion.min_reads`, `min_agreement`, `min_confidence` e,
  se exigido, sintaxe válida) — inclusive no fim do vídeo;
* `revise`: além disso, um evento de revisão sempre que uma fusão estável
  posterior tiver outro texto;
* `on_exit`: um evento por trilha, quando ela termina.

Com `fusion.exit_min_reads > 0`, uma trilha que termina sem decisão publica com
menos leituras (`decision = 'exit_relaxed'`). A decisão **vigente** de uma trilha é
a de maior `revision` (visão `v_decisions`).

| Coluna | Descrição |
|---|---|
| `track_id`, `frame_index`, `video_time_s`, `media_time_utc`, `ts` | Trilha e instante da decisão (mídia, real, produção). |
| `plate_text`, `plate_layout`, `valid_syntax` | Placa publicada. |
| `confidence`, `agreement` | Confiança e concordância da fusão no momento da decisão. |
| `n_reads` | Leituras que votaram **no momento da decisão**. |
| `track_first_seen_s` | Início da trilha na mídia. `video_time_s − track_first_seen_s` é o intervalo de mídia, não o tempo real até disponibilizar a decisão. |
| `decision_elapsed_ms` | Relógio monotônico: primeira imagem da trilha disponível no host até a produção da decisão. Inclui OCR tardio no fim da fonte. Não inclui exposição/transporte da câmera nem confirmação de persistência/entrega ao consumidor. Nulo quando desconhecido. |
| `object_class`, `object_category` | Classe da trilha. |
| `bbox_x1..y2`, `speed_kmh`, `direction`, `crop_path` | Estado no momento da decisão. |
| `decision` | `stable` (critério de publicação), `revision` (fusão estável posterior com outro texto) ou `exit_relaxed` (na saída, com `fusion.exit_min_reads` leituras). |
| `revision` | 0 na primeira decisão da trilha; 1, 2, ... nas revisões. |

## `crossings` — cruzamentos de linhas virtuais

| Coluna | Descrição |
|---|---|
| `line_name`, `direction` | Linha e sentido (`in`/`out`, pelo lado da reta). |
| `video_time_s`, `media_time_utc`, `ts`, `frame_index` | Instante do cruzamento. |
| `speed_kmh`, `plate_text`, `object_class`, `object_category` | Estado da trilha no cruzamento (`plate_text` pode ser provisória). |

## Visões

| Visão | Conteúdo |
|---|---|
| `v_decisions` | Decisões vigentes: por trilha, o evento de maior `revision`. Sem revisões, igual a `events`. |
| `v_unique_plates` | Uma linha por placa publicada e sessão (sobre as decisões vigentes). |
| `v_object_counts`, `v_category_share`, `v_vulnerable_exposure` | Composição do tráfego sobre trilhas **encerradas**. |
| `v_line_composition` | Cruzamentos por linha, categoria e sentido. |
| `v_session_performance` | Vazão (quadros/tempo), latência e serviço médios/máximos, descartes, taxa de prazo perdido. Percentis: calcule de `frames.latency_ms`. |

## Tabelas da avaliação (`lprts eval` / `eval-db`)

Gravadas ao lado do relatório JSON, prontas para modelagem:

| Arquivo | Unidade | Colunas principais |
|---|---|---|
| `<relatorio>_veiculos.csv` | veículo anotado | `vehicle`, `plate_text`, `max_plate_width_px`, `readable`, `excluded_difficult`, `detected`, `n_tracks`, `n_attempts`, `n_reads_text`, `single_read_accuracy`, `best_read_correct`, `fused_text`, `fused_correct`, `published` (alguma decisão vigente), `published_correct` (decisão vigente), `dominant_published_correct` (decisão da trilha que mais leu o veículo: a linha "Publicado" do efeito da fusão), `first_published_correct` (primeira decisão emitida para o veículo), `published_exit_relaxed`, `n_events` (decisões vigentes), `n_wrong_events`, `n_revisions`, `n_wrong_emitted` (inclui as revisadas), `first_correct_time_s` (instante de vídeo da primeira decisão correta), `time_to_correct_s` (latência de decisão: tempo de vídeo do primeiro quadro anotado até essa decisão), `elapsed_to_correct_s` (relógio do host, da primeira imagem da trilha até a mesma decisão; nulo antes do esquema 4), `mean_sharpness` |
| `<relatorio>_leituras.csv` | tentativa de OCR | `gt_vehicle`, `gt_plate`, `iou`, `text`, `raw_text`, `correct`, `cer`, `empty`, `confidence`, `n_corrections`, `sharpness`, `brightness`, `contrast`, `plate_w`, `plate_h`, `ocr_ms`, `wait_ms` |
| `<relatorio>_eventos.csv` | evento publicado | `gt_vehicle`, `gt_plate`, `correct`, `confidence`, `agreement`, `n_reads`, `decision_delay_s` (tempo de vídeo do início da trilha até a decisão), `decision_elapsed_s` (relógio do host, da primeira imagem da trilha até a decisão; nulo antes do esquema 4), `decision_ts` (UTC em que a decisão foi produzida), `decision`, `revision`, `in_force` (1 = decisão vigente), `out_of_scope` (só com `--scope-min-width`: 1 = evento de trilha sem veículo anotado e placa mais estreita que o escopo, fora da precisão) |

As leituras de um mesmo veículo **não são independentes**: em modelos de
regressão sobre `_leituras.csv`, use efeitos aleatórios por `gt_vehicle` (ou erros
agrupados por veículo).
