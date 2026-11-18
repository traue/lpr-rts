-- Esquema do banco de resultados (SQLite) - versao 4.
-- 4 acrescenta o relogio do host ate a decisao; 2 e 3 sao migradas ao abrir.
--
-- Convencoes que valem para TODAS as tabelas (ver docs/DICIONARIO_DE_DADOS.md):
--   * coordenadas em pixels da imagem ORIGINAL da fonte (mesmo espaco do ground
--     truth e da calibracao), qualquer que seja source.resize_width;
--   * video_time_s / *_s = tempo de midia em segundos (desde o inicio do video,
--     ou desde a abertura do stream ao vivo) - a base de toda medida fisica;
--   * *_t_s em `frames` = relogio de processamento, segundos desde o primeiro
--     quadro da sessao - so para medir vazao e latencia;
--   * ts = horario (UTC) em que a linha foi PRODUZIDA pelo pipeline;
--     media_time_utc = horario real do quadro, quando conhecido.
--
-- Niveis:
--   sessions     -> uma execucao (config, ambiente, proveniencia)
--   frames       -> telemetria de cada quadro processado
--   detections   -> cada deteccao (objeto ou placa) de cada quadro
--   track_points -> estado de cada trilha em cada quadro em que foi reportada
--   tracks       -> resumo final de cada trilha (gravado quando ela termina)
--   reads        -> cada tentativa de OCR (inclusive as que nao deram texto)
--   events       -> cada placa publicada pela fusao (a "placa lida")
--   crossings    -> cruzamentos de linhas virtuais

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    schema_version    INTEGER NOT NULL DEFAULT 4,
    started_at        TEXT NOT NULL,             -- ISO-8601 UTC (relogio de parede)
    finished_at       TEXT,
    end_reason        TEXT,                      -- fim_da_fonte | interrompido | erro | ...
    source_uri        TEXT NOT NULL,
    source_kind       TEXT,                      -- arquivo | ao_vivo | pasta
    source_fps        REAL,
    source_width      INTEGER,                   -- resolucao ORIGINAL
    source_height     INTEGER,
    processing_width  INTEGER,                   -- resolucao processada (resize_width)
    processing_height INTEGER,
    frame_count       INTEGER,
    recording_start_utc TEXT,                    -- instante real do 1o quadro, se conhecido
    realtime          INTEGER NOT NULL DEFAULT 0,  -- 1 = modo tempo real (pode descartar)
    deadline_ms       REAL,                      -- prazo de latencia usado em frames.deadline_miss
    detector          TEXT,
    plate_detector    TEXT,
    ocr_engine        TEXT,
    device            TEXT,                      -- dispositivo EFETIVO por componente
    code_version      TEXT,                      -- commit do git (+dirty) ou versao do pacote
    config_hash       TEXT,                      -- hash da configuracao que afeta resultados
    config_json       TEXT,                      -- config completa, para reprodutibilidade
    environment_json  TEXT,                      -- SO, hardware, energia, pacotes, componentes
    stats_json        TEXT,                      -- contadores finais (descartes, linhas perdidas...)
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    frame_index     INTEGER NOT NULL,
    video_time_s    REAL,
    capture_t_s     REAL,                        -- quadro disponivel (relogio de processamento)
    start_t_s       REAL,                        -- inicio do processamento
    done_t_s        REAL,                        -- resultado pronto
    detection_ran   INTEGER,                     -- 0 nos quadros intermediarios de detect_every_n
    n_object_dets   INTEGER,                     -- deteccoes de objeto neste quadro
    n_plate_dets    INTEGER,                     -- deteccoes de placa neste quadro
    n_tracks        INTEGER,                     -- trilhas reportadas neste quadro
    n_vehicle_tracks INTEGER,                    -- ...das quais veiculos motorizados
    n_person_tracks INTEGER,                     -- ...das quais pessoas
    n_rois_skipped  INTEGER,                     -- veiculos sem busca de placa (estreitos demais ou max_rois_per_frame)
    capture_ms      REAL,                        -- leitura/decodificacao
    queue_ms        REAL,                        -- espera entre captura e processamento
    detect_ms       REAL,
    track_ms        REAL,
    ocr_ms          REAL,                        -- offline: OCR; tempo real: agendar+colher
    analytics_ms    REAL,
    total_ms        REAL,                        -- tempo de SERVICO (soma dos estagios)
    latency_ms      REAL,                        -- PONTA A PONTA: captura -> resultado
    deadline_miss   INTEGER,                     -- latency_ms > sessions.deadline_ms
    ocr_in_flight   INTEGER,                     -- leituras de OCR ainda em execucao
    fps             REAL,                        -- vazao na janela movel dos ultimos 60 quadros
    dropped_frames  INTEGER,                     -- descartes ACUMULADOS na sessao
    frames_skipped  INTEGER                      -- quadros da fonte pulados antes deste
);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    frame_index     INTEGER NOT NULL,
    video_time_s    REAL,
    kind            TEXT NOT NULL,               -- object | plate
    label           TEXT,
    class_id        INTEGER,
    score           REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    track_id        INTEGER                      -- trilha associada neste quadro (NULL = nenhuma)
);

CREATE TABLE IF NOT EXISTS track_points (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    frame_index     INTEGER NOT NULL,
    video_time_s    REAL,
    predicted       INTEGER,                     -- 1 = caixa so prevista (sem deteccao)
    score           REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,          -- caixa da trilha (pos-Kalman)
    plate_x1 REAL, plate_y1 REAL, plate_x2 REAL, plate_y2 REAL  -- placa associada (NULL = nenhuma)
);

CREATE TABLE IF NOT EXISTS tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    object_class    TEXT,                        -- carro, pessoa, bicicleta, placa_veicular...
    object_category TEXT,                        -- pessoa | veiculo_motorizado | ...
    is_vulnerable   INTEGER DEFAULT 0,           -- pedestre, ciclista ou motociclista
    first_frame     INTEGER,
    last_frame      INTEGER,                     -- ultimo quadro com deteccao associada
    first_seen_s    REAL,                        -- tempo de midia (s)
    last_seen_s     REAL,
    dwell_s         REAL,                        -- permanencia, em tempo de midia
    n_frames        INTEGER,                     -- quadros com deteccao associada (hits)
    n_reads         INTEGER,                     -- leituras de OCR com texto (todas)
    plate_text      TEXT,                        -- fusao FINAL sobre as leituras retidas
    plate_layout    TEXT,
    plate_confidence REAL,
    plate_agreement REAL,
    plate_valid_syntax INTEGER,
    plate_stable    INTEGER,                     -- a fusao final atende ao criterio de publicacao
    published       INTEGER,                     -- houve evento para esta trilha
    published_text  TEXT,                        -- texto da decisao VIGENTE ('' se nao houve)
    published_frame INTEGER,                     -- quadro da PRIMEIRA decisao
    published_time_s REAL,                       -- instante da PRIMEIRA decisao
    speed_kmh       REAL,                        -- NULL sem calibracao metrica
    direction       TEXT,                        -- direcao NA IMAGEM: direita|esquerda|baixo|cima|parado
    trajectory_json TEXT,                        -- [[x, y, t_s, frame], ...] (centro; dizimada se longa)
    crop_path       TEXT,
    end_reason      TEXT,                        -- lost | end_of_stream | reset ('' = provisoria)
    first_published_text TEXT DEFAULT '',        -- texto da primeira decisao (antes de revisoes)
    n_revisions     INTEGER DEFAULT 0,           -- revisoes publicadas (fusion.publish_policy=revise)
    published_decision TEXT DEFAULT '',          -- tipo da decisao vigente (ver events.decision)
    UNIQUE(session_id, track_id)
);

CREATE TABLE IF NOT EXISTS reads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    frame_index     INTEGER,                     -- quadro de onde o recorte foi tirado
    video_time_s    REAL,
    collected_frame INTEGER,                     -- quadro em que entrou na fusao (tempo real: depois)
    raw_text        TEXT,                        -- saida do motor ANTES da gramatica
    text            TEXT,                        -- normalizado e corrigido ('' = OCR sem texto)
    n_corrections   INTEGER,                     -- caracteres trocados pela gramatica
    confidence      REAL,                        -- escala propria de cada motor (nao calibrada)
    char_confidences TEXT,                       -- JSON, alinhado a `text` (vazio se indisponivel)
    layout          TEXT,
    valid_syntax    INTEGER,
    engine          TEXT,
    ocr_ms          REAL,                        -- tempo dentro do motor
    wait_ms         REAL,                        -- espera na fila do pool de OCR
    sharpness       REAL,                        -- variancia do Laplaciano do recorte processado
    brightness      REAL,
    contrast        REAL,
    plate_w         REAL,                        -- largura/altura da placa (pixels originais)
    plate_h         REAL,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    ts              TEXT NOT NULL,               -- quando o evento foi PRODUZIDO (UTC)
    video_time_s    REAL,                        -- quadro de referencia, em tempo de midia
    media_time_utc  TEXT,                        -- instante real do quadro, se conhecido
    frame_index     INTEGER,
    plate_text      TEXT NOT NULL,
    plate_layout    TEXT,
    valid_syntax    INTEGER,
    confidence      REAL,
    agreement       REAL,
    n_reads         INTEGER,                     -- leituras que votaram NO MOMENTO da decisao
    track_first_seen_s REAL,                     -- primeiro quadro da trilha, em tempo de midia
    decision_elapsed_ms REAL,                    -- monotonic: primeira imagem disponivel -> decisao
    object_class    TEXT,
    object_category TEXT,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    speed_kmh       REAL,
    direction       TEXT,
    crop_path       TEXT,
    -- stable = criterio de publicacao (fusion.min_reads etc.); revision = fusao
    -- estavel posterior com outro texto (publish_policy=revise); exit_relaxed =
    -- trilha terminou sem decisao e publicou com fusion.exit_min_reads leituras.
    decision        TEXT NOT NULL DEFAULT 'stable',
    revision        INTEGER NOT NULL DEFAULT 0   -- 0 = primeira decisao da trilha; vigente = a maior
);

CREATE TABLE IF NOT EXISTS crossings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    line_name       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    video_time_s    REAL,
    media_time_utc  TEXT,
    frame_index     INTEGER,
    direction       TEXT,                        -- "in" | "out"
    speed_kmh       REAL,
    plate_text      TEXT,                        -- placa provisoria no instante do cruzamento
    object_class    TEXT,
    object_category TEXT
);

-- Indices para as consultas mais comuns de analise.
CREATE INDEX IF NOT EXISTS idx_events_session   ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_plate     ON events(plate_text);
CREATE INDEX IF NOT EXISTS idx_reads_session    ON reads(session_id, track_id);
CREATE INDEX IF NOT EXISTS idx_reads_frame      ON reads(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_frames_session   ON frames(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_dets_frame       ON detections(session_id, frame_index, kind);
CREATE INDEX IF NOT EXISTS idx_points_track     ON track_points(session_id, track_id);
CREATE INDEX IF NOT EXISTS idx_points_frame     ON track_points(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_tracks_session   ON tracks(session_id);
CREATE INDEX IF NOT EXISTS idx_tracks_plate     ON tracks(plate_text);
CREATE INDEX IF NOT EXISTS idx_crossings_line   ON crossings(session_id, line_name);
CREATE INDEX IF NOT EXISTS idx_tracks_class     ON tracks(session_id, object_class);
CREATE INDEX IF NOT EXISTS idx_tracks_category  ON tracks(session_id, object_category);
CREATE INDEX IF NOT EXISTS idx_events_class     ON events(session_id, object_class);
CREATE INDEX IF NOT EXISTS idx_events_track     ON events(session_id, track_id, revision);

-- Decisoes VIGENTES: por trilha, o evento de maior revisao. Sem revisoes
-- (publish_policy first_stable ou on_exit) e igual a tabela events.
CREATE VIEW IF NOT EXISTS v_decisions AS
SELECT e.*
FROM events e
WHERE e.revision = (SELECT MAX(e2.revision) FROM events e2
                    WHERE e2.session_id = e.session_id AND e2.track_id = e.track_id);

-- Uma linha por placa unica publicada, por sessao (decisoes vigentes).
CREATE VIEW IF NOT EXISTS v_unique_plates AS
SELECT session_id,
       plate_text,
       COUNT(*)            AS n_events,
       MIN(video_time_s)   AS first_video_time_s,
       MAX(video_time_s)   AS last_video_time_s,
       MIN(media_time_utc) AS first_media_time_utc,
       AVG(confidence)     AS avg_confidence,
       MAX(confidence)     AS max_confidence,
       AVG(speed_kmh)      AS avg_speed_kmh,
       GROUP_CONCAT(DISTINCT object_class) AS object_classes
FROM v_decisions
WHERE plate_text IS NOT NULL AND plate_text <> ''
GROUP BY session_id, plate_text;

-- Composicao do trafego: uma linha por classe de objeto observada.
-- Conta trilhas ENCERRADAS (end_reason preenchido): todo objeto confirmado,
-- com ou sem placa, inclusive os que sairam de cena antes do fim do video.
CREATE VIEW IF NOT EXISTS v_object_counts AS
SELECT session_id,
       object_class,
       object_category,
       COUNT(*)                                  AS n_tracks,
       SUM(CASE WHEN published = 1 THEN 1 ELSE 0 END) AS n_com_placa,
       ROUND(AVG(dwell_s), 2)                    AS dwell_medio_s,
       ROUND(AVG(speed_kmh), 1)                  AS vel_media_kmh,
       MAX(is_vulnerable)                        AS vulneravel
FROM tracks
WHERE end_reason <> ''
GROUP BY session_id, object_class, object_category;

-- Agregado por categoria, com a participacao percentual de cada uma.
CREATE VIEW IF NOT EXISTS v_category_share AS
SELECT session_id,
       object_category,
       COUNT(*) AS n_tracks,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY session_id), 2) AS pct,
       ROUND(AVG(speed_kmh), 1) AS vel_media_kmh
FROM tracks
WHERE end_reason <> ''
GROUP BY session_id, object_category;

-- Exposicao de usuarios vulneraveis: base para recortes de seguranca viaria.
CREATE VIEW IF NOT EXISTS v_vulnerable_exposure AS
SELECT t.session_id,
       t.object_class,
       COUNT(*)                     AS n_vulneraveis,
       ROUND(AVG(t.dwell_s), 2)     AS dwell_medio_s,
       (SELECT COUNT(*) FROM tracks m
         WHERE m.session_id = t.session_id AND m.end_reason <> ''
           AND m.object_category = 'veiculo_motorizado') AS n_motorizados,
       (SELECT ROUND(AVG(m.speed_kmh), 1) FROM tracks m
         WHERE m.session_id = t.session_id AND m.end_reason <> ''
           AND m.object_category = 'veiculo_motorizado') AS vel_media_motorizados
FROM tracks t
WHERE t.is_vulnerable = 1 AND t.end_reason <> ''
GROUP BY t.session_id, t.object_class;

-- Fluxo por linha virtual e classe: composicao do trafego por ponto de contagem.
CREATE VIEW IF NOT EXISTS v_line_composition AS
SELECT session_id,
       line_name,
       object_category,
       direction,
       COUNT(*)                 AS n,
       ROUND(AVG(speed_kmh), 1) AS vel_media_kmh
FROM crossings
GROUP BY session_id, line_name, object_category, direction;

-- Desempenho por sessao. A vazao e quadros/tempo (nao a media de uma media
-- movel), e a latencia e a PONTA A PONTA. Percentis (p95/p99) nao existem em
-- SQLite: calcule-os na analise a partir de frames.latency_ms.
CREATE VIEW IF NOT EXISTS v_session_performance AS
SELECT session_id,
       COUNT(*)                                        AS n_frames,
       CASE WHEN MAX(done_t_s) > MIN(done_t_s)
            THEN (COUNT(*) - 1) / (MAX(done_t_s) - MIN(done_t_s)) END AS throughput_fps,
       AVG(total_ms)                                   AS avg_service_ms,
       MAX(total_ms)                                   AS max_service_ms,
       AVG(latency_ms)                                 AS avg_latency_ms,
       MAX(latency_ms)                                 AS max_latency_ms,
       AVG(queue_ms)                                   AS avg_queue_ms,
       AVG(detect_ms)                                  AS avg_detect_ms,
       AVG(ocr_ms)                                     AS avg_ocr_ms,
       MAX(dropped_frames)                             AS dropped_frames,
       SUM(frames_skipped)                             AS frames_skipped,
       1.0 * MAX(dropped_frames) / (COUNT(*) + MAX(dropped_frames)) AS drop_rate,
       AVG(deadline_miss)                              AS deadline_miss_rate
FROM frames
GROUP BY session_id;
