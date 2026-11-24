# Dados extraídos

Visão geral do banco e receitas de análise. A referência coluna a coluna está em
[DICIONARIO_DE_DADOS.md](DICIONARIO_DE_DADOS.md); o esquema executável, em
[`src/lprts/analytics/schema.sql`](../src/lprts/analytics/schema.sql).

---

## 1. Modelo de dados

```
sessions (1) ──┬── (N) frames        telemetria de cada quadro processado
               ├── (N) detections    cada detecção (objeto ou placa)
               ├── (N) track_points  cada trilha em cada quadro
               ├── (N) tracks        resumo final de cada trilha
               ├── (N) reads         cada tentativa de OCR (com ou sem texto)
               ├── (N) events        decisão publicada ← "a placa lida"
               │                      (v_decisions: só as vigentes)
               └── (N) crossings     cruzamentos de linhas virtuais
```

### Três convenções que não podem ser esquecidas

1. **Coordenadas em pixels da imagem original**, qualquer que seja
   `resize_width`. É o espaço do ground truth e da calibração.
2. **Tempo de mídia** (`video_time_s`, `*_seen_s`, `dwell_s`) é a base de toda
   medida física. O relógio de processamento (`frames.*_t_s`) só serve para
   vazão e latência.
3. **`ts` é a hora em que a linha foi produzida.** Para "a que horas o veículo
   passou", use `media_time_utc` — preenchido quando o início real da gravação
   é conhecido (câmera ao vivo: automático; vídeo gravado:
   `lprts run --recording-start 2026-03-14T08:00:00-03:00`).

### Por que guardar tudo, do bruto ao consolidado

`events` responde "que placas passaram" — com `fusion.publish_policy: revise`, cada
revisão é mais um evento; conte veículos e placas sobre **`v_decisions`** (a decisão
vigente de cada trilha), que nos demais casos é igual a `events`. `reads` guarda cada tentativa de OCR,
com confiança por caractere, nitidez e caixa; `detections` e `track_points`
guardam o que o detector e o rastreador produziram em cada quadro.

Com isso, quase toda análise é refeita **sem reprocessar o vídeo**: reavaliar
contra um ground truth corrigido (`lprts eval-db`), refazer a fusão com outros
parâmetros (`lprts tune`), medir o rastreamento, recalcular prazos perdidos para
outro requisito de latência.

Em execução 24/7 o volume cresce rápido; desligue o que não precisar
(`analytics.store_detections`, `store_track_points`, `store_raw_reads`).

### Versão do esquema

Bancos criados pela versão 0.1 (esquema 1) são recusados: seus dados têm erros
conhecidos (ver [CHANGELOG](../CHANGELOG.md)). Use um banco novo.

### Um banco por processamento

Cada processamento (`lprts run`, `lprts eval`, **Iniciar** na interface) apaga o
banco configurado e os recortes das sessões dele antes de gravar
(`analytics.clear_previous`, padrão `true`): uma consulta sem `WHERE session_id`
descreve o último vídeo processado, não a mistura de tudo o que já rodou. Só
bancos do LPR-RTS são apagados; um `--db` apontado para outro arquivo é erro. Para
juntar vídeos num banco: `--keep-previous`, ou `lprts batch` (um banco consolidado
por execução do lote).

Com câmera ao vivo ou em execução contínua, isso quer dizer que **reiniciar o
processo apaga o histórico**: use `--keep-previous` ou `analytics.clear_previous:
false` no perfil (os `tempo_real*.yaml` não o fazem).

---

## 2. Consultas úteis

### Composição do tráfego

```sql
-- Quem passou por aqui, e em que proporção (trilhas encerradas)
SELECT object_category,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
       ROUND(AVG(speed_kmh), 1) AS vel_media,
       ROUND(AVG(dwell_s), 1)   AS permanencia_s
FROM tracks WHERE end_reason <> ''
GROUP BY object_category ORDER BY n DESC;
```

### Fluxo por hora do dia (exige o início real da gravação)

```sql
SELECT strftime('%Y-%m-%d %H', media_time_utc) AS hora,
       COUNT(*) AS veiculos,
       COUNT(DISTINCT plate_text) AS placas_unicas,
       ROUND(AVG(speed_kmh), 1) AS vel_media
FROM v_decisions
WHERE plate_text <> '' AND media_time_utc IS NOT NULL
GROUP BY hora ORDER BY hora;
```

### Exposição de usuários vulneráveis

```sql
-- Pedestres, ciclistas e motociclistas por janela de 5 minutos de vídeo
SELECT CAST(first_seen_s / 300 AS INT) AS janela_5min,
       SUM(object_category = 'pessoa')              AS pedestres,
       SUM(object_class = 'bicicleta')              AS ciclistas,
       SUM(object_class = 'motocicleta')            AS motos,
       SUM(object_category = 'veiculo_motorizado')  AS motorizados
FROM tracks WHERE end_reason <> ''
GROUP BY janela_5min ORDER BY janela_5min;
```

### Desempenho de tempo real

```sql
-- Latência ponta a ponta, fila e prazos por bloco de 10 s de processamento
SELECT CAST(done_t_s / 10 AS INT) AS bloco_10s,
       COUNT(*)                   AS quadros,
       ROUND(AVG(latency_ms), 2)  AS lat_media_ms,
       ROUND(MAX(latency_ms), 2)  AS lat_max_ms,
       ROUND(AVG(queue_ms), 2)    AS fila_media_ms,
       SUM(frames_skipped)        AS descartados,
       ROUND(AVG(deadline_miss), 4) AS taxa_prazo_perdido
FROM frames WHERE session_id = :sessao
GROUP BY bloco_10s ORDER BY bloco_10s;
```

Percentis (p95, p99) não existem no SQLite — calcule-os em pandas/R/DuckDB:

```python
import duckdb
con = duckdb.connect()
con.execute("INSTALL sqlite; LOAD sqlite; ATTACH 'runs/avaliacao.db' AS lpr (TYPE sqlite);")
con.sql("""
    SELECT session_id,
           quantile_cont(latency_ms, 0.50) AS p50,
           quantile_cont(latency_ms, 0.95) AS p95,
           quantile_cont(latency_ms, 0.99) AS p99
    FROM lpr.frames GROUP BY session_id
""").show()
```

### Onde o OCR erra

```sql
SELECT engine,
       COUNT(*)                                        AS tentativas,
       ROUND(AVG(text = ''), 3)                        AS taxa_sem_texto,
       ROUND(AVG(confidence), 3)                       AS conf_media,
       ROUND(AVG(valid_syntax), 3)                     AS taxa_sintaxe_ok,
       ROUND(AVG(n_corrections > 0), 3)                AS taxa_corrigida,
       ROUND(AVG(ocr_ms), 2)                           AS ms_medio
FROM reads GROUP BY engine;
```

Acerto (e não só sintaxe) exige o ground truth: use a tabela
`<relatorio>_leituras.csv` gerada pela avaliação, que já traz `correct` e `cer`
por tentativa.

### Contribuição da gramática

```sql
-- Quantas leituras a gramática alterou, e quantas tornou "válidas"
SELECT engine,
       ROUND(AVG(raw_text <> text), 3)                 AS taxa_alterada,
       ROUND(AVG(n_corrections), 3)                    AS correcoes_medias
FROM reads WHERE text <> '' GROUP BY engine;
```

### Latência de decisão

```sql
-- Quanto tempo de vídeo entre o primeiro avistamento e a publicação
-- (revision = 0: a primeira decisão de cada trilha; as revisões vêm depois)
SELECT ROUND(video_time_s - track_first_seen_s, 2) AS atraso_s, n_reads, plate_text,
       decision
FROM events WHERE revision = 0 ORDER BY atraso_s;
```

### Decisão online × fusão final

```sql
-- Trilhas cuja placa publicada difere da fusão sobre todas as leituras
SELECT track_id, published_text, plate_text, n_reads
FROM tracks WHERE published = 1 AND published_text <> plate_text;
```

---

## 3. Análise externa

### DuckDB

```python
import duckdb
con = duckdb.connect()
con.execute("INSTALL sqlite; LOAD sqlite;")
con.execute("ATTACH 'runs/avaliacao.db' AS lpr (TYPE sqlite);")
df = con.sql("SELECT * FROM lpr.v_decisions").df()
```

### pandas

```python
import sqlite3, pandas as pd
con = sqlite3.connect("runs/avaliacao.db")
leituras = pd.read_sql_query("SELECT * FROM reads", con)
quadros = pd.read_sql_query("SELECT * FROM frames", con)
```

### R

```r
library(DBI)
con <- dbConnect(RSQLite::SQLite(), "runs/avaliacao.db")
eventos <- dbGetQuery(con, "SELECT * FROM v_decisions")
```

### Trajetórias

```python
import json, sqlite3
con = sqlite3.connect("runs/avaliacao.db")
for tid, traj in con.execute("SELECT track_id, trajectory_json FROM tracks"):
    pontos = json.loads(traj)        # [[x, y, t_s, quadro], ...] (pixels originais)
    print(tid, len(pontos), "pontos")
# Trajetória completa, quadro a quadro:
#   SELECT * FROM track_points WHERE track_id = ? ORDER BY frame_index
```

---

## 4. Tabelas e figuras para o artigo

```bash
lprts report --db runs/lprts.db --out runs/relatorio
lprts report --db a.db --db b.db --min-dwell 1.0 --no-figures
```

Lê todas as sessões dos bancos e gera `videos.csv`, `objetos.csv`, `funil.csv`,
`ocr_largura.csv`, `desempenho.csv`, o resumo `relatorio.md` e as figuras
`fig_composicao`, `fig_latencia`, `fig_estagios`, `fig_ocr_largura` (PDF + PNG). As
tabelas não têm textos de placa e podem acompanhar o artigo. Duas escolhas ficam
explícitas:

- **Objeto = trilha confirmada** (permanência ≥ `--min-dwell`, 0,5 s por padrão).
  Em vídeo de rua, metade das trilhas de carro dura menos de 1 s: são fragmentos do
  mesmo veículo perdido numa oclusão, e a contagem bruta superestima o tráfego.
  `objetos.csv` traz as duas contagens.
- **Funil das placas por veículo confirmado**: placa detectada em algum quadro,
  alguma tentativa de OCR, alguma leitura com sintaxe válida, placa publicada. É o
  que mostra onde o sistema perde cada veículo (distância, OCR, fusão).

São medidas do sistema, não acurácia: quantas placas publicadas estão certas e
quantos objetos passaram exigem ground truth (`lprts eval`, [AVALIACAO.md](AVALIACAO.md)).

## 5. Exportar

```bash
lprts export --db runs/avaliacao.db                              # a sessao mais recente
lprts export --db runs/avaliacao.db --session <id> --format parquet
lprts export --db runs/avaliacao.db --session <id> --format json  # so o resumo
lprts export --db runs/avaliacao.db --all                        # todas as sessoes
```

Por padrão sai **uma sessão** — um vídeo —, em `runs/export/<vídeo>_<sessão>/`:
todas as tabelas e visões filtradas por ela e o resumo `sessao.json`. Com `--all`
saem todas as sessões juntas (o `track_id` recomeça a cada sessão: agrupe sempre
por `session_id, track_id`).

---

## 6. Privacidade

Placas são **dados pessoais indiretos**: identificam um veículo e, por
consequência, uma pessoa. Considerações para pesquisa com dados reais:

- No Brasil, o tratamento está sujeito à **LGPD** (Lei 13.709/2018). Pesquisa
  acadêmica tem base legal própria (art. 7º, IV), mas isso **não dispensa**
  minimização, finalidade declarada e segurança.
- Submeta o protocolo ao **Comitê de Ética em Pesquisa** da instituição antes da coleta.
- Para publicar, **pseudonimize**: `SHA-256(placa + sal_secreto)` preserva a
  capacidade de contar veículos únicos sem revelar identidades. O sal deve ser
  secreto — sem ele, o espaço de placas brasileiras é pequeno o bastante para
  ataque de força bruta.
- `analytics.save_crops: false` desliga o salvamento de imagens, que são o dado
  mais sensível gravado.
- Defina e documente uma política de retenção; não guarde indefinidamente.

```python
import hashlib, os, sqlite3

SAL = os.environ["LPR_SAL"]          # fora do banco e fora do repositório

def pseudonimo(placa: str) -> str:
    """Estável entre sessões, mas não reversível sem o sal."""
    return hashlib.sha256((placa + SAL).encode()).hexdigest()[:16]

con = sqlite3.connect("runs/avaliacao.db")
for placa, t, vel, classe in con.execute(
    "SELECT plate_text, video_time_s, speed_kmh, object_class FROM v_decisions WHERE plate_text <> ''"
):
    print(pseudonimo(placa), t, vel, classe)   # publicável
```

### Os dados do artigo

Os dados do artigo, no Zenodo (<https://doi.org/10.5281/zenodo.24351664>), não são pseudonimizados: os vídeos, os
gabaritos e os bancos das sessões trazem o texto das placas, porque a reprodução depende deles.
As condições de uso estão no registro.
