"""Persistencia dos resultados em SQLite, com escrita assincrona.

Por que SQLite e nao um servidor de banco: a POC precisa rodar num Mac sem
containers nem servicos; SQLite e da biblioteca padrao, e um unico arquivo
facil de versionar/compartilhar, e o DuckDB le esse arquivo diretamente para
analise. Em modo WAL ele sustenta com folga a taxa de escrita de um pipeline
de video.

Por que uma thread de escrita: num sistema de tempo real, ``INSERT`` sincrono no
laco de video introduz picos de latencia (o ``fsync`` pode custar dezenas de ms).
Aqui o pipeline apenas enfileira; uma thread dedicada grava em lotes dentro de
transacoes.

Integridade
-----------
* **Linhas de resultado nunca sao descartadas em silencio.** Eventos, trilhas,
  leituras e cruzamentos aguardam no maximo ``enqueue_timeout`` por espaco;
  saturacao ou falha da escritora provoca ``StorageWriteError`` e falha da
  sessao. Retorno do enqueue confirma admissao; ``flush`` confirma commit.
  So a telemetria volumosa
  (quadros, deteccoes, pontos de trilha) pode ser descartada se o disco nao
  acompanhar - e cada descarte e contado por tabela e gravado na sessao.
* **Lote com erro nao some inteiro.** Se uma transacao falha, as linhas sao
  regravadas uma a uma; erros em resultados criticos propagam ao produtor.
* **Versao do esquema.** Um banco criado por versao anterior e recusado: os
  dados antigos tem erros conhecidos (base de tempo, trilhas truncadas) e nao
  podem ser misturados com os novos.
* **Coordenadas.** Toda caixa e gravada em pixels da imagem ORIGINAL.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from ..core.types import Detection, FrameResult, PlateRead, TrackState
from ..utils.logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 4
APPLICATION_ID = 0x4C505254  # LPRT, SQLite application_id
# Versoes que migram para a atual SEM perda: so ganharam colunas com padrao que
# descreve exatamente o que elas gravavam. A 2 so conhecia a primeira decisao
# estavel - decision='stable', revision=0 sao os valores verdadeiros.
_MIGRACOES: dict[int, tuple[str, ...]] = {
    2: (
        "ALTER TABLE events ADD COLUMN decision TEXT NOT NULL DEFAULT 'stable'",
        "ALTER TABLE events ADD COLUMN revision INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE tracks ADD COLUMN first_published_text TEXT DEFAULT ''",
        "ALTER TABLE tracks ADD COLUMN n_revisions INTEGER DEFAULT 0",
        "ALTER TABLE tracks ADD COLUMN published_decision TEXT DEFAULT ''",
        "UPDATE tracks SET first_published_text = published_text, "
        "published_decision = CASE WHEN published = 1 THEN 'stable' ELSE '' END",
        "DROP VIEW IF EXISTS v_unique_plates",      # recriada sobre v_decisions
    ),
    3: ("ALTER TABLE events ADD COLUMN decision_elapsed_ms REAL",),
}

_SENTINEL = object()

# Tabelas cujas linhas sao o resultado em si: nunca descartadas por fila cheia.
_CRITICAS = frozenset({"sessions", "events", "tracks", "reads", "crossings"})


class SchemaVersionError(RuntimeError):
    """O banco foi criado por outra versao do esquema."""


class StorageWriteError(RuntimeError):
    """O resultado nao pode ser persistido; a sessao deve falhar explicitamente."""


def _is_results_database(con: sqlite3.Connection) -> bool:
    """Reconhece bancos antigos e atuais pelas colunas do dominio, nunca so pelo nome da tabela."""
    required = {
        "sessions": {"session_id", "started_at", "source_uri"},
        "events": {"session_id", "track_id", "plate_text"},
        "reads": {"session_id", "track_id", "text"},
    }
    app_id = con.execute("PRAGMA application_id").fetchone()[0]
    return app_id in (0, APPLICATION_ID) and all(
        columns <= {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for table, columns in required.items()
    )


class _FlushRequest:
    """Marcador enfileirado para forcar a gravacao do lote pendente.

    Esvaziar a fila NAO basta como sincronizacao: a thread escritora ja retirou
    os itens para um lote local que ainda nao foi commitado. Este marcador
    atravessa a fila na ordem e so libera quem espera depois do commit.
    """

    __slots__ = ("done",)

    def __init__(self) -> None:
        self.done = threading.Event()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse_utc(valor: str | datetime | None) -> datetime | None:
    if valor is None or valor == "":
        return None
    if isinstance(valor, datetime):
        dt = valor
    else:
        texto = str(valor).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(texto)
    if dt.tzinfo is None:
        raise ValueError(
            f"recording_start sem fuso horario: {valor!r}. Use ISO-8601 com fuso, "
            "ex.: 2026-03-14T08:00:00-03:00"
        )
    return dt.astimezone(timezone.utc)


def legacy_schema_version(db_path: str | Path) -> int | None:
    """Versao do esquema de um banco de OUTRA versao; ``None`` se for atual ou nao for banco."""
    try:
        with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as con:
            tem = _is_results_database(con)
            versao = con.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.Error:
        return None
    if tem and versao != SCHEMA_VERSION and versao not in _MIGRACOES:
        return int(versao or 1)
    return None


def archive_legacy_databases(
    pasta: str | Path = "runs", destino: str | Path | None = None
) -> list[tuple[Path, Path]]:
    """Move para ``<pasta>/legado/`` os bancos de outra versao do esquema.

    Nada e apagado: o banco (e seus arquivos -wal/-shm) so muda de lugar, com a
    versao e a data no nome. Devolve os pares (origem, destino) movidos.
    """
    pasta = Path(pasta)
    destino = Path(destino) if destino else pasta / "legado"
    movidos: list[tuple[Path, Path]] = []
    if not pasta.is_dir():
        return movidos
    carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")
    for db in sorted(pasta.glob("*.db")):
        versao = legacy_schema_version(db)
        if versao is None:
            continue
        destino.mkdir(parents=True, exist_ok=True)
        alvo = destino / f"{db.stem}.esquema{versao}.{carimbo}.db"
        for sufixo in ("", "-wal", "-shm"):
            origem = Path(str(db) + sufixo)
            if origem.exists():
                origem.rename(Path(str(alvo) + sufixo))
        movidos.append((db, alvo))
    return movidos


@dataclass(slots=True)
class ClearSummary:
    """O que ``clear_results`` removeu."""

    db_path: Path
    db_removed: bool = False
    n_sessions: int = 0
    n_crops: int = 0

    def describe(self) -> str:
        if not self.db_removed:
            return f"Nenhum dado anterior em {self.db_path}"
        return (f"Dados anteriores removidos: {self.db_path} ({self.n_sessions} sessao(oes)) "
                f"e {self.n_crops} recorte(s)")


def clear_results(db_path: str | Path, crops_dir: str | Path | None = None) -> ClearSummary:
    """Apaga um banco de resultados e os recortes das sessoes dele.

    So apaga um banco identificado pelo esquema de dominio do LPR-RTS,
    do esquema: um ``--db`` apontado por engano para o arquivo de outro programa
    gera erro, e nada e removido. Dos recortes, saem so os das sessoes do banco
    (o nome comeca pelo ``session_id``): a mesma pasta serve ao banco da
    interface e ao da avaliacao, e limpar um nao apaga os recortes do outro.
    """
    db = Path(db_path)
    resumo = ClearSummary(db)
    if not db.exists():
        return resumo
    sessoes: set[str] = set()
    if db.stat().st_size > 0:
        try:
            with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as con:
                if not _is_results_database(con):
                    raise sqlite3.DatabaseError("Esquema nao identificado")
                sessoes = {r[0] for r in con.execute("SELECT session_id FROM sessions")}
                if any(not isinstance(s, str) or not s for s in sessoes):
                    raise sqlite3.DatabaseError("Identificador de sessao invalido")
        except sqlite3.Error as exc:
            raise ValueError(
                f"{db} nao e um banco do LPR-RTS; para nao apagar dados de outro programa, "
                "nada foi removido. Use outro --db."
            ) from exc

    pasta = Path(crops_dir) if crops_dir else None
    if pasta is not None and pasta.is_dir() and sessoes:
        with os.scandir(pasta) as entradas:
            for entrada in entradas:
                if (entrada.is_file() and entrada.name.endswith(".jpg")
                        and entrada.name.split("_", 1)[0] in sessoes):
                    os.unlink(entrada.path)
                    resumo.n_crops += 1
    for sufixo in ("", "-wal", "-shm"):
        Path(str(db) + sufixo).unlink(missing_ok=True)
    resumo.db_removed = True
    resumo.n_sessions = len(sessoes)
    return resumo


def _load_schema() -> str:
    try:
        return resources.files("lprts.analytics").joinpath("schema.sql").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        return (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _orig(bbox: tuple[float, float, float, float] | None, scale: float) -> tuple[float, ...]:
    """Caixa em coordenadas processadas -> pixels da imagem original."""
    if bbox is None:
        return (None, None, None, None)  # type: ignore[return-value]
    s = scale if scale > 0 else 1.0
    return tuple(float(v) / s for v in bbox)


class ResultStore:
    """Gravador de resultados. Use como context manager para garantir o flush."""

    def __init__(
        self,
        db_path: str | Path,
        session_id: str | None = None,
        batch_size: int = 256,
        async_writes: bool = True,
        store_raw_reads: bool = True,
        store_frame_telemetry: bool = True,
        store_detections: bool = True,
        store_track_points: bool = True,
        queue_size: int = 50_000,
        enqueue_timeout: float = 0.1,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.batch_size = batch_size
        self.async_writes = async_writes
        self.store_raw_reads = store_raw_reads
        self.store_frame_telemetry = store_frame_telemetry
        self.store_detections = store_detections
        self.store_track_points = store_track_points
        if enqueue_timeout < 0:
            raise ValueError("enqueue_timeout deve ser nao negativo")
        self.enqueue_timeout = enqueue_timeout

        self._conn: sqlite3.Connection | None = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._session_started = False
        self._recording_start: datetime | None = None
        self._dropped: dict[str, int] = {}
        self._failed: dict[str, int] = {}
        self._write_error: BaseException | None = None
        self._writer_done = threading.Event()
        self._closing = False

    # -- ciclo de vida -----------------------------------------------------

    def open(self) -> ResultStore:
        if self._conn is not None:
            raise RuntimeError("Banco ja esta aberto")
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            self._initialize_schema()
        except BaseException:
            self._conn.rollback()
            self._conn.close()
            self._conn = None
            raise
        self._write_error = None
        self._writer_done.clear()
        self._closing = False
        if self.async_writes:
            self._thread = threading.Thread(target=self._writer_loop, name="db-writer", daemon=True)
            self._thread.start()
        log.info("Banco de resultados: %s (sessao %s)", self.db_path, self.session_id)
        return self

    def _initialize_schema(self) -> None:
        assert self._conn is not None
        existe = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
        versao = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if not existe:
            foreign_tables = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' LIMIT 1"
            ).fetchone()
            app_id = self._conn.execute("PRAGMA application_id").fetchone()[0]
            if foreign_tables or app_id not in (0, APPLICATION_ID):
                raise SchemaVersionError(f"{self.db_path} nao tem um esquema LPR-RTS reconhecido")
        if (existe and (versao in _MIGRACOES or versao == SCHEMA_VERSION)
                and not _is_results_database(self._conn)):
            raise SchemaVersionError(f"{self.db_path} nao tem um esquema LPR-RTS reconhecido")
        if existe and versao in _MIGRACOES:
            with self._conn:
                # O contexto da conexao sozinho NAO abre transacao para DDL no sqlite3.
                self._conn.execute("BEGIN IMMEDIATE")
                for version in range(versao, SCHEMA_VERSION):
                    for sql in _MIGRACOES[version]:
                        self._conn.execute(sql)
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            log.info("Banco %s migrado do esquema %d para o %d.", self.db_path, versao,
                     SCHEMA_VERSION)
            versao = SCHEMA_VERSION
        if existe and versao != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"O banco {self.db_path} foi criado por outra versao do LPR-RTS "
                f"(esquema {versao or 1}; esta versao usa {SCHEMA_VERSION}). Os dados das "
                "versoes anteriores tem erros conhecidos - base de tempo, trilhas "
                "truncadas, descartes nao gravados - e nao podem ser misturados com os "
                "novos. Mova ou apague o arquivo, ou use outro banco (--db)."
            )
        # WAL: leitores (a UI, um notebook) nao bloqueiam o escritor.
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL troca um fsync por transacao por durabilidade em queda de energia:
        # aceitavel para telemetria de video e varias vezes mais rapido.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_load_schema())
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        self._conn.commit()

    def close(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        if self.async_writes and self._thread is not None:
            if self._thread.is_alive() and not self._closing:
                try:
                    self._queue.put(_SENTINEL, timeout=max(0.0, deadline - time.monotonic()))
                except queue.Full as exc:
                    raise StorageWriteError("Fila de escrita saturada durante fechamento") from exc
                self._closing = True
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                # A conexao ainda esta em uso; o chamador pode repetir quando a escritora sair.
                raise StorageWriteError("Tempo esgotado esperando a escritora encerrar")
            self._thread = None
        if self._conn is not None:
            try:
                if self._write_error is not None and self._session_started:
                    # So depois de a escritora parar: nada mais disputa a conexao.
                    self._conn.execute(
                        "UPDATE sessions SET finished_at=?, end_reason='erro_persistencia', "
                        "stats_json=? WHERE session_id=?",
                        (_utcnow(), json.dumps({"rows_lost": self.rows_lost,
                                               "storage_error": str(self._write_error)}),
                         self.session_id),
                    )
                    self._session_started = False
                self._conn.commit()
            finally:
                self._conn.close()
                self._conn = None
        perdidas = self.rows_lost
        if perdidas:
            log.warning("Linhas nao gravadas por tabela: %s", perdidas)
        self._raise_write_error()

    def __enter__(self) -> ResultStore:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        try:
            self.finish_session()
        finally:
            self.close()

    @property
    def rows_lost(self) -> dict[str, int]:
        """Linhas perdidas por tabela (fila cheia ou erro de gravacao)."""
        total: dict[str, int] = dict(self._dropped)
        for tabela, n in self._failed.items():
            total[tabela] = total.get(tabela, 0) + n
        return total

    # -- fila --------------------------------------------------------------

    def _raise_write_error(self) -> None:
        if self._write_error is not None:
            raise StorageWriteError(f"Falha na persistencia: {self._write_error}") from self._write_error

    def _enqueue(self, table: str, sql: str, params: tuple) -> None:
        self._raise_write_error()
        if self._conn is None or self._closing:
            raise StorageWriteError("Banco fechado ou encerrando")
        if not self.async_writes:
            self._execute_now(sql, params, table)
            return
        try:
            if table in _CRITICAS:
                # Resultado nao se descarta: espera a escritora (que grava
                # milhares de linhas por segundo) antes de desistir.
                self._queue.put((table, sql, params), timeout=self.enqueue_timeout)
            else:
                # Telemetria nunca bloqueia o pipeline; o descarte e contado.
                self._queue.put_nowait((table, sql, params))
        except queue.Full as exc:
            self._dropped[table] = self._dropped.get(table, 0) + 1
            if table in _CRITICAS:
                self._write_error = StorageWriteError(f"Fila de escrita saturada: {table}")
                raise self._write_error from exc
        self._raise_write_error()

    def _execute_now(self, sql: str, params: tuple, table: str = "") -> None:
        assert self._conn is not None
        with self._lock:
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                self._failed[table] = self._failed.get(table, 0) + 1
                log.error("Falha ao gravar em %s: %s", table or "?", exc)
                if table in _CRITICAS:
                    self._write_error = exc
                    self._raise_write_error()

    def _writer_loop(self) -> None:
        assert self._conn is not None
        batch: list[tuple[str, str, tuple]] = []
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    item = None
                if item is _SENTINEL:
                    self._flush(batch)
                    return
                if isinstance(item, _FlushRequest):
                    self._flush(batch)
                    batch = []
                    item.done.set()
                    continue
                if item is not None:
                    batch.append(item)
                if len(batch) >= self.batch_size or (item is None and batch):
                    self._flush(batch)
                    batch = []
        except BaseException as exc:
            for table, _sql, _params in batch:
                self._failed[table] = self._failed.get(table, 0) + 1
            self._write_error = exc
            log.error("Escritora encerrada com erro: %s", exc)
        finally:
            # Linhas aceitas e pendentes nao podem mais ser gravadas apos a falha da
            # escritora: sao contadas, os marcadores liberados, e a falha chega ao chamador.
            while self._write_error is not None:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, tuple):
                    table = pending[0]
                    self._failed[table] = self._failed.get(table, 0) + 1
                elif isinstance(pending, _FlushRequest):
                    pending.done.set()
            self._writer_done.set()

    def _flush(self, batch: list[tuple[str, str, tuple]]) -> None:
        if not batch or self._conn is None:
            return
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                for _table, sql, params in batch:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                log.error("Falha ao gravar lote (%s); regravando linha a linha.", exc)
                critical_error: sqlite3.Error | None = None
                for table, sql, params in batch:
                    try:
                        self._conn.execute(sql, params)
                    except sqlite3.Error as exc_linha:
                        self._failed[table] = self._failed.get(table, 0) + 1
                        log.error("Linha descartada em %s: %s", table, exc_linha)
                        if table in _CRITICAS:
                            critical_error = exc_linha
                self._conn.commit()
                if critical_error is not None:
                    batch.clear()  # as linhas que falharam ja foram contadas acima
                    raise StorageWriteError("Falha ao gravar resultado critico") from critical_error
        batch.clear()

    def flush(self, timeout: float = 30.0) -> None:
        """Bloqueia ate tudo que foi enfileirado estar commitado no disco.

        Necessario antes de ler os proprios resultados (``query``), senao a
        leitura corre contra o lote ainda pendente na thread escritora.
        """
        self._raise_write_error()
        if not self.async_writes:
            return
        if self._thread is None or not self._thread.is_alive():
            return
        request = _FlushRequest()
        deadline = time.monotonic() + timeout
        try:
            self._queue.put(request, timeout=timeout)
        except queue.Full as exc:
            raise StorageWriteError("Fila de escrita saturada durante o flush") from exc
        while not request.done.is_set():
            self._raise_write_error()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StorageWriteError("Tempo esgotado esperando o flush do banco")
            request.done.wait(min(remaining, 0.05))
        self._raise_write_error()

    # -- sessao ------------------------------------------------------------

    def start_session(
        self,
        source_uri: str,
        source_fps: float = 0.0,
        source_width: int = 0,
        source_height: int = 0,
        frame_count: int = 0,
        realtime: bool = False,
        detector: str = "",
        plate_detector: str = "",
        ocr_engine: str = "",
        device: str = "",
        config: dict | None = None,
        environment: dict | None = None,
        notes: str = "",
        source_kind: str = "",
        processing_width: int = 0,
        processing_height: int = 0,
        recording_start_utc: str | datetime | None = None,
        deadline_ms: float | None = None,
        code_version: str = "",
        config_hash: str = "",
    ) -> str:
        """Registra a sessao. Grava config, ambiente e proveniencia para reprodutibilidade."""
        self._recording_start = _parse_utc(recording_start_utc)
        self._execute_now(
            """INSERT OR REPLACE INTO sessions
               (session_id, schema_version, started_at, source_uri, source_kind, source_fps,
                source_width, source_height, processing_width, processing_height, frame_count,
                recording_start_utc, realtime, deadline_ms, detector, plate_detector, ocr_engine,
                device, code_version, config_hash, config_json, environment_json, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.session_id, SCHEMA_VERSION, _utcnow(), str(source_uri), source_kind,
                float(source_fps), int(source_width), int(source_height),
                int(processing_width or source_width), int(processing_height or source_height),
                int(frame_count),
                self._recording_start.isoformat(timespec="milliseconds")
                if self._recording_start else None,
                int(realtime), float(deadline_ms) if deadline_ms else None,
                detector, plate_detector, ocr_engine, device, code_version, config_hash,
                json.dumps(config or {}, default=str, ensure_ascii=False),
                json.dumps(environment or {}, default=str, ensure_ascii=False),
                notes,
            ),
            "sessions",
        )
        self._session_started = True
        return self.session_id

    def finish_session(self, end_reason: str = "", stats: dict | None = None) -> None:
        if not self._session_started:
            return
        self.flush()
        dados = dict(stats or {})
        dados["rows_lost"] = self.rows_lost
        self._execute_now(
            """UPDATE sessions SET finished_at=?, end_reason=COALESCE(NULLIF(?, ''), end_reason),
                      stats_json=? WHERE session_id=?""",
            (_utcnow(), end_reason, json.dumps(dados, default=str, ensure_ascii=False),
             self.session_id),
            "sessions",
        )
        self._session_started = False

    def add_note(self, texto: str) -> None:
        """Anexa uma anotacao a sessao.

        Usado para registrar decisoes tomadas em tempo de execucao - como o
        fallback automatico para o modo so-placa. Sem isso, a configuracao
        gravada no banco descreveria um comportamento diferente do que de fato
        ocorreu, e o experimento deixaria de ser reproduzivel.
        """
        if not self._session_started or not texto:
            return
        self._execute_now(
            "UPDATE sessions SET notes = TRIM(COALESCE(notes,'') || ? ) WHERE session_id=?",
            (f"\n{texto}", self.session_id),
            "sessions",
        )

    def media_time_utc(self, video_time_s: float | None) -> str | None:
        """Instante real de um tempo de midia, se o inicio da gravacao e conhecido."""
        if self._recording_start is None or video_time_s is None:
            return None
        return (self._recording_start + timedelta(seconds=float(video_time_s))).isoformat(
            timespec="milliseconds"
        )

    # -- escrita de dominio ------------------------------------------------

    def add_event(
        self,
        track: TrackState,
        frame_index: int,
        video_time_s: float,
        fused: Any | None = None,
        crop_path: str = "",
        scale: float = 1.0,
        decision: str = "stable",
        revision: int = 0,
        decision_elapsed_ms: float | None = None,
    ) -> None:
        """Grava uma decisao sobre a placa de uma trilha (a "placa lida").

        ``fused`` e o resultado da fusao que motivou a publicacao; os campos de
        placa vem dele (e nao do estado mutavel da trilha), para que o evento
        descreva exatamente a decisao tomada.
        """
        x1, y1, x2, y2 = _orig(track.bbox, scale)
        if fused is not None:
            texto, layout = fused.text, fused.layout.value
            conf, acordo, n_reads = fused.confidence, fused.agreement, fused.n_reads
            valido = fused.valid_syntax
        else:
            texto, layout = track.plate_text or "", track.plate_layout.value
            conf, acordo, n_reads = track.plate_confidence, 0.0, len(track.reads)
            valido = False
        self._enqueue(
            "events",
            """INSERT INTO events
               (session_id, track_id, ts, video_time_s, media_time_utc, frame_index, plate_text,
                plate_layout, valid_syntax, confidence, agreement, n_reads, track_first_seen_s,
                object_class, object_category, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                speed_kmh, direction, crop_path, decision, revision, decision_elapsed_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.session_id, track.track_id, _utcnow(), float(video_time_s),
                self.media_time_utc(video_time_s), int(frame_index),
                texto or "", layout, int(valido), float(conf), float(acordo), int(n_reads),
                float(track.first_seen),
                track.object_class.value, track.category.value,
                x1, y1, x2, y2,
                track.speed_kmh, track.direction, crop_path, decision, int(revision),
                decision_elapsed_ms,
            ),
        )

    def add_read(
        self,
        track_id: int,
        read: PlateRead,
        frame_index: int | None = None,
        video_time_s: float | None = None,
        scale: float = 1.0,
    ) -> None:
        """Grava uma tentativa de OCR (nivel de quadro), inclusive sem texto.

        O quadro e o tempo gravados sao os do RECORTE (``read.frame_index``), nao
        os do quadro em que o resultado assincrono foi colhido.
        """
        if not self.store_raw_reads:
            return
        quadro = read.frame_index if read.frame_index >= 0 else (frame_index or 0)
        tempo = read.video_time_s if read.frame_index >= 0 else (video_time_s or 0.0)
        bx = _orig(read.bbox, scale) if read.bbox else (0.0, 0.0, 0.0, 0.0)
        largura = (bx[2] - bx[0]) if read.bbox else None
        altura = (bx[3] - bx[1]) if read.bbox else None
        self._enqueue(
            "reads",
            """INSERT INTO reads
               (session_id, track_id, frame_index, video_time_s, collected_frame, raw_text, text,
                n_corrections, confidence, char_confidences, layout, valid_syntax, engine,
                ocr_ms, wait_ms, sharpness, brightness, contrast, plate_w, plate_h,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.session_id, int(track_id), int(quadro), float(tempo),
                int(read.collected_frame), read.raw_text, read.text, int(read.n_corrections),
                float(read.confidence),
                json.dumps([round(float(c), 5) for c in read.char_confidences]),
                read.layout.value, int(read.valid_syntax), read.engine,
                float(read.ocr_ms), float(read.wait_ms), float(read.sharpness),
                float(read.brightness), float(read.contrast), largura, altura,
                float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]),
            ),
        )

    def add_frame_telemetry(
        self,
        result: FrameResult,
        fps: float,
        t0: float | None = None,
        start_t: float | None = None,
        done_t: float | None = None,
        deadline_ms: float | None = None,
        ocr_in_flight: int = 0,
    ) -> None:
        """Grava a telemetria de um quadro (tempos relativos ao 1o quadro da sessao)."""
        if not self.store_frame_telemetry:
            return
        t = result.timing
        base = t0 if t0 is not None else result.frame.timestamp

        def rel(v: float | None) -> float | None:
            return None if v is None else float(v - base)

        atrasado = (
            int(t.latency_ms > deadline_ms) if deadline_ms and t.latency_ms > 0 else None
        )
        self._enqueue(
            "frames",
            """INSERT INTO frames
               (session_id, frame_index, video_time_s, capture_t_s, start_t_s, done_t_s,
                detection_ran, n_object_dets, n_plate_dets, n_tracks, n_vehicle_tracks,
                n_person_tracks, n_rois_skipped, capture_ms, queue_ms, detect_ms, track_ms,
                ocr_ms, analytics_ms, total_ms, latency_ms, deadline_miss, ocr_in_flight, fps,
                dropped_frames, frames_skipped)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.session_id, int(result.frame.index), float(result.frame.source_pts),
                rel(result.frame.timestamp), rel(start_t), rel(done_t),
                int(result.detection_ran), len(result.vehicles), len(result.plates),
                len(result.tracks), result.n_vehicles, result.n_persons,
                int(result.extra.get("rois_skipped", 0)),
                t.capture_ms, t.queue_ms, t.detect_ms, t.track_ms, t.ocr_ms, t.analytics_ms,
                t.total_ms, t.latency_ms, atrasado, int(ocr_in_flight), float(fps),
                int(result.dropped_frames), int(result.frames_skipped),
            ),
        )

    def add_detection(
        self, frame_index: int, video_time_s: float, kind: str, det: Detection,
        scale: float = 1.0,
    ) -> None:
        if not self.store_detections:
            return
        x1, y1, x2, y2 = _orig(det.bbox, scale)
        self._enqueue(
            "detections",
            """INSERT INTO detections
               (session_id, frame_index, video_time_s, kind, label, class_id, score,
                x1, y1, x2, y2, track_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.session_id, int(frame_index), float(video_time_s), kind, det.label,
                int(det.class_id), float(det.score), x1, y1, x2, y2, det.track_id,
            ),
        )

    def add_track_point(
        self,
        track: TrackState,
        frame_index: int,
        video_time_s: float,
        plate_bbox: tuple[float, float, float, float] | None = None,
        scale: float = 1.0,
    ) -> None:
        if not self.store_track_points:
            return
        x1, y1, x2, y2 = _orig(track.bbox, scale)
        p1, q1, p2, q2 = _orig(plate_bbox, scale)
        self._enqueue(
            "track_points",
            """INSERT INTO track_points
               (session_id, track_id, frame_index, video_time_s, predicted, score,
                x1, y1, x2, y2, plate_x1, plate_y1, plate_x2, plate_y2)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.session_id, track.track_id, int(frame_index), float(video_time_s),
                int(track.time_since_update > 0), float(track.score),
                x1, y1, x2, y2, p1, q1, p2, q2,
            ),
        )

    def add_crossing(
        self,
        track: TrackState,
        line_name: str,
        direction: str,
        frame_index: int,
        video_time_s: float,
    ) -> None:
        self._enqueue(
            "crossings",
            """INSERT INTO crossings
               (session_id, track_id, line_name, ts, video_time_s, media_time_utc, frame_index,
                direction, speed_kmh, plate_text, object_class, object_category)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.session_id, track.track_id, line_name, _utcnow(), float(video_time_s),
                self.media_time_utc(video_time_s), int(frame_index), direction,
                track.speed_kmh, track.plate_text or "",
                track.object_class.value, track.category.value,
            ),
        )

    def upsert_track(
        self,
        track: TrackState,
        fused: Any | None = None,
        crop_path: str = "",
        scale: float = 1.0,
    ) -> None:
        """Grava/atualiza o resumo de uma trilha.

        Chamado quando a trilha TERMINA (``end_reason`` preenchido), com o estado
        final completo; e tambem na publicacao, como registro provisorio
        (``end_reason`` vazio) que a gravacao final substitui.
        """
        s = scale if scale > 0 else 1.0
        trajectory = [
            [round(x / s, 1), round(y / s, 1), round(t, 4), f]
            for (x, y), t, f in zip(track.centroids, track.timestamps, track.frames, strict=False)
        ]
        if fused is not None:
            texto, layout, conf = fused.text or "", fused.layout.value, fused.confidence
            acordo, valido, estavel = fused.agreement, fused.valid_syntax, fused.stable
        else:
            texto, layout, conf = (track.plate_text or ""), track.plate_layout.value, \
                track.plate_confidence
            acordo, valido, estavel = 0.0, False, track.plate_stable
        self._enqueue(
            "tracks",
            """INSERT INTO tracks
               (session_id, track_id, object_class, object_category, is_vulnerable,
                first_frame, last_frame, first_seen_s, last_seen_s, dwell_s, n_frames, n_reads,
                plate_text, plate_layout, plate_confidence, plate_agreement, plate_valid_syntax,
                plate_stable, published, published_text, published_frame, published_time_s,
                speed_kmh, direction, trajectory_json, crop_path, end_reason,
                first_published_text, n_revisions, published_decision)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id, track_id) DO UPDATE SET
                 object_class=excluded.object_class, object_category=excluded.object_category,
                 is_vulnerable=excluded.is_vulnerable,
                 last_frame=excluded.last_frame, last_seen_s=excluded.last_seen_s,
                 dwell_s=excluded.dwell_s, n_frames=excluded.n_frames, n_reads=excluded.n_reads,
                 plate_text=excluded.plate_text, plate_layout=excluded.plate_layout,
                 plate_confidence=excluded.plate_confidence,
                 plate_agreement=excluded.plate_agreement,
                 plate_valid_syntax=excluded.plate_valid_syntax,
                 plate_stable=excluded.plate_stable, published=excluded.published,
                 published_text=excluded.published_text,
                 published_frame=excluded.published_frame,
                 published_time_s=excluded.published_time_s,
                 speed_kmh=excluded.speed_kmh, direction=excluded.direction,
                 trajectory_json=excluded.trajectory_json,
                 crop_path=CASE WHEN excluded.crop_path <> '' THEN excluded.crop_path
                                ELSE tracks.crop_path END,
                 end_reason=excluded.end_reason,
                 first_published_text=excluded.first_published_text,
                 n_revisions=excluded.n_revisions,
                 published_decision=excluded.published_decision""",
            (
                self.session_id, track.track_id, track.object_class.value,
                track.category.value, int(track.is_vulnerable),
                track.first_frame, track.last_frame, float(track.first_seen),
                float(track.last_seen), float(track.dwell_s), int(track.hits),
                int(track.n_reads_total or len(track.reads)),
                texto, layout, float(conf), float(acordo), int(valido), int(estavel),
                int(track.published_text is not None), track.published_text or "",
                int(track.published_frame), track.published_time_s,
                track.speed_kmh, track.direction, json.dumps(trajectory), crop_path,
                track.end_reason, track.first_published_text or "", int(track.n_revisions),
                track.published_decision,
            ),
        )

    # -- leitura -----------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Consulta arbitraria (leitura), util em notebooks e na exportacao."""
        self.flush()
        assert self._conn is not None
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            return list(self._conn.execute(sql, params))

    def session_summary(self) -> dict[str, Any]:
        """Resumo agregado da sessao atual, para o fim da execucao."""
        rows = self.query(
            """SELECT
                 (SELECT COUNT(*) FROM v_decisions WHERE session_id=?)                  AS n_events,
                 (SELECT COUNT(DISTINCT plate_text) FROM v_decisions
                   WHERE session_id=? AND plate_text<>'')                               AS n_unique_plates,
                 (SELECT COUNT(*) FROM events WHERE session_id=? AND revision>0)        AS n_revisions,
                 (SELECT COUNT(*) FROM tracks WHERE session_id=?)                       AS n_tracks,
                 (SELECT COUNT(*) FROM reads WHERE session_id=?)                        AS n_reads,
                 (SELECT COUNT(*) FROM frames WHERE session_id=?)                       AS n_frames,
                 (SELECT throughput_fps FROM v_session_performance WHERE session_id=?)  AS throughput_fps,
                 (SELECT AVG(latency_ms) FROM frames WHERE session_id=?)                AS avg_latency_ms,
                 (SELECT MAX(dropped_frames) FROM frames WHERE session_id=?)            AS dropped
            """,
            (self.session_id,) * 9,
        )
        return dict(rows[0]) if rows else {}
