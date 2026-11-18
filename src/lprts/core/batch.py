"""Processamento em lote de muitos videos: multiprocessos ou MPI.

## Onde cada forma de paralelismo se aplica

Este projeto tem dois problemas de paralelismo, com respostas diferentes:

======================  =====================================  ====================
problema                natureza                               ferramenta correta
======================  =====================================  ====================
Um video, ao vivo       estagios acoplados, estado compartilhado  threads (``core.parallel``)
Muitos videos, offline  tarefas independentes, sem estado comum   processos ou **MPI**
======================  =====================================  ====================

Dentro de um video, threads vencem: os estagios trocam quadros de dezenas de MB
e compartilham os pesos do modelo; serializar isso entre processos custaria mais
que o ganho. Entre videos, o problema e *embaracosamente paralelo* - cada video e
independente -, e ai processos e MPI sao a escolha certa, porque cada worker
carrega o proprio modelo e nao ha nada a compartilhar.

## Quando usar MPI

MPI so compensa quando os workers estao em **maquinas diferentes**: um cluster
HPC processando um corpus de centenas de horas de video. Numa unica maquina, o
backend ``process`` faz o mesmo sem a complexidade de um runtime MPI.

Uso num cluster com SLURM::

    srun -n 64 python -m lprts.core.batch --videos corpus/*.mp4 --backend mpi

O backend MPI usa despacho **dinamico** (mestre-trabalhador): o rank 0 entrega
uma tarefa por vez a quem pedir. Com particao estatica, um unico video longo
deixaria dezenas de nucleos ociosos no fim do lote.

## Resultados

Cada worker grava num banco proprio (``shard_<id>.db``) para evitar contencao de
escrita; ao final os fragmentos sao consolidados num unico arquivo. Uma sessao
por video, entao a proveniencia de cada leitura permanece rastreavel.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import AppConfig
from ..utils.logging import get_logger
from .pipeline import LPRPipeline

log = get_logger(__name__)

# Tabelas consolidadas na juncao dos fragmentos (ordem respeita a chave estrangeira).
_MERGE_TABLES = (
    "sessions", "tracks", "events", "reads", "frames", "crossings", "detections", "track_points",
)


@dataclass(slots=True)
class VideoResult:
    """Resumo do processamento de um video."""

    video: str
    ok: bool = False
    session_id: str = ""
    db_path: str = ""
    n_frames: int = 0
    n_events: int = 0
    n_unique_plates: int = 0
    counts_by_class: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    fps: float = 0.0
    worker: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class BatchResult:
    """Resultado agregado de um lote."""

    results: list[VideoResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    backend: str = ""
    n_workers: int = 0
    merged_db: str = ""

    @property
    def ok(self) -> list[VideoResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[VideoResult]:
        return [r for r in self.results if not r.ok]

    @property
    def total_frames(self) -> int:
        return sum(r.n_frames for r in self.ok)

    @property
    def aggregate_fps(self) -> float:
        """Vazao agregada do lote - a metrica que o paralelismo melhora."""
        return self.total_frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def counts_by_class(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for r in self.ok:
            for cls, n in r.counts_by_class.items():
                total[cls] = total.get(cls, 0) + n
        return dict(sorted(total.items(), key=lambda kv: kv[1], reverse=True))

    def summary(self) -> str:
        linhas = [
            f"Backend .............: {self.backend} ({self.n_workers} workers)",
            f"Videos ..............: {len(self.ok)} ok, {len(self.failed)} com falha",
            f"Quadros processados .: {self.total_frames}",
            f"Tempo total .........: {self.elapsed_s:.1f} s",
            f"Vazao agregada ......: {self.aggregate_fps:.1f} quadros/s",
        ]
        if self.merged_db:
            linhas.append(f"Banco consolidado ...: {self.merged_db}")
        if self.counts_by_class:
            comp = ", ".join(f"{k}={v}" for k, v in list(self.counts_by_class.items())[:8])
            linhas.append(f"Objetos por classe ..: {comp}")
        for r in self.failed:
            linhas.append(f"  FALHA {Path(r.video).name}: {r.error}")
        return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Unidade de trabalho
# ---------------------------------------------------------------------------

def process_one_video(
    video: str,
    config_dict: dict,
    db_path: str = "",
    worker: str = "",
    parallel_workers: int = 0,
    out_dir: str = "runs/lote",
) -> VideoResult:
    """Processa um video inteiro. Executa dentro de um worker (processo ou rank).

    Recebe a configuracao como dicionario porque precisa atravessar a fronteira
    do processo - dataclasses aninhadas nao sao serializaveis por padrao em
    todos os contextos de multiprocessing.
    """
    from ..factory import build_pipeline, build_source, build_store, close_session, open_session

    started = time.perf_counter()
    worker = worker or f"pid{os.getpid()}"
    # Sem caminho explicito, cada processo escolhe o proprio fragmento.
    db_path = db_path or _shard_path(Path(out_dir), worker)
    resultado = VideoResult(video=video, db_path=db_path, worker=worker)

    source = store = None
    pipeline: LPRPipeline | None = None
    session_started = False
    session_close_attempted = False
    try:
        cfg = AppConfig.from_dict(config_dict)
        cfg.source.uri = video
        cfg.source.realtime = False        # lote e sempre offline
        cfg.analytics.db_path = db_path

        source = build_source(cfg)
        store = build_store(cfg)

        if parallel_workers > 1:
            from ..factory import build_parallel_pipeline

            pipeline = build_parallel_pipeline(cfg, parallel_workers, store)

        else:
            pipeline = build_pipeline(cfg, store)

        open_session(store, cfg, source, pipeline, notes=f"lote:{resultado.worker}")
        session_started = True
        resultado.session_id = store.session_id

        n = 0
        for _ in pipeline.run(source):
            n += 1
        pipeline.finalize()
        session_close_attempted = True
        close_session(store, pipeline, source)

        m = pipeline.metrics.snapshot()
        resumo = store.session_summary()
        resultado.ok = True
        resultado.session_id = store.session_id
        resultado.n_frames = n
        resultado.n_events = int(resumo.get("n_events") or 0)
        resultado.n_unique_plates = int(resumo.get("n_unique_plates") or 0)
        resultado.counts_by_class = dict(m.counts_by_class)

    except Exception as exc:  # noqa: BLE001 - um video ruim nao aborta o lote
        resultado.error = f"{type(exc).__name__}: {exc}"
        if session_started and not session_close_attempted and store is not None and pipeline is not None:
            try:
                close_session(store, pipeline, source, end_reason="erro")
            except Exception as finish_exc:  # noqa: BLE001 - conservar a falha original
                resultado.error += (
                    f"; Erro ao finalizar sessao: {type(finish_exc).__name__}: {finish_exc}"
                )
        log.warning("Falha ao processar %s: %s", video, resultado.error)
    finally:
        for recurso in (pipeline, store, source):
            try:
                if recurso is not None:
                    if hasattr(recurso, "stop"):
                        recurso.stop()
                    elif hasattr(recurso, "close"):
                        recurso.close()
            except Exception as exc:  # noqa: BLE001 - continuar limpando os outros recursos
                resultado.ok = False
                detail = f"Erro ao encerrar {type(recurso).__name__}: {type(exc).__name__}: {exc}"
                resultado.error = f"{resultado.error}; {detail}" if resultado.error else detail
                log.warning("Falha ao processar %s: %s", video, detail)

    resultado.elapsed_s = time.perf_counter() - started
    resultado.fps = resultado.n_frames / resultado.elapsed_s if resultado.elapsed_s > 0 else 0.0
    return resultado


# ---------------------------------------------------------------------------
# Consolidacao dos fragmentos
# ---------------------------------------------------------------------------

def _copiable_columns(con: sqlite3.Connection, table: str) -> list[str]:
    """Colunas a copiar na juncao, EXCLUINDO a chave primaria autoincremental.

    Este detalhe e critico e nada obvio: cada fragmento numera suas linhas a
    partir de 1, entao um ``INSERT ... SELECT *`` levaria ids repetidos para o
    alvo. Com ``INSERT OR IGNORE``, as colisoes seriam descartadas **em
    silencio** - o banco consolidado sairia com uma fracao das linhas e nada
    indicaria o problema. Copiando apenas as demais colunas, o alvo gera ids
    novos e nenhuma linha se perde.
    """
    colunas = []
    for row in con.execute(f"PRAGMA table_info({table})"):
        nome, tipo, pk = row[1], (row[2] or "").upper(), row[5]
        if pk and tipo == "INTEGER":
            continue        # id autoincremental: deixa o alvo atribuir
        colunas.append(nome)
    return colunas


def merge_databases(shards: Sequence[str | Path], target: str | Path) -> Path:
    """Consolida os bancos dos workers num unico arquivo.

    Cada worker grava no proprio arquivo para evitar contencao de escrita - varios
    processos no mesmo SQLite se serializariam no bloqueio e anulariam o
    paralelismo. A juncao usa ``ATTACH`` + ``INSERT SELECT``, que roda no motor
    do SQLite, sem trazer as linhas para o Python.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    from ..analytics.store import ResultStore

    # Cria o alvo com o esquema completo (inclusive visoes e indices).
    ResultStore(target, async_writes=False).open().close()

    from ..analytics.store import SCHEMA_VERSION

    con = sqlite3.connect(str(target))
    # As chaves estrangeiras sao desligadas durante a juncao: as sessoes de um
    # fragmento sao inseridas antes das filhas, mas a ordem entre fragmentos nao
    # e garantida quando se insere tabela a tabela.
    con.execute("PRAGMA foreign_keys=OFF")
    total = 0
    for shard in shards:
        shard = Path(shard)
        if not shard.exists():
            continue
        with sqlite3.connect(str(shard)) as verif:
            versao = verif.execute("PRAGMA user_version").fetchone()[0]
        if versao != SCHEMA_VERSION:
            log.error(
                "Fragmento %s ignorado: esquema %s (esperado %s).", shard, versao, SCHEMA_VERSION
            )
            continue
        con.execute("ATTACH DATABASE ? AS shard", (str(shard),))
        try:
            for tabela in _MERGE_TABLES:
                colunas = _copiable_columns(con, tabela)
                lista = ", ".join(colunas)
                # `sessions` tem chave primaria de texto (session_id), unica por
                # execucao: ali OR IGNORE apenas evita duplicar uma sessao que
                # aparecesse em dois fragmentos.
                modo = "INSERT OR IGNORE" if tabela == "sessions" else "INSERT"
                cur = con.execute(
                    f"{modo} INTO main.{tabela} ({lista}) SELECT {lista} FROM shard.{tabela}"
                )
                total += max(cur.rowcount, 0)
            con.commit()
        except sqlite3.Error as exc:
            log.error("Falha ao consolidar %s: %s", shard, exc)
            con.rollback()
        finally:
            con.execute("DETACH DATABASE shard")
    con.execute("PRAGMA foreign_keys=ON")
    con.commit()
    con.close()
    log.info("Consolidados %d fragmentos (%d linhas) em %s", len(shards), total, target)
    return target


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _shard_path(out_dir: Path, ident: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"shard_{ident}.db")


def _init_worker(n_workers: int) -> None:
    """Preparo de cada processo do pool, antes da primeira tarefa.

    Roda com o metodo ``spawn``, logo acontece antes de OpenCV, PyTorch ou ONNX
    Runtime serem importados neste processo - que e exatamente quando as
    variaveis de thread ainda tem efeito.
    """
    from ..runtime import configure_all, limit_native_threads

    limit_native_threads(n_workers)
    configure_all()


def run_sequential(
    videos: list[str], config: AppConfig, out_dir: Path, parallel_workers: int = 0
) -> list[VideoResult]:
    """Referencia de comparacao: um video por vez."""
    cfg = config.to_dict()
    return [
        process_one_video(v, cfg, _shard_path(out_dir, "0"), "seq", parallel_workers)
        for v in videos
    ]


def run_multiprocess(
    videos: list[str],
    config: AppConfig,
    out_dir: Path,
    n_workers: int = 0,
    parallel_workers: int = 0,
) -> list[VideoResult]:
    """Paralelismo entre videos numa unica maquina, via processos.

    Processos e nao threads: cada worker carrega o proprio modelo e nao ha estado
    a compartilhar entre videos. O agendamento e dinamico por construcao - o
    ``ProcessPoolExecutor`` entrega a proxima tarefa a quem terminar primeiro,
    o que importa porque videos tem duracoes muito diferentes.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 1)
    cfg = config.to_dict()
    resultados: list[VideoResult] = []

    # ``spawn`` e nao ``fork``: o worker precisa comecar sem as bibliotecas
    # nativas carregadas, para que _init_worker consiga limitar as threads de
    # cada uma antes do primeiro uso (ver runtime.limit_native_threads).
    contexto = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=contexto,
        initializer=_init_worker,
        initargs=(n_workers,),
    ) as pool:
        # O fragmento e nomeado pelo PID do worker, resolvido dentro do proprio
        # processo: derivar do indice do video faria dois processos abrirem o
        # mesmo arquivo SQLite ao mesmo tempo.
        futuros = {
            pool.submit(
                process_one_video, v, cfg, "", "", parallel_workers, str(out_dir)
            ): v
            for v in videos
        }
        for fut in as_completed(futuros):
            try:
                resultados.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                resultados.append(
                    VideoResult(video=futuros[fut], error=f"{type(exc).__name__}: {exc}")
                )
            log.info("Lote: %d/%d concluidos", len(resultados), len(videos))
    return resultados


def run_mpi(
    videos: list[str], config: AppConfig, out_dir: Path, parallel_workers: int = 0
) -> list[VideoResult]:
    """Paralelismo entre nos de um cluster, via MPI (mestre-trabalhador dinamico).

    O rank 0 coordena e nao processa: entrega uma tarefa por vez a quem pedir.
    Com particao estatica (``videos[rank::size]``), um unico video longo deixaria
    dezenas de nucleos parados no fim do lote - o desbalanceamento e a regra, nao
    a excecao, num corpus de video real.

    Retorna a lista completa apenas no rank 0; os demais devolvem lista vazia.
    """
    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError(
            "Backend MPI exige mpi4py. Instale com: pip install mpi4py\n"
            "E execute com um lancador MPI, por exemplo:\n"
            "  mpirun -n 8 python -m lprts.core.batch --videos ... --backend mpi"
        ) from exc

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    TAG_PEDIDO, TAG_TAREFA, TAG_FIM, TAG_RESULTADO = 1, 2, 3, 4

    if size == 1:
        log.warning("MPI com um unico rank: equivale a execucao sequencial.")
        return run_sequential(videos, config, out_dir, parallel_workers)

    if rank == 0:
        # --- mestre: despacha e coleta ---
        proxima = 0
        encerrados = 0
        resultados: list[VideoResult] = []
        while encerrados < size - 1:
            status = MPI.Status()
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            origem, tag = status.Get_source(), status.Get_tag()

            if tag == TAG_RESULTADO:
                resultados.append(msg)
                log.info("MPI: %d/%d concluidos", len(resultados), len(videos))

            if proxima < len(videos):
                comm.send(videos[proxima], dest=origem, tag=TAG_TAREFA)
                proxima += 1
            else:
                comm.send(None, dest=origem, tag=TAG_FIM)
                encerrados += 1
        return resultados

    # --- trabalhador: pede, processa, devolve ---
    cfg = config.to_dict()
    db = _shard_path(out_dir, f"r{rank}")
    comm.send(None, dest=0, tag=TAG_PEDIDO)
    while True:
        status = MPI.Status()
        tarefa = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_FIM or tarefa is None:
            break
        resultado = process_one_video(tarefa, cfg, db, f"rank{rank}", parallel_workers)
        comm.send(resultado, dest=0, tag=TAG_RESULTADO)
    return []


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

BACKENDS = ("sequential", "process", "mpi")


def _avisar_se_contraproducente(config: AppConfig, n_workers: int) -> None:
    """Avisa quando um video por processo tende a ficar mais lento que em serie.

    Um worker que roda inferencia em PyTorch ja distribui a propria conta por
    todos os nucleos. N desses em paralelo nao somam vazao - disputam. Medido
    com 6 videos numa maquina de 4 nucleos: 209 s em 3 processos contra 74 s em
    serie. Vale avisar em vez de deixar o usuario concluir que o paralelismo
    nao funciona.
    """
    nucleos = os.cpu_count() or 4
    if n_workers <= 1 or nucleos > 8:
        return
    pesado = config.detector.detect_vehicles or config.detector.plate_detector == "yolo"
    if not pesado:
        return
    log.warning(
        "O detector em PyTorch ja usa todos os %d nucleos desta maquina; %d "
        "processos simultaneos tendem a ficar MAIS LENTOS que --backend "
        "sequential. Para medir, compare os dois; para ganhar escala de "
        "verdade, use --backend mpi com um video por no.",
        nucleos,
        n_workers,
    )


def run_batch(
    videos: Sequence[str | Path],
    config: AppConfig | None = None,
    out_dir: str | Path = "runs/lote",
    backend: str = "process",
    n_workers: int = 0,
    parallel_workers: int = 0,
    merge: bool = True,
) -> BatchResult:
    """Processa varios videos com o backend escolhido e consolida os resultados.

    ``parallel_workers > 1`` ativa tambem o paralelismo *dentro* de cada video
    (``core.parallel``). Combinar os dois so compensa quando ha mais nucleos que
    videos; caso contrario, um worker por video ja satura a maquina.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Backend desconhecido: {backend!r}. Opcoes: {BACKENDS}")

    config = config or AppConfig()
    lista = [str(v) for v in videos]
    if not lista:
        raise ValueError("Nenhum video informado.")

    # Cada execucao grava numa pasta propria. Antes os fragmentos ficavam
    # direto em out_dir e eram reaproveitados entre execucoes (o mesmo PID, ou o
    # mesmo "shard_seq.db"), e o banco consolidado misturava sessoes antigas.
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    if backend == "mpi":
        try:
            from mpi4py import MPI

            run_id = MPI.COMM_WORLD.bcast(run_id, root=0)
        except ImportError:
            pass
    out_dir = Path(out_dir) / run_id

    iniciado = time.perf_counter()
    if backend == "mpi":
        resultados = run_mpi(lista, config, out_dir, parallel_workers)
        try:
            from mpi4py import MPI

            if MPI.COMM_WORLD.Get_rank() != 0:
                return BatchResult(backend=backend)   # so o rank 0 consolida
            n_workers = MPI.COMM_WORLD.Get_size() - 1
        except ImportError:
            pass
    elif backend == "process":
        n_workers = n_workers or max(1, (os.cpu_count() or 4) - 1)
        _avisar_se_contraproducente(config, n_workers)
        resultados = run_multiprocess(lista, config, out_dir, n_workers, parallel_workers)
    else:
        n_workers = 1
        resultados = run_sequential(lista, config, out_dir, parallel_workers)

    lote = BatchResult(
        results=resultados,
        elapsed_s=time.perf_counter() - iniciado,
        backend=backend,
        n_workers=n_workers,
    )

    if merge and lote.ok:
        shards = sorted({r.db_path for r in lote.ok if r.db_path})
        alvo = out_dir / "consolidado.db"
        merge_databases(shards, alvo)
        lote.merged_db = str(alvo)
    return lote


def main() -> int:
    """Ponto de entrada para ``python -m lprts.core.batch`` (usado sob mpirun)."""
    import argparse
    import glob

    from ..utils.logging import setup_logging

    ap = argparse.ArgumentParser(description="Processamento em lote de videos (LPR-RTS)")
    ap.add_argument("--videos", nargs="+", required=True, help="arquivos ou padroes glob")
    ap.add_argument("--backend", default="process", choices=BACKENDS)
    ap.add_argument("--config", default=None, help="YAML de configuracao")
    ap.add_argument("--out", default="runs/lote", help="pasta dos resultados")
    ap.add_argument("--workers", type=int, default=0, help="processos (backend process)")
    ap.add_argument("--parallel-workers", type=int, default=0,
                    help="workers de deteccao DENTRO de cada video")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    arquivos: list[str] = []
    for padrao in args.videos:
        expandido = sorted(glob.glob(padrao))
        arquivos.extend(expandido or [padrao])

    config = AppConfig.load(args.config) if args.config else AppConfig()
    resultado = run_batch(
        arquivos, config, args.out, args.backend, args.workers,
        args.parallel_workers, merge=not args.no_merge,
    )
    if resultado.results or resultado.backend != "mpi":
        print(resultado.summary())
    return 0 if not resultado.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
