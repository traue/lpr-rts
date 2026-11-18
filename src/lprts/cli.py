"""Interface de linha de comando do LPR-RTS.

Comandos::

    lprts doctor               # verifica se a maquina esta pronta para executar
    lprts diagnose             # mostra em que estagio o pipeline para, num video
    lprts info                 # ambiente, dispositivo e motores disponiveis
    lprts demo                 # gera video sintetico e executa tudo (sem downloads)
    lprts run                  # executa o pipeline (com ou sem interface)
    lprts eval                 # executa e avalia contra ground truth
    lprts eval-db              # reavalia uma sessao ja gravada (sem reprocessar)
    lprts compare              # compara dois relatorios nos mesmos veiculos
    lprts export               # exporta o banco para CSV/Parquet/JSON
    lprts report               # tabelas e figuras descritivas para o artigo
    lprts sessions             # lista as sessoes gravadas
    lprts bench                # mede desempenho isolando cada estagio
    lprts calibrate            # desenha linhas virtuais e calibra a homografia
    lprts batch                # processa muitos videos (multiprocessos ou MPI)
    lprts tune                 # otimiza hiperparametros (fronteira de Pareto)
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table

from .config import AppConfig, ConfigError
from .utils.logging import get_logger, setup_logging

app = typer.Typer(
    name="lprts",
    help="LPR-RTS - reconhecimento de placas veiculares em tempo real (POC de doutorado).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
log = get_logger(__name__)


def _load_config(config_paths: list[Path] | Path | None, **overrides) -> AppConfig:
    """Carrega um ou mais YAMLs (os posteriores sobrepoem) e aplica as opcoes da CLI."""
    if isinstance(config_paths, Path):
        config_paths = [config_paths]
    try:
        cfg = AppConfig.load_many(config_paths) if config_paths else AppConfig()
    except ConfigError as exc:
        console.print(str(exc), style="red", markup=False, highlight=False)
        raise SystemExit(2) from None
    cfg.override(**overrides)
    return cfg


def _print_kv(title: str, data: dict) -> None:
    table = Table(title=title, show_header=False, box=None, title_justify="left")
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")
    for key, value in data.items():
        table.add_row(str(key), str(value))
    console.print(table)


def _fail(exc: BaseException, code: int = 2) -> None:
    console.print(str(exc), style="red", markup=False, highlight=False)
    raise SystemExit(code) from None


CONFIG_OPT = typer.Option(
    None, "--config", "-c",
    help="YAML de configuracao (repita para combinar: o posterior sobrepoe o anterior).",
)
SCOPE_OPT = typer.Option(
    0.0, "--scope-min-width", min=0.0,
    help="Avaliar so placas com pelo menos esta largura, em px da imagem original: as "
         "anotadas mais estreitas contam como difficult e o que o sistema fizer com placas "
         "mais estreitas e sem anotacao e ignorado. 0 = avaliar tudo.",
)
CLEAR_OPT = typer.Option(
    None, "--clear-previous/--keep-previous",
    help="Apagar o banco (--db) e os recortes das execucoes anteriores ao iniciar. "
         "Sem a opcao, vale o YAML (analytics.clear_previous, padrao: apagar).",
)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@app.command()
def doctor(
    camera: bool = typer.Option(False, "--camera", help="Testar tambem o acesso a webcam."),
    rapido: bool = typer.Option(False, "--rapido", help="Pular o teste de fumaca do pipeline."),
) -> None:
    """Verifica se o ambiente esta pronto: dependencias, GPU, modelos e permissoes."""
    setup_logging("ERROR")
    from .doctor import ERRO, OK, ambiente, proximo_passo, resumo, run_all

    console.print()
    _print_kv("Ambiente", ambiente())
    console.print()

    with console.status("[cyan]Verificando...[/cyan]"):
        checks = run_all(testar_camera=camera, rapido=rapido)

    simbolos = {
        OK: ("[green]  OK  [/green]", "white"),
        "aviso": ("[yellow] AVISO[/yellow]", "yellow"),
        ERRO: ("[red] ERRO [/red]", "red"),
    }
    tabela = Table(title="Diagnostico", show_lines=False)
    tabela.add_column("", width=6)
    tabela.add_column("Verificacao", style="cyan", no_wrap=True)
    tabela.add_column("Estado")
    for c in checks:
        simbolo, cor = simbolos[c.estado]
        from rich.text import Text

        tabela.add_row(simbolo, c.nome, Text(c.detalhe, style=cor))
    console.print(tabela)

    pendencias = [c for c in checks if c.estado != OK and c.solucao]
    if pendencias:
        console.print()
        console.print("[bold]Como resolver:[/bold]")
        for c in pendencias:
            console.print(f"  [cyan]{c.nome}[/cyan]")
            # markup=False: as solucoes contem os extras do pip entre colchetes
            # ('.[dl]'), que o rich interpretaria como marcacao.
            console.print(f"    {c.solucao}", markup=False, highlight=False)

    n_ok, n_aviso, n_erro = resumo(checks)
    console.print()
    console.print(
        f"[green]{n_ok} ok[/green]  [yellow]{n_aviso} aviso(s)[/yellow]  "
        f"[red]{n_erro} erro(s)[/red]"
    )
    console.print()
    console.print(proximo_passo(checks), soft_wrap=True, highlight=False)
    raise SystemExit(1 if n_erro else 0)


@app.command()
def diagnose(
    source: str = typer.Option(..., "--source", "-s", help="Video ou camera a analisar."),
    config: list[Path] | None = CONFIG_OPT,
    frames: int = typer.Option(150, "--frames", "-n", help="Quantos quadros analisar."),
    plate_detector: str | None = typer.Option(None, "--plate-detector"),
    vehicle_detector: bool | None = typer.Option(
        None, "--vehicle-detector/--no-vehicle-detector",
        help="Sem a opcao, vale o que o YAML diz (detector.detect_vehicles).",
    ),
    ocr_engine: str | None = typer.Option(None, "--ocr"),
    classes: str | None = typer.Option(None, "--classes"),
) -> None:
    """Mostra em que estagio do pipeline o fluxo para (tabela de placas vazia)."""
    setup_logging("ERROR")
    from .diagnose import relatorio, run_diagnosis

    cfg = _load_config(
        config,
        **{
            "source.uri": source,
            "detector.plate_detector": plate_detector,
            "detector.detect_vehicles": vehicle_detector,
            "detector.classes": classes,
            "ocr.engine": ocr_engine,
        },
    )
    try:
        with console.status(f"[cyan]Analisando {frames} quadros...[/cyan]"):
            d = run_diagnosis(cfg, max_frames=frames)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _fail(exc)
    console.print(relatorio(d, source), soft_wrap=True, highlight=False)
    raise SystemExit(0 if d.ok else 1)


@app.command()
def info() -> None:
    """Mostra o ambiente, o dispositivo de inferencia e os motores disponiveis."""
    from .detect.backends import resolve_device
    from .ocr.engines import available_engines
    from .provenance import code_version, hardware, opencv_conflict, power_state

    setup_logging("WARNING")
    device = resolve_device("auto")
    hw = hardware()
    _print_kv(
        "Ambiente",
        {
            "Processador": hw.get("cpu_model") or hw.get("machine"),
            "Nucleos (logicos)": hw.get("logical_cpus"),
            "Memoria (GB)": f"{(hw.get('memory_bytes') or 0) / 1e9:.1f}",
            "Energia": power_state() or "-",
            "Codigo": code_version(),
        },
    )
    console.print()
    _print_kv(
        "Inferencia",
        {
            "Preferencia resolvida": f"{device.kind} - {device.description}",
            "Providers ONNX": ", ".join(device.onnx_providers),
            "Motores de OCR": ", ".join(available_engines()),
            "Observacao": (
                "com pesos .pt o YOLO roda em PyTorch (MPS/CUDA/CPU); so .mlpackage "
                "roda no CoreML. O dispositivo EFETIVO de cada componente e gravado na sessao."
            ),
        },
    )
    if opencv_conflict():
        console.print(
            "\n[yellow]opencv-python e opencv-python-headless estao instalados juntos "
            "(conflito). Veja `lprts doctor`.[/yellow]"
        )
    console.print()
    for label, path in (
        ("Pesos de veiculo", "data/models/yolo11n.pt"),
        ("Pesos de placa", "data/models/plate_yolo.pt"),
    ):
        exists = Path(path).exists()
        mark = "[green]presente[/green]" if exists else "[yellow]ausente[/yellow]"
        console.print(f"  {label}: {path} - {mark}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    source: str = typer.Option("0", "--source", "-s", help="Video, pasta, indice de webcam ou URL RTSP."),
    config: list[Path] | None = CONFIG_OPT,
    realtime: bool | None = typer.Option(
        None, "--realtime/--offline",
        help="Descartar quadros para manter a latencia. Sem a opcao, vale o YAML.",
    ),
    ui: bool | None = typer.Option(
        None, "--ui/--headless", help="Abrir a interface grafica (padrao: ui.enabled do YAML)."
    ),
    ocr_engine: str | None = typer.Option(None, "--ocr", help="auto|fastplate|paddle|easyocr|tesseract|template"),
    plate_detector: str | None = typer.Option(None, "--plate-detector", help="auto|yolo|classic"),
    vehicle_detector: bool | None = typer.Option(
        None, "--vehicle-detector/--no-vehicle-detector",
        help="Sem a opcao, vale o que o YAML diz (detector.detect_vehicles).",
    ),
    device: str | None = typer.Option(None, "--device", help="auto|coreml|mps|cuda|cpu"),
    db: Path | None = typer.Option(None, "--db", help="Caminho do banco de resultados."),
    max_frames: int = typer.Option(0, "--max-frames", help="Parar apos N quadros (0 = sem limite)."),
    width: int | None = typer.Option(None, "--width", help="Largura de processamento."),
    save_video: Path | None = typer.Option(None, "--save-video", help="Gravar o video anotado."),
    parallel: int | None = typer.Option(
        None, "--parallel", "-j",
        help="Workers em paralelo: 0 = serial, -1 = automatico. Sem a opcao, vale "
             "detector.workers do YAML (automatico: paralelo offline, serial em tempo real).",
    ),
    classes: str | None = typer.Option(
        None, "--classes", help="Classes a detectar: veiculos | trafego | completo."
    ),
    recording_start: str | None = typer.Option(
        None, "--recording-start",
        help="Instante real do 1o quadro (ISO-8601 com fuso), para videos gravados.",
    ),
    clear_previous: bool | None = CLEAR_OPT,
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Executa o pipeline de reconhecimento de placas."""
    setup_logging(log_level)
    cfg = _load_config(
        config,
        **{
            "source.uri": source,
            "source.realtime": realtime,
            "source.max_frames": max_frames,
            "source.resize_width": width,
            "source.recording_start": recording_start,
            "ocr.engine": ocr_engine,
            "detector.plate_detector": plate_detector,
            "detector.detect_vehicles": vehicle_detector,
            "detector.device": device,
            "analytics.db_path": str(db) if db else None,
            "detector.classes": classes,
            "analytics.clear_previous": clear_previous,
            "ui.enabled": ui,
            "log_level": log_level,
        },
    )

    # Erros de configuracao previsiveis viram mensagem, nao traceback.
    try:
        from .factory import build_plate_detector

        build_plate_detector(cfg).close()
    except (FileNotFoundError, ValueError) as exc:
        _fail(exc)

    if cfg.ui.enabled:
        try:
            from .ui.app import launch_gui

            raise SystemExit(launch_gui(cfg))
        except ImportError as exc:
            console.print(
                f"[yellow]Interface grafica indisponivel ({exc}). "
                "Instale com: pip install '.[ui]'. Seguindo sem interface.[/yellow]"
            )
    raise SystemExit(_run_headless(cfg, save_video, parallel))


def _build_runtime(cfg: AppConfig, parallel: int | None = None, clear: bool = False):
    """Valida fonte e modelos antes de substituir resultados anteriores.

    ``parallel`` None = ``detector.workers`` do YAML; 0 = serial; -1 = automatico.
    """
    from .core.parallel import auto_workers
    from .factory import (
        build_configured_pipeline,
        build_parallel_pipeline,
        build_pipeline,
        build_source,
        build_store,
        check_calibration,
        clear_previous_results,
    )

    with ExitStack() as cleanup:
        source = build_source(cfg)
        cleanup.callback(source.close)
        check_calibration(cfg, source)
        if parallel is None:
            pipeline = build_configured_pipeline(cfg)
        elif parallel:
            pipeline = build_parallel_pipeline(cfg, auto_workers(cfg) if parallel < 0 else parallel)
        else:
            pipeline = build_pipeline(cfg)
        cleanup.callback(pipeline.stop)
        pipeline.configure_for_source(source)
        if clear:
            limpeza = clear_previous_results(cfg)
            if limpeza is not None and limpeza.db_removed:
                console.print(f"[yellow]{limpeza.describe()}.[/yellow]", highlight=False)
        store = build_store(cfg)
        cleanup.callback(store.close)
        pipeline.store = store
        cleanup.pop_all()
        return source, store, pipeline


def _run_headless(cfg: AppConfig, save_video: Path | None = None,
                  parallel: int | None = None) -> int:
    """Execucao sem interface: util para lotes, servidores e medicao limpa."""
    import cv2

    from .analytics.store import SchemaVersionError
    from .factory import close_session, describe_pipeline, open_session
    from .ocr.engines import EngineUnavailableError
    from .ui.overlay import draw_overlay

    try:
        source, store, pipeline = _build_runtime(cfg, parallel, clear=True)
    except (SchemaVersionError, EngineUnavailableError, FileNotFoundError, ValueError,
            RuntimeError) as exc:
        _fail(exc)

    with ExitStack() as cleanup:
        cleanup.callback(source.close)
        cleanup.callback(store.close)
        cleanup.callback(pipeline.stop)
        _print_kv("Configuracao efetiva", describe_pipeline(cfg, pipeline))
        open_session(store, cfg, source, pipeline)

        writer = None
        started = time.perf_counter()
        n = 0
        motivo = "fim_da_fonte"
        try:
            with console.status("[cyan]Processando...[/cyan]") as status:
                for result in pipeline.run(source):
                    n += 1
                    if save_video is not None:
                        if writer is None:
                            save_video.parent.mkdir(parents=True, exist_ok=True)
                            h, w = result.frame.image.shape[:2]
                            writer = cv2.VideoWriter(
                                str(save_video), cv2.VideoWriter.fourcc(*"mp4v"),
                                getattr(source, "fps", 30.0) or 30.0, (w, h),
                            )
                        writer.write(draw_overlay(result.frame.image.copy(), result, cfg.ui))
                    if n % 15 == 0:
                        m = pipeline.metrics.snapshot()
                        status.update(
                            f"[cyan]quadro {n} | {m.fps:5.1f} FPS | latencia p95 "
                            f"{m.latency_e2e.p95:5.1f} ms | {m.n_unique_plates} placas[/cyan]"
                        )
        except KeyboardInterrupt:
            motivo = "interrompido"
            console.print("\n[yellow]Interrompido pelo usuario.[/yellow]")
        except Exception:
            motivo = "erro"
            raise
        finally:
            try:
                pipeline.finalize()
            except BaseException:
                motivo = "erro"
                raise
            finally:
                if writer is not None:
                    writer.release()
                close_session(store, pipeline, source, motivo)
            elapsed = time.perf_counter() - started

        m = pipeline.metrics.snapshot()
        console.print()
        _print_kv(
            "Resultado",
            {
                "Quadros processados": n,
                "Tempo total": f"{elapsed:.2f} s",
                "Vazao": f"{n / elapsed:.1f} quadros/s" if elapsed > 0 else "-",
                "Latencia ponta a ponta p50 / p95": (
                    f"{m.latency_e2e.p50:.1f} / {m.latency_e2e.p95:.1f} ms"
                ),
                "Tempo de servico p50 / p95": f"{m.latency.p50:.1f} / {m.latency.p95:.1f} ms",
                "Deteccao p50": f"{m.detect.p50:.1f} ms",
                "Tentativas de OCR": m.n_ocr_attempts,
                "Leituras com texto": m.n_reads,
                "Taxa de sintaxe valida": f"{m.syntax_valid_rate:.1%}",
                "Placas unicas publicadas": m.n_unique_plates,
                "Quadros descartados": m.frames_dropped,
            },
        )

        if m.counts_by_class:
            tabela = Table(title="Objetos rastreados (objetos unicos)")
            tabela.add_column("Classe", style="cyan")
            tabela.add_column("Categoria", style="magenta")
            tabela.add_column("Quantidade", justify="right")
            from .core.taxonomy import category as _cat

            for cls, k in sorted(m.counts_by_class.items(), key=lambda kv: kv[1], reverse=True):
                tabela.add_row(cls, _cat(cls).value, str(k))
            console.print(tabela)
            if m.n_vulnerable:
                console.print(
                    f"  [yellow]Usuarios vulneraveis da via (pedestres, ciclistas, "
                    f"motociclistas): {m.n_vulnerable}[/yellow]"
                )

        if hasattr(pipeline, "parallel_stats"):
            _print_kv("Paralelismo", pipeline.parallel_stats())

        rows = store.query(
            "SELECT plate_text, n_events, ROUND(avg_confidence,3) AS conf FROM v_unique_plates "
            "WHERE session_id=? ORDER BY n_events DESC LIMIT 25",
            (store.session_id,),
        )
        if rows:
            table = Table(title="Placas publicadas")
            table.add_column("Placa", style="green bold")
            table.add_column("Eventos", justify="right")
            table.add_column("Confianca", justify="right")
            for r in rows:
                table.add_row(r["plate_text"], str(r["n_events"]), f"{r['conf']:.3f}")
            console.print(table)

        perdidas = store.rows_lost
        if perdidas:
            console.print(f"[red]Linhas nao gravadas no banco: {perdidas}[/red]")
        console.print(f"\nBanco: [cyan]{cfg.analytics.db_path}[/cyan]  (sessao {store.session_id})")
        console.print(f"Exportar: [cyan]lprts export --db {cfg.analytics.db_path} --session {store.session_id}[/cyan]")

        return 0


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

@app.command()
def demo(
    frames: int = typer.Option(400, "--frames", help="Quadros do video sintetico."),
    vehicles: int = typer.Option(4, "--vehicles"),
    keep: bool = typer.Option(True, "--keep/--no-keep", help="Manter os arquivos gerados."),
) -> None:
    """Gera um video sintetico, executa e avalia o pipeline - sem downloads."""
    import subprocess

    setup_logging("INFO")
    video = Path("data/videos/sintetico.mp4")
    if _sintetico_compativel(video, frames, vehicles):
        # O video versionado e o de referencia: regenera-lo sobrescreve o arquivo do
        # git, e a codificacao do mp4 muda com a versao do OpenCV - mesmo ground
        # truth, outros pixels, outro resultado (medido: 0,56 -> 0,33 por veiculo).
        console.rule("[bold cyan]1/2 - Video sintetico de referencia (ja existe)")
    else:
        console.rule("[bold cyan]1/2 - Gerando video sintetico")
        subprocess.run(
            [sys.executable, "-m", "lprts.synthetic", "--out", str(video),
             "--frames", str(frames), "--vehicles", str(vehicles)],
            check=True,
        )

    console.rule("[bold cyan]2/2 - Executando e avaliando contra o ground truth")
    cfg = AppConfig()
    cfg.source.uri = str(video)
    cfg.source.resize_width = None
    # Serial: o desempenho que a demo mostra e o custo por quadro (em paralelo a
    # acuracia seria a mesma, e a latencia incluiria a fila). 400 quadros nao pedem mais.
    cfg.detector.workers = 0
    cfg.detector.detect_vehicles = False
    cfg.detector.plate_detector = "classic"
    cfg.ocr.engine = "template"
    cfg.ocr.min_plate_width = 55
    cfg.ocr.ocr_every_n = 1
    cfg.ocr.max_reads_published = 0
    cfg.fusion.max_reads_per_track = 0
    # O demo recria o proprio banco a cada execucao (analytics.clear_previous).
    cfg.analytics.db_path = "runs/demo.db"
    gt = video.with_name(video.stem + "_gt.csv")
    _evaluate(cfg, gt, Path("runs/demo_report.json"))

    console.print()
    console.rule("[bold cyan]Para ver na interface grafica")
    console.print(
        f"  [cyan]lprts run --source {video} "
        "--no-vehicle-detector --plate-detector classic[/cyan]"
    )
    console.print(
        "\n  Os veiculos deste video sao formas sinteticas, que um detector treinado\n"
        "  em fotos reais (COCO) nao reconhece. As opcoes acima dizem ao sistema\n"
        "  para rastrear as placas diretamente.",
        soft_wrap=True, highlight=False,
    )
    if not keep:
        video.unlink(missing_ok=True)
        gt.unlink(missing_ok=True)


def _sintetico_compativel(video: Path, frames: int, vehicles: int) -> bool:
    """O video sintetico existente foi gerado com estes parametros (e tem o GT)?"""
    meta = video.with_name(video.stem + "_meta.json")
    gt = video.with_name(video.stem + "_gt.csv")
    if not (video.exists() and meta.exists() and gt.exists()):
        return False
    try:
        args = json.loads(meta.read_text(encoding="utf-8")).get("args", {})
    except (OSError, json.JSONDecodeError):
        return False
    return args.get("frames") == frames and args.get("vehicles") == vehicles


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@app.command(name="eval")
def eval_cmd(
    source: str = typer.Option(..., "--source", "-s", help="Video a avaliar."),
    ground_truth: Path = typer.Option(..., "--gt", help="CSV/JSON de ground truth."),
    config: list[Path] | None = CONFIG_OPT,
    ocr_engine: str | None = typer.Option(None, "--ocr"),
    plate_detector: str | None = typer.Option(None, "--plate-detector"),
    vehicle_detector: bool | None = typer.Option(
        None, "--vehicle-detector/--no-vehicle-detector",
        help="Sem a opcao, vale o que o YAML diz (detector.detect_vehicles).",
    ),
    device: str | None = typer.Option(None, "--device"),
    db: Path = typer.Option(Path("runs/eval.db"), "--db", help="Banco onde a sessao e gravada."),
    out: Path = typer.Option(Path("runs/report.json"), "--out", "-o"),
    iou: float = typer.Option(0.5, "--iou", help="Limiar de IoU para verdadeiro positivo."),
    scope_min_width: float = SCOPE_OPT,
    max_frames: int = typer.Option(0, "--max-frames"),
    realtime: bool | None = typer.Option(
        None, "--realtime/--offline",
        help="Avaliar sob restricao de tempo real (descartes e OCR assincrono). "
             "Nao deterministico: use com --repeat.",
    ),
    repeat: int = typer.Option(1, "--repeat", help="Repeticoes (agrega media e desvio)."),
    parallel: int | None = typer.Option(
        None, "--parallel", "-j",
        help="Workers em paralelo (0 = serial, -1 = automatico); sem a opcao, vale o YAML. "
             "Acuracia identica; offline em paralelo, a latencia por quadro inclui a fila.",
    ),
    clear_previous: bool | None = CLEAR_OPT,
) -> None:
    """Executa o pipeline e o avalia contra anotacoes de referencia."""
    setup_logging("WARNING")
    cfg = _load_config(
        config,
        **{
            "source.uri": source,
            "source.realtime": realtime,
            "source.max_frames": max_frames,
            "ocr.engine": ocr_engine,
            "detector.plate_detector": plate_detector,
            "detector.detect_vehicles": vehicle_detector,
            "detector.device": device,
            "analytics.db_path": str(db),
            "analytics.clear_previous": clear_previous,
        },
    )
    if repeat < 1:
        raise typer.BadParameter("--repeat deve ser >= 1")
    if repeat == 1:
        _evaluate(cfg, ground_truth, out, iou, parallel, scope_min_width=scope_min_width)
        return
    relatorios = []
    for i in range(repeat):
        console.rule(f"[bold cyan]Repeticao {i + 1}/{repeat}")
        alvo = out.with_name(f"{out.stem}_rep{i + 1}{out.suffix}")
        # Limpa so antes da primeira: as repeticoes ficam juntas no banco.
        relatorios.append(
            _evaluate(cfg, ground_truth, alvo, iou, parallel, quiet=True, clear=i == 0,
                      scope_min_width=scope_min_width)
        )
    _aggregate_reports(relatorios, out)


def _evaluate(
    cfg: AppConfig,
    gt_path: Path,
    out: Path,
    iou_threshold: float = 0.5,
    parallel: int | None = None,
    quiet: bool = False,
    clear: bool = True,
    scope_min_width: float = 0.0,
):
    """Executa uma sessao gravando tudo e a avalia a partir do banco."""
    from .analytics.store import SchemaVersionError
    from .eval.groundtruth import load_ground_truth
    from .eval.session import evaluate_session
    from .factory import close_session, open_session
    from .ocr.engines import EngineUnavailableError

    gt = load_ground_truth(gt_path)
    # A avaliacao precisa dos dados completos no banco.
    cfg.analytics.store_raw_reads = True
    cfg.analytics.store_detections = True
    cfg.analytics.store_track_points = True
    cfg.analytics.store_frame_telemetry = True
    try:
        source, store, pipeline = _build_runtime(cfg, parallel, clear=clear)
    except (SchemaVersionError, EngineUnavailableError, FileNotFoundError, ValueError,
            RuntimeError) as exc:
        _fail(exc)
    with ExitStack() as cleanup:
        cleanup.callback(source.close)
        cleanup.callback(store.close)
        cleanup.callback(pipeline.stop)
        open_session(store, cfg, source, pipeline, notes=f"avaliacao:{gt_path}")

        n_frames = 0
        motivo = "fim_da_fonte"
        try:
            with console.status("[cyan]Processando...[/cyan]"):
                for _ in pipeline.run(source):
                    n_frames += 1
        except BaseException as exc:
            motivo = "interrompido" if isinstance(exc, KeyboardInterrupt) else "erro"
            raise
        finally:
            try:
                pipeline.finalize()
            except BaseException:
                motivo = "erro"
                raise
            finally:
                close_session(store, pipeline, source, motivo)
        sid = store.session_id

    with console.status("[cyan]Avaliando...[/cyan]"):
        report = evaluate_session(cfg.analytics.db_path, sid, gt, iou_threshold,
                                  scope_min_width=scope_min_width)
    if not quiet:
        console.print(report.to_text(), soft_wrap=True, highlight=False)
    path = report.save_json(out)
    tabelas = report.save_tables(out)
    console.print(f"\nRelatorio JSON: [cyan]{path}[/cyan]")
    console.print("Tabelas para analise: " + ", ".join(f"[cyan]{t}[/cyan]" for t in tabelas))
    return report


def _aggregate_reports(relatorios: list, out: Path) -> None:
    """Media, desvio e extremos das metricas principais entre repeticoes."""
    chaves = {
        "revocacao_veiculo": lambda r: r.vehicle_level.get("recall"),
        "precisao_eventos": lambda r: r.vehicle_level.get("precision"),
        "revocacao_1a_decisao": lambda r: r.vehicle_level.get("first_decision_recall"),
        "precisao_emitidos": lambda r: r.vehicle_level.get("precision_emitted"),
        "exatidao_fusao": lambda r: (r.fusion_comparison or {}).get("fused"),
        "exatidao_publicada": lambda r: (r.fusion_comparison or {}).get("published"),
        "f1": lambda r: r.vehicle_level.get("f1"),
        "exatidao_tentativa": lambda r: r.frame_level.get("accuracy"),
        "vazao_fps": lambda r: r.performance.get("throughput_fps"),
        "latencia_p95_ms": lambda r: r.performance.get("latency_p95_ms"),
        "latencia_p99_ms": lambda r: r.performance.get("latency_p99_ms"),
        "taxa_descarte": lambda r: r.performance.get("drop_rate"),
        "taxa_prazo_perdido": lambda r: r.performance.get("deadline_miss_rate"),
    }
    agregado: dict = {"n_repeticoes": len(relatorios),
                      "sessoes": [r.session_id for r in relatorios], "metricas": {}}
    tabela = Table(title=f"Agregado de {len(relatorios)} repeticoes")
    for col in ("metrica", "media", "desvio", "min", "max"):
        tabela.add_column(col, justify="right")
    for nome, fn in chaves.items():
        vals = [v for r in relatorios if (v := fn(r)) is not None and not math.isnan(v)]
        if not vals:
            continue
        media = sum(vals) / len(vals)
        desvio = (sum((v - media) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        agregado["metricas"][nome] = {
            "media": media, "desvio": desvio, "min": min(vals), "max": max(vals), "valores": vals,
        }
        tabela.add_row(nome, f"{media:.4f}", f"{desvio:.4f}", f"{min(vals):.4f}", f"{max(vals):.4f}")
    console.print(tabela)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agregado, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nAgregado: [cyan]{out}[/cyan]")


@app.command(name="eval-db")
def eval_db(
    db: Path = typer.Option(..., "--db", help="Banco com a sessao gravada."),
    ground_truth: Path = typer.Option(..., "--gt", help="CSV/JSON de ground truth."),
    session: str | None = typer.Option(None, "--session", help="Sessao (padrao: a mais recente)."),
    out: Path = typer.Option(Path("runs/report.json"), "--out", "-o"),
    iou: float = typer.Option(0.5, "--iou"),
    assoc_iou: float = typer.Option(0.3, "--assoc-iou", help="IoU para associar trilha a veiculo."),
    scope_min_width: float = SCOPE_OPT,
) -> None:
    """Reavalia uma sessao ja gravada - sem reprocessar o video."""
    setup_logging("WARNING")
    from .eval.groundtruth import load_ground_truth
    from .eval.session import evaluate_session

    try:
        report = evaluate_session(db, session, load_ground_truth(ground_truth), iou, assoc_iou,
                                  scope_min_width=scope_min_width)
    except (FileNotFoundError, ValueError) as exc:
        _fail(exc)
    console.print(report.to_text(), soft_wrap=True, highlight=False)
    path = report.save_json(out)
    tabelas = report.save_tables(out)
    console.print(f"\nRelatorio JSON: [cyan]{path}[/cyan]")
    console.print("Tabelas para analise: " + ", ".join(f"[cyan]{t}[/cyan]" for t in tabelas))


@app.command()
def compare(
    report_a: Path = typer.Argument(..., help="Relatorio JSON da configuracao A."),
    report_b: Path = typer.Argument(..., help="Relatorio JSON da configuracao B."),
) -> None:
    """Compara dois relatorios nos MESMOS veiculos (McNemar e diferenca com IC)."""
    from .eval.stats import paired_comparison

    a = json.loads(report_a.read_text(encoding="utf-8"))
    b = json.loads(report_b.read_text(encoding="utf-8"))
    va = {v["vehicle"]: v for v in a.get("vehicles", []) if not v.get("excluded_difficult")}
    vb = {v["vehicle"]: v for v in b.get("vehicles", []) if not v.get("excluded_difficult")}
    comuns = sorted(set(va) & set(vb))
    if not comuns:
        _fail(ValueError("Os relatorios nao tem veiculos em comum (mesmo video e ground truth?)."))
    if len(comuns) < max(len(va), len(vb)):
        console.print(f"[yellow]Comparando {len(comuns)} veiculos em comum "
                      f"(A tem {len(va)}, B tem {len(vb)}).[/yellow]")
    r = paired_comparison(
        [bool(va[k]["published_correct"]) for k in comuns],
        [bool(vb[k]["published_correct"]) for k in comuns],
    )
    _print_kv(
        "Comparacao pareada (placa correta publicada, por veiculo)",
        {
            "Veiculos": r["n"],
            f"Revocacao A ({report_a.name})": f"{r['rate_a']:.4f}",
            f"Revocacao B ({report_b.name})": f"{r['rate_b']:.4f}",
            "Diferenca B - A [IC 95%]": (
                f"{r['diff_b_minus_a']:+.4f} [{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}]"
            ),
            "So A acertou / so B acertou": f"{r['only_a_correct']} / {r['only_b_correct']}",
            "McNemar exato (p)": f"{r['mcnemar_p']:.4g}",
        },
    )


# ---------------------------------------------------------------------------
# export / sessions
# ---------------------------------------------------------------------------

@app.command()
def export(
    db: Path = typer.Option(Path("runs/lprts.db"), "--db"),
    out: Path = typer.Option(Path("runs/export"), "--out", "-o",
                             help="Pasta base; cada video vai para <out>/<video>_<sessao>/."),
    session: str | None = typer.Option(
        None, "--session", help="Sessao a exportar (padrao: a mais recente; ver `lprts sessions`)."
    ),
    todas: bool = typer.Option(
        False, "--all", help="Exportar o banco inteiro, todas as sessoes juntas."
    ),
    fmt: str = typer.Option("csv", "--format", "-f", help="csv|parquet|json"),
) -> None:
    """Exporta os dados do video analisado para analise externa (R, pandas, DuckDB...).

    Por padrao sai so a sessao mais recente - o ultimo video processado -, com
    todas as tabelas filtradas por ela e o resumo ``sessao.json``. O banco
    acumula todas as execucoes; ``--all`` exporta tudo junto.
    """
    setup_logging("WARNING")
    from .analytics.export import (
        export_analyzed_video,
        export_csv,
        export_parquet,
        session_source,
    )

    if todas:
        if session:
            raise typer.BadParameter("use --session OU --all")
        if fmt == "csv":
            paths = export_csv(db, out)
        elif fmt == "parquet":
            paths = export_parquet(db, out)
        else:
            raise typer.BadParameter("--all aceita csv ou parquet")
        console.print(f"[yellow]Banco inteiro (todas as sessoes) exportado para {out}[/yellow]")
    else:
        try:
            sid, pasta, paths = export_analyzed_video(db, out, session, fmt)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(f"Video [cyan]{session_source(db, sid)}[/cyan] (sessao {sid}) -> {pasta}")
    for path in paths:
        console.print(f"  [green]OK[/green] {path}")


@app.command()
def report(
    db: list[Path] = typer.Option(
        [Path("runs/lprts.db")], "--db", help="Banco(s) de resultados (repita para varios)."
    ),
    out: Path = typer.Option(Path("runs/relatorio"), "--out", "-o"),
    session: list[str] | None = typer.Option(
        None, "--session", help="Sessoes a incluir (padrao: todas dos bancos)."
    ),
    min_dwell: float = typer.Option(
        0.5, "--min-dwell", help="Permanencia minima (s) para contar um objeto."
    ),
    figures: bool = typer.Option(True, "--figures/--no-figures"),
) -> None:
    """Tabelas e figuras descritivas para o artigo (sem textos de placa).

    Composicao do trafego, funil das placas, OCR por tamanho de placa, custo por
    estagio e tempo real, de todas as sessoes dos bancos. Sao medidas do
    sistema; acuracia exige ground truth (``lprts eval``).
    """
    setup_logging("WARNING")
    from .analytics.study_report import build_report

    try:
        arquivos = build_report(db, out, session or None, min_dwell, figures)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        _fail(exc)
    for caminho in arquivos:
        console.print(f"  [green]OK[/green] {caminho}")
    console.print(f"\nResumo: [cyan]{out / 'relatorio.md'}[/cyan]")


@app.command()
def sessions(db: Path = typer.Option(Path("runs/lprts.db"), "--db")) -> None:
    """Lista as sessoes gravadas no banco."""
    setup_logging("WARNING")
    from .analytics.export import list_sessions

    rows = list_sessions(db)
    if not rows:
        console.print("[yellow]Nenhuma sessao encontrada.[/yellow]")
        return
    table = Table(title=f"Sessoes em {db}")
    for col in ("Sessao", "Inicio", "Fonte", "OCR", "Dispositivo", "Codigo", "Eventos", "Placas"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["session_id"], (r["started_at"] or "")[:19], Path(r["source_uri"] or "").name,
            r["ocr_engine"] or "-", r["device"] or "-", r.get("code_version") or "-",
            str(r["n_events"]), str(r["n_plates"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# bench / calibrate
# ---------------------------------------------------------------------------

@app.command()
def bench(
    source: str = typer.Option(..., "--source", "-s"),
    frames: int = typer.Option(200, "--frames"),
    device: str | None = typer.Option(None, "--device"),
    ocr_engine: str | None = typer.Option(None, "--ocr"),
    plate_detector: str | None = typer.Option(None, "--plate-detector"),
    vehicle_detector: bool | None = typer.Option(
        None, "--vehicle-detector/--no-vehicle-detector",
        help="Sem a opcao, vale o que o YAML diz (detector.detect_vehicles).",
    ),
) -> None:
    """Mede o desempenho de cada estagio isoladamente."""
    setup_logging("WARNING")
    import subprocess

    cmd = [sys.executable, "-m", "lprts.benchmark", "--source", source, "--frames", str(frames)]
    for flag, value in (("--device", device), ("--ocr", ocr_engine),
                        ("--plate-detector", plate_detector)):
        if value:
            cmd += [flag, value]
    if vehicle_detector is False:
        cmd.append("--no-vehicle-detector")
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


@app.command()
def calibrate(
    source: str = typer.Option(..., "--source", "-s", help="Video ou camera a calibrar."),
    out: Path = typer.Option(Path("configs/calibracao.yaml"), "--out", "-o"),
    frame: int = typer.Option(0, "--frame", help="Quadro a usar como referencia."),
) -> None:
    """Desenha linhas virtuais e calibra a homografia sobre um quadro."""
    setup_logging("WARNING")
    from .ui.calibration import run_calibration

    raise SystemExit(run_calibration(source, out, frame))


@app.command()
def batch(
    videos: list[str] = typer.Argument(..., help="Arquivos de video ou padroes glob."),
    backend: str = typer.Option("process", "--backend", help="sequential | process | mpi"),
    config: list[Path] | None = CONFIG_OPT,
    out: Path = typer.Option(Path("runs/lote"), "--out", "-o"),
    workers: int = typer.Option(0, "--workers", "-w", help="0 = nucleos disponiveis - 1"),
    parallel_workers: int = typer.Option(
        0, "--parallel-workers", help="Workers de deteccao DENTRO de cada video."
    ),
    merge: bool = typer.Option(True, "--merge/--no-merge", help="Consolidar os bancos."),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Processa muitos videos em paralelo (multiprocessos ou MPI num cluster)."""
    import glob as _glob

    setup_logging(log_level)
    from .core.batch import run_batch

    arquivos: list[str] = []
    for padrao in videos:
        arquivos.extend(sorted(_glob.glob(padrao)) or [padrao])
    if not arquivos:
        raise typer.BadParameter("Nenhum video encontrado.")

    cfg = _load_config(config)
    console.print(f"[cyan]Processando {len(arquivos)} videos com backend '{backend}'...[/cyan]")
    resultado = run_batch(arquivos, cfg, out, backend, workers, parallel_workers, merge)
    console.print()
    console.print(resultado.summary(), soft_wrap=True, highlight=False)
    raise SystemExit(1 if resultado.failed else 0)


@app.command()
def tune(
    db: Path = typer.Option(..., "--db", help="Banco com as leituras brutas (tabela reads)."),
    ground_truth: Path = typer.Option(..., "--gt", help="Ground truth (com caixas) para medir acerto."),
    space: str = typer.Option("fusion", "--space", help="fusion | realtime | tracker"),
    trials: int = typer.Option(500, "--trials", "-n"),
    backend: str = typer.Option("auto", "--backend", help="auto | optuna | nsga2 | random"),
    session: str | None = typer.Option(None, "--session"),
    objectives: str = typer.Option(
        "precisao,revocacao", "--objectives", help="Objetivos separados por virgula."
    ),
    holdout: float = typer.Option(
        0.3, "--holdout",
        help="Fracao de VEICULOS reservada para validar a fronteira (0 = sem validacao).",
    ),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("runs/tuning.json"), "--out", "-o"),
    grid: bool = typer.Option(False, "--grid", help="Varredura exaustiva em vez de busca."),
) -> None:
    """Otimiza hiperparametros e emite a fronteira de Pareto.

    Para o espaco ``fusion``, a avaliacao usa as leituras ja gravadas no banco -
    nenhum video e reprocessado. A verdade de cada trilha vem da POSICAO das
    leituras (caixas do ground truth), e a revocacao e medida sobre os veiculos
    anotados. Com ``--holdout``, a fronteira e escolhida numa parte dos veiculos
    e reavaliada na outra.
    """
    setup_logging("WARNING")
    from .eval.groundtruth import load_ground_truth
    from .optimize.pareto import Direction, Trial
    from .optimize.refusion import (
        LoadInfo,
        attach_ground_truth,
        evaluate_fusion,
        grid_search,
        load_reads,
        make_objective,
        read_statistics,
        split_by_vehicle,
    )
    from .optimize.space import build_space
    from .optimize.tuner import Tuner

    if space != "fusion":
        console.print(
            f"[yellow]O espaco '{space}' exige reprocessar o video a cada tentativa "
            "e ainda nao esta ligado a este comando. Use --space fusion.[/yellow]"
        )
        raise SystemExit(2)

    info_carga = LoadInfo()
    tracks = load_reads(db, session, info=info_carga)
    if not tracks:
        console.print(f"[red]Nenhuma leitura encontrada em {db}.[/red]")
        console.print("Rode primeiro com analytics.store_raw_reads ativado.")
        raise SystemExit(1)
    if info_carga.censored_sessions:
        console.print(
            "[yellow]ATENCAO: sessoes gravadas com ocr.max_reads_published > 0 "
            f"({len(info_carga.censored_sessions)}). Trilhas publicadas pararam de ser lidas: "
            "configuracoes que exigem mais leituras sao avaliadas sobre dados censurados. "
            "Grave com configs/avaliacao.yaml (max_reads_published: 0).[/yellow]"
        )

    gt = load_ground_truth(ground_truth)
    tracks = attach_ground_truth(tracks, gt)
    veiculos = {k for k, v in gt.vehicles().items() if not v.all_difficult}
    _print_kv("Leituras carregadas", read_statistics(tracks))
    console.print()

    if holdout > 0:
        particao = split_by_vehicle(tracks, veiculos, holdout, seed)
        ajuste, validacao = particao.tune_tracks, particao.holdout_tracks
        n_ajuste, n_validacao = len(particao.tune_vehicles), len(particao.holdout_vehicles)
    else:
        ajuste, validacao = tracks, []
        n_ajuste, n_validacao = len(veiculos), 0

    if grid:
        import json
        import time

        inicio = time.perf_counter()
        resultados = grid_search(ajuste, n_vehicles=n_ajuste)
        decorrido = time.perf_counter() - inicio

        ordenados = sorted(resultados, key=lambda x: x[1].f1, reverse=True)
        tabela = Table(title=f"Varredura exaustiva ({len(resultados)} configuracoes, ajuste)")
        for col in ("modo", "min_reads", "acordo", "Precisao", "Revocacao", "F1", "CER"):
            tabela.add_column(col, justify="right")
        for params, m in ordenados[:15]:
            tabela.add_row(
                str(params["fusion.mode"]), str(params["fusion.min_reads"]),
                f"{params['fusion.min_agreement']:.2f}", f"{m.precision:.3f}",
                f"{m.recall:.3f}", f"{m.f1:.3f}", f"{m.cer:.4f}",
            )
        console.print(tabela)

        # A grade inteira roda sobre as leituras ja gravadas: medir o custo por
        # configuracao e o argumento central do metodo - reanalisar nao custa video.
        por_config_ms = decorrido * 1000 / len(resultados) if resultados else 0.0
        console.print(
            f"\n{len(resultados)} configuracoes em {decorrido:.3f} s "
            f"({por_config_ms:.2f} ms por configuracao, sem reprocessar video)."
        )

        # A frente de Pareto: as configuracoes que ninguem domina em precisao e
        # revocacao ao mesmo tempo. E ela, nao o melhor F1, que mostra a escolha.
        from .optimize.pareto import Trial, pareto_front

        nomes_grade = ["precisao", "revocacao"]
        ensaios = [
            Trial(number=i, params=params, objectives=m.as_objectives())
            for i, (params, m) in enumerate(resultados)
        ]
        frente = pareto_front(ensaios, nomes_grade, ["max", "max"])

        relatorio = {
            "db": str(db),
            "session": session,
            "space": space,
            "mode": "grid",
            "n_configs": len(resultados),
            "elapsed_s": decorrido,
            "ms_per_config": por_config_ms,
            "reads": read_statistics(ajuste),
            "holdout": {
                "n_tune_vehicles": n_ajuste,
                "n_holdout_vehicles": n_validacao,
                "seed": seed,
            },
            "pareto_front": [
                {"params": t.params, "objectives": t.objectives} for t in frente
            ],
            "trials": [
                {"params": params, "objectives": m.as_objectives()}
                for params, m in ordenados
            ],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(
            f"Frente de Pareto: {len(frente)} de {len(resultados)} configuracoes.\n"
            f"Estudo completo: [cyan]{out}[/cyan]"
        )
        return

    nomes = [o.strip() for o in objectives.split(",") if o.strip()]
    direcoes: list[Direction] = ["min" if n == "cer" else "max" for n in nomes]
    tuner = Tuner(build_space(space), nomes, direcoes, seed=seed)
    estudo = tuner.optimize(
        make_objective(ajuste, nomes, n_vehicles=n_ajuste), n_trials=trials, backend=backend
    )
    console.print(estudo.to_text(), soft_wrap=True, highlight=False)

    dados = estudo.to_dict()
    if validacao:
        tabela = Table(title=f"Fronteira reavaliada nos veiculos reservados ({holdout:.0%})")
        for col in ("modo", "min_reads", "acordo", "conf", "P ajuste", "R ajuste",
                    "P valid.", "R valid."):
            tabela.add_column(col, justify="right")
        reavaliados = []
        # Um representante por ponto distinto da fronteira: configuracoes
        # empatadas nos objetivos de ajuste repetiriam a mesma linha.
        representantes: dict[tuple, Trial] = {}
        for t in estudo.front:
            chave = tuple(round(t.objectives.get(n, 0.0), 6) for n in nomes)
            representantes.setdefault(chave, t)
        for t in list(representantes.values())[:15]:
            p = t.params
            m = evaluate_fusion(
                validacao, mode=str(p.get("fusion.mode", "char")),
                min_reads=int(cast("int | float | str", p.get("fusion.min_reads", 3))),
                min_agreement=float(cast("int | float | str", p.get("fusion.min_agreement", 0.6))),
                min_confidence=float(cast("int | float | str", p.get("fusion.min_confidence", 0.45))),
                require_valid_syntax=bool(p.get("fusion.require_valid_syntax", True)),
                n_vehicles=n_validacao,
                publish_policy=str(p.get("fusion.publish_policy", "first_stable")),
                exit_min_reads=int(cast("int | float | str", p.get("fusion.exit_min_reads", 0))),
            )
            reavaliados.append({"params": p, "validacao": m.as_objectives()})
            tabela.add_row(
                str(p.get("fusion.mode")), str(p.get("fusion.min_reads")),
                f"{p.get('fusion.min_agreement', 0):.2f}", f"{p.get('fusion.min_confidence', 0):.2f}",
                f"{t.objectives.get('precisao', 0):.3f}", f"{t.objectives.get('revocacao', 0):.3f}",
                f"{m.precision:.3f}", f"{m.recall:.3f}",
            )
        console.print(tabela)
        dados["holdout"] = {"fraction": holdout, "seed": seed, "front_on_holdout": reavaliados}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    console.print(f"\nEstudo completo: [cyan]{out}[/cyan]")


def main() -> None:
    # Antes de qualquer coisa: a telemetria do ONNX Runtime precisa ser desligada
    # antes da criacao da primeira sessao de inferencia (ver lprts.runtime).
    from .runtime import configure_all

    configure_all()
    app()


if __name__ == "__main__":
    main()
