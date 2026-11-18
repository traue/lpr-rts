"""Construcao dos componentes do pipeline a partir da configuracao.

Centralizado aqui para que CLI, interface grafica, lote e avaliacao montem
exatamente o mesmo sistema - e registrem a sessao com os mesmos metadados.
Divergencia entre eles seria uma fonte silenciosa de resultados nao
reproduziveis.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .analytics.store import ClearSummary, ResultStore, clear_results
from .capture.source import ImageFolderSource, VideoSource
from .config import AppConfig
from .core.pipeline import LPRPipeline
from .detect.backends import DeviceInfo, resolve_device
from .detect.base import Detector
from .detect.classic_plate import ClassicPlateDetector
from .ocr.base import OCREngine
from .ocr.engines import build_ocr_engine
from .utils.logging import get_logger

log = get_logger(__name__)


def _preparar_runtime() -> None:
    """Garante que o ONNX Runtime esteja configurado antes de criar sessoes."""
    from .runtime import configure_onnxruntime

    configure_onnxruntime()


def coreml_weights(weights: Path, device: DeviceInfo, imgsz: int) -> Path:
    """No CoreML, troca o ``.pt`` pelo ``.mlpackage`` exportado ao lado, se compativel.

    Com ``device: auto`` (ou ``coreml``) no Apple Silicon o dispositivo resolvido
    e "coreml", mas um ``.pt`` roda em PyTorch/MPS. Medido no M5 Pro, quadro de
    1280 px: YOLO11n 7,4 ms (p99 8,3) no MPS contra 5,2 ms (p99 5,7) no CoreML;
    pipeline em tempo real, latencia p99 24,9 ms contra 13,7 ms.

    So troca se o ``.mlpackage`` respeitar a configuracao: entrada igual a
    ``imgsz`` e sem NMS embutido (que ignoraria ``iou``). Pesos, runtime e
    parametros efetivos ficam em ``sessions.detector``/``components``. Para
    forcar o PyTorch, use ``device: mps``.
    """
    if device.kind != "coreml" or weights.suffix.lower() != ".pt":
        return weights
    alternativa = weights.with_suffix(".mlpackage")
    if not alternativa.exists():
        return weights
    from .detect.yolo_detector import coreml_export_info

    info = coreml_export_info(str(alternativa))
    if info is None or info["nms_embedded"] or info["imgsz"] != [imgsz, imgsz]:
        log.warning(
            "%s nao usado (%s); seguindo com %s em PyTorch. Reexporte com entrada %d: "
            "python scripts/download_models.py --coreml "
            "(--imgsz para veiculos, --plate-imgsz para placas)",
            alternativa.name,
            "metadados ilegiveis" if info is None
            else "NMS embutido" if info["nms_embedded"]
            else f"entrada {info['imgsz']} != imgsz {imgsz}",
            weights.name, imgsz,
        )
        return weights
    log.info("Usando %s (CoreML) no lugar de %s.", alternativa.name, weights.name)
    return alternativa


# Raiz do checkout (src/lprts/factory.py -> ../..). Instalado como wheel, nao
# contem data/models e a busca abaixo simplesmente nao acha nada.
_RAIZ_DO_PROJETO = Path(__file__).resolve().parents[2]


def resolve_weights(caminho: str | Path) -> Path:
    """Pesos como pedidos ou, se nao existirem a partir daqui, na raiz do projeto.

    Os perfis usam caminhos relativos (``data/models/...``). Aberto de outra
    pasta, o app nao os achava e caia, so com um aviso, no modo so-placa e no
    detector classico: outro pipeline, outro resultado. Um caminho que existe a
    partir da pasta atual fica como esta; a sessao grava o caminho efetivo.
    """
    p = Path(caminho)
    if p.is_absolute() or p.exists():
        return p
    candidato = _RAIZ_DO_PROJETO / p
    return candidato if candidato.exists() else p


def build_vehicle_detector(config: AppConfig) -> Detector | None:
    """Detector de objetos; ``None`` se desabilitado ou sem pesos disponiveis.

    O nome mantem "vehicle" por compatibilidade com o pipeline, mas o detector
    reconhece todas as classes do preset configurado (pessoas, bicicletas,
    animais...), nao apenas veiculos.
    """
    if not config.detector.detect_vehicles:
        return None
    weights = resolve_weights(config.detector.vehicle_weights)
    if not weights.exists():
        log.warning(
            "Pesos de veiculo nao encontrados (%s). Seguindo em modo so-placa. "
            "Rode `python scripts/download_models.py` para habilitar a deteccao de veiculos.",
            weights,
        )
        return None
    _preparar_runtime()
    from .detect.yolo_detector import build_object_detector as _build

    device = resolve_device(config.detector.device)
    return _build(
        coreml_weights(weights, device, config.detector.imgsz),
        device=device,
        preset=config.detector.classes,
        conf=config.detector.conf,
        iou=config.detector.iou,
        imgsz=config.detector.imgsz,
        half=config.detector.half,
        runtime=config.detector.runtime,
    )


def build_plate_detector(config: AppConfig) -> Detector:
    """Detector de placas conforme ``plate_detector``: yolo, classic ou auto.

    Em ``auto``, usa o modelo YOLO se os pesos existirem e cai para o baseline
    classico caso contrario - e o que garante que o sistema roda sem downloads.
    A escolha efetiva fica registrada na sessao.
    """
    mode = config.detector.plate_detector.lower()
    weights = resolve_weights(config.detector.plate_weights)

    if mode == "classic":
        return ClassicPlateDetector(conf=config.detector.plate_conf)

    if mode in ("yolo", "auto"):
        if weights.exists():
            _preparar_runtime()
            from .detect.yolo_detector import build_plate_detector as _build

            device = resolve_device(config.detector.device)
            return _build(
                coreml_weights(weights, device, config.detector.plate_imgsz),
                device=device,
                conf=config.detector.plate_conf,
                iou=config.detector.iou,
                imgsz=config.detector.plate_imgsz,
                half=config.detector.half,
                runtime=config.detector.runtime,
            )
        if mode == "yolo":
            raise FileNotFoundError(
                f"Pesos do detector de placas nao encontrados: {weights}\n"
                "\n"
                "Nao existe um peso oficial e universal para deteccao de placas - "
                "por isso ele nao acompanha o projeto. Voce tem tres opcoes:\n"
                "\n"
                "  1. Usar o detector classico, que funciona agora:\n"
                "       --plate-detector classic     (ou 'auto', que faz isso sozinho)\n"
                "\n"
                "  2. Baixar um YOLO de placas de terceiros (para testar):\n"
                "       python scripts/download_models.py --plate morsetech-n\n"
                "\n"
                "  3. Treinar um com dados brasileiros (melhor precisao):\n"
                "       python scripts/train_plate_detector.py --data data/placas.yaml\n"
                "     Veja a secao 9 do README para os datasets disponiveis."
            )
        log.warning(
            "Pesos de placa nao encontrados (%s); usando o detector classico (baseline).", weights
        )
        return ClassicPlateDetector(conf=config.detector.plate_conf)

    raise ValueError(
        f"Modo de detector de placas invalido: {mode!r}. Use 'yolo', 'classic' ou 'auto'."
    )


def build_ocr(config: AppConfig) -> OCREngine:
    _preparar_runtime()
    kwargs: dict = {}
    if config.ocr.engine in ("fastplate", "auto"):
        kwargs["model_name"] = config.ocr.model_name
        kwargs["force_coreml"] = config.ocr.force_coreml
        kwargs["threads"] = config.ocr.threads
    engine = build_ocr_engine(config.ocr.engine, **kwargs)
    engine.max_corrections = config.ocr.max_grammar_corrections
    return engine


def build_source(config: AppConfig) -> VideoSource | ImageFolderSource:
    """Abre a fonte de video (arquivo, pasta de imagens, webcam ou stream)."""
    uri = config.source.uri
    path = Path(uri)
    if path.is_dir():
        return ImageFolderSource(
            path, resize_width=config.source.resize_width,
            keep_original=config.ocr.full_resolution,
        ).open()
    source = VideoSource(
        uri,
        realtime=config.source.realtime,
        queue_size=config.source.queue_size,
        target_fps=config.source.target_fps,
        resize_width=config.source.resize_width,
        keep_original=config.ocr.full_resolution,
        backend=config.source.backend,
        open_timeout_ms=config.source.open_timeout_ms,
        read_timeout_ms=config.source.read_timeout_ms,
    ).open()
    try:
        if config.source.start_frame > 0:
            source.seek(config.source.start_frame)
    except BaseException:
        source.close()
        raise
    return source


def clear_previous_results(config: AppConfig) -> ClearSummary | None:
    """No inicio de um processamento, apaga os resultados anteriores do banco configurado.

    Chamado uma vez por processamento (Iniciar, ``lprts run``, ``lprts eval``),
    nunca por sessao: as repeticoes de ``lprts eval --repeat`` ficam no mesmo
    banco. ``None`` com ``analytics.clear_previous: false``.
    """
    if not config.analytics.clear_previous:
        return None
    resumo = clear_results(config.analytics.db_path, config.analytics.crops_dir)
    if resumo.db_removed:
        log.info("Limpeza: %s", resumo.describe())
    return resumo


def build_store(config: AppConfig, session_id: str | None = None) -> ResultStore:
    return ResultStore(
        config.analytics.db_path,
        session_id=session_id,
        batch_size=config.analytics.batch_size,
        async_writes=config.analytics.async_writes,
        store_raw_reads=config.analytics.store_raw_reads,
        store_frame_telemetry=config.analytics.store_frame_telemetry,
        store_detections=config.analytics.store_detections,
        store_track_points=config.analytics.store_track_points,
    ).open()


def build_pipeline(
    config: AppConfig, store: ResultStore | None = None, warmup: bool = True
) -> LPRPipeline:
    """Monta o pipeline inteiro e aquece os modelos."""
    with ExitStack() as cleanup:
        vehicle = build_vehicle_detector(config)
        if vehicle is not None:
            cleanup.callback(vehicle.close)
        plate = build_plate_detector(config)
        cleanup.callback(plate.close)
        ocr = build_ocr(config)
        cleanup.callback(ocr.close)
        if warmup:
            for component in (vehicle, plate, ocr):
                if component is not None:
                    component.warmup()
        pipeline = LPRPipeline(config, vehicle_detector=vehicle, plate_detector=plate,
                               ocr_engine=ocr, store=store)
        cleanup.pop_all()  # O pipeline passa a ser dono dos componentes.
    try:
        return pipeline.start()
    except BaseException:
        pipeline.stop()
        raise


def build_parallel_pipeline(
    config: AppConfig, n_workers: int, store: ResultStore | None = None,
) -> LPRPipeline:
    """Monta o paralelo com a mesma garantia de limpeza do pipeline serial."""
    from .core.parallel import ParallelPipeline

    def detectors() -> tuple[Detector | None, Detector]:
        with ExitStack() as cleanup:
            vehicle = build_vehicle_detector(config)
            if vehicle is not None:
                cleanup.callback(vehicle.close)
            plate = build_plate_detector(config)
            cleanup.callback(plate.close)
            cleanup.pop_all()
            return vehicle, plate

    with ExitStack() as cleanup:
        vehicle, plate = detectors()
        for component in (vehicle, plate):
            if component is not None:
                cleanup.callback(component.close)
        ocr = build_ocr(config)
        cleanup.callback(ocr.close)
        ocr.warmup()
        pipeline = ParallelPipeline(config, detector_factory=detectors, n_workers=n_workers,
                                    ocr_factory=lambda: build_ocr(config),
                                    vehicle_detector=vehicle, plate_detector=plate,
                                    ocr_engine=ocr, store=store)
        cleanup.pop_all()
    try:
        return pipeline.start()
    except BaseException:
        # Inclui Ctrl-C durante o aquecimento; o pipeline e dono dos modelos
        # desde pop_all, mesmo se start ainda nao terminou.
        pipeline.stop()
        raise


def build_configured_pipeline(
    config: AppConfig, store: ResultStore | None = None,
) -> LPRPipeline:
    """Serial ou paralelo, conforme ``detector.workers`` (ver ``resolve_workers``)."""
    from .core.parallel import resolve_workers

    n = resolve_workers(config)
    return build_parallel_pipeline(config, n, store) if n > 0 else build_pipeline(config, store)


def check_calibration(config: AppConfig, source: Any) -> None:
    """Recusa calibracao feita numa resolucao diferente da fonte.

    Linhas e homografia sao marcadas em pixels da imagem original; aplicadas a
    um video de outra resolucao, contariam cruzamentos no lugar errado e
    distorceriam a velocidade sem nenhum sinal de erro.
    """
    cal = config.analytics.calibration or {}
    tamanho = cal.get("image_size")
    largura, altura = getattr(source, "width", 0), getattr(source, "height", 0)
    if tamanho and largura and altura and (int(tamanho[0]), int(tamanho[1])) != (largura, altura):
        raise ValueError(
            f"A calibracao foi feita em {tamanho[0]}x{tamanho[1]}, mas a fonte tem "
            f"{largura}x{altura}. Recalibre com `lprts calibrate` nesta fonte."
        )
    if (cal.get("image_points") or config.analytics.lines) and not tamanho:
        log.warning(
            "A calibracao/linhas nao registram a resolucao em que foram marcadas "
            "(image_size). Confira que correspondem a resolucao ORIGINAL da fonte."
        )


def _source_kind(source: Any) -> str:
    if isinstance(source, ImageFolderSource):
        return "pasta"
    return "ao_vivo" if getattr(source, "is_stream", False) else "arquivo"


def open_session(
    store: ResultStore,
    config: AppConfig,
    source: Any,
    pipeline: LPRPipeline,
    notes: str = "",
) -> str:
    """Registra a sessao com todos os metadados - o mesmo caminho para todo mundo."""
    pipeline.configure_for_source(source)
    largura_p, altura_p = getattr(source, "processing_size", (0, 0))
    inicio = config.source.recording_start
    if not inicio and getattr(source, "start_utc", None) is not None:
        inicio = source.start_utc
    env = pipeline.environment()
    # O decodificador muda os pixels (pouco) e o custo da captura (muito).
    env["video_backend"] = getattr(source, "backend_name", "") or ""
    return store.start_session(
        source_uri=str(config.source.uri),
        source_fps=float(getattr(source, "fps", 0.0) or 0.0),
        source_width=int(getattr(source, "width", 0) or 0),
        source_height=int(getattr(source, "height", 0) or 0),
        frame_count=int(getattr(source, "frame_count", 0) or 0),
        realtime=config.source.realtime,
        detector=pipeline.vehicle_detector.name if pipeline.vehicle_detector else "",
        plate_detector=pipeline.plate_detector.name if pipeline.plate_detector else "",
        ocr_engine=pipeline.ocr_engine.name if pipeline.ocr_engine else "",
        device=pipeline.device_summary(),
        config=config.to_dict(),
        environment=env,
        notes=notes,
        source_kind=_source_kind(source),
        processing_width=largura_p,
        processing_height=altura_p,
        recording_start_utc=inicio,
        deadline_ms=pipeline.deadline_ms,
        code_version=str(env.get("code_version", "")),
        config_hash=config.fingerprint(),
    )


def close_session(
    store: ResultStore, pipeline: LPRPipeline, source: Any, end_reason: str = "fim_da_fonte"
) -> None:
    """Finaliza a sessao gravando os contadores da execucao."""
    stats = pipeline.stats()
    stats["source_dropped"] = int(getattr(source, "dropped", 0) or 0)
    stats["source_reconnections"] = int(getattr(source, "reconnections", 0) or 0)
    if hasattr(pipeline, "parallel_stats"):
        stats["parallel"] = pipeline.parallel_stats()
    store.finish_session(end_reason=end_reason, stats=stats)


def describe_pipeline(config: AppConfig, pipeline: LPRPipeline) -> dict[str, str]:
    """Resumo textual da configuracao efetiva (vai para o log e para o banco)."""
    return {
        "fonte": str(config.source.uri),
        "modo": "tempo real" if config.source.realtime else "offline (todos os quadros)",
        "detector_objetos": (
            pipeline.vehicle_detector.name if pipeline.vehicle_detector else "desabilitado"
        ),
        "classes": config.detector.classes,
        "detector_placa": pipeline.plate_detector.name if pipeline.plate_detector else "nenhum",
        "ocr": pipeline.ocr_engine.name if pipeline.ocr_engine else "nenhum",
        "dispositivo": pipeline.device_summary(),
        "workers": (
            f"{pipeline.n_workers} (paralelo"
            + (", OCR antecipado)" if getattr(pipeline, "prefetch_enabled", False) else ")")
            if hasattr(pipeline, "n_workers") else "0 (serial)"
        ),
    }
