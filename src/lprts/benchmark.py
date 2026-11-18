#!/usr/bin/env python3
"""Mede o custo de cada estagio do pipeline isoladamente.

Por que medir por estagio e nao so o FPS agregado: o FPS diz *se* o sistema
atende ao tempo real, mas nao *o que* limita. Este script separa captura,
deteccao, rastreamento e OCR, mostrando onde investir - trocar o modelo,
reduzir a resolucao, ou paralelizar o OCR.

Uso::

    python scripts/benchmark.py --source data/videos/sintetico.mp4 --frames 300
    python scripts/benchmark.py --source 0 --device coreml --ocr fastplate
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from contextlib import ExitStack

from lprts.config import AppConfig
from lprts.core.clock import percentiles
from lprts.detect.backends import describe_environment
from lprts.factory import build_pipeline, build_source
from lprts.utils.logging import setup_logging


def _row(name: str, values: list[float], total_ms: float) -> str:
    if not values:
        return f"  {name:<14} {'-':>9}"
    p = percentiles(values)
    share = 100.0 * sum(values) / total_ms if total_ms > 0 else 0.0
    return (
        f"  {name:<14} {p.mean:>8.2f} {p.p50:>8.2f} {p.p95:>8.2f} {p.p99:>8.2f} "
        f"{p.max:>8.2f} {share:>7.1f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ocr", default=None)
    ap.add_argument("--plate-detector", default=None)
    ap.add_argument("--no-vehicle-detector", action="store_true")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10, help="quadros ignorados na estatistica")
    args = ap.parse_args()

    setup_logging("WARNING")
    cfg = AppConfig()
    cfg.source.uri = args.source
    cfg.source.realtime = False        # benchmark nao descarta quadros
    cfg.source.max_frames = args.frames + args.warmup
    cfg.analytics.db_path = "runs/bench.db"
    cfg.analytics.store_raw_reads = False
    cfg.analytics.store_frame_telemetry = False
    cfg.analytics.save_crops = False
    if args.width:
        cfg.source.resize_width = args.width
    if args.device:
        cfg.detector.device = args.device
    if args.ocr:
        cfg.ocr.engine = args.ocr
    if args.plate_detector:
        cfg.detector.plate_detector = args.plate_detector
    if args.no_vehicle_detector:
        cfg.detector.detect_vehicles = False

    with ExitStack() as cleanup:
        source = build_source(cfg)
        cleanup.callback(source.close)
        pipeline = build_pipeline(cfg, store=None, warmup=True)
        cleanup.callback(pipeline.stop)

        print("=" * 78)
        print("LPR-RTS - benchmark de desempenho")
        print("=" * 78)
        env = describe_environment()
        print(f"  plataforma ..: {env['platform']}")
        print(f"  processador .: {env['processor']}")
        print(f"  codigo ......: {env.get('code_version', '?')}")
        print(f"  energia .....: {env.get('power') or '-'}")
        print(f"  dispositivo .: {pipeline.device_summary()}")
        print(f"  veiculos ....: {pipeline.vehicle_detector.name if pipeline.vehicle_detector else 'desabilitado'}")
        print(f"  placas ......: {pipeline.plate_detector.name if pipeline.plate_detector else '-'}")
        print(f"  ocr .........: {pipeline.ocr_engine.name if pipeline.ocr_engine else '-'}")
        print(f"  fonte .......: {args.source} ({getattr(source, 'width', 0)}x{getattr(source, 'height', 0)})")
        print()

        stages: dict[str, list[float]] = {
            "capture": [], "detect": [], "track": [], "ocr": [], "analytics": [], "total": [],
        }
        wall: list[float] = []
        n = 0
        started = time.perf_counter()
        last = started

        for result in pipeline.run(source):
            n += 1
            if n <= args.warmup:      # descarta o transiente inicial
                last = time.perf_counter()
                continue
            t = result.timing
            stages["capture"].append(t.capture_ms)
            stages["detect"].append(t.detect_ms)
            stages["track"].append(t.track_ms)
            stages["ocr"].append(t.ocr_ms)
            stages["analytics"].append(t.analytics_ms)
            stages["total"].append(t.total_ms)
            now = time.perf_counter()
            wall.append((now - last) * 1000.0)
            last = now
            if len(stages["total"]) >= args.frames:
                break

        pipeline.finalize()
        elapsed = time.perf_counter() - started
        measured = len(stages["total"])
        if measured == 0:
            print("Nenhum quadro medido - o video e mais curto que --warmup?")
            return 1

        total_sum = sum(stages["total"])
        print(f"{'estagio':<16} {'media':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'fatia':>8}")
        print("  " + "-" * 74)
        print(_row("captura", stages["capture"], 0.0))
        for name in ("detect", "track", "ocr", "analytics"):
            print(_row(name, stages[name], total_sum))
        print("  " + "-" * 74)
        print(_row("SERVICO", stages["total"], total_sum))
        print(_row("relogio/quadro", wall, 0.0))

        fps_wall = measured / (sum(wall) / 1000.0) if wall else 0.0
        fps_theoretical = 1000.0 / statistics.mean(stages["total"]) if stages["total"] else 0.0
        m = pipeline.metrics.snapshot()

        print()
        print(f"  quadros medidos ......: {measured} (de {n}; {args.warmup} de aquecimento)")
        print(f"  tempo total ..........: {elapsed:.2f} s")
        print(f"  FPS medido ...........: {fps_wall:.1f}")
        print(f"  FPS teorico (1/lat.) .: {fps_theoretical:.1f}")
        print(f"  leituras de OCR ......: {m.n_reads}  (sintaxe valida: {m.syntax_valid_rate:.1%})")
        print(f"  placas unicas ........: {m.n_unique_plates}")

        # O veredito usa o tempo de RELOGIO por quadro (inclui leitura/decodificacao
        # e gravacao), nao so o tempo de servico: e ele que precisa caber no
        # periodo de quadro para acompanhar a fonte sem descartar.
        budget_30 = 1000.0 / 30.0
        p95 = percentiles(wall).p95
        print()
        if p95 <= budget_30:
            print(f"  [OK] p95 por quadro = {p95:.1f} ms <= {budget_30:.1f} ms: acompanha 30 fps.")
        elif p95 <= 1000.0 / 15.0:
            print(f"  [ATENCAO] p95 por quadro = {p95:.1f} ms: acompanha ~15 fps, nao 30.")
            print("            Reduza --width, use detector.detect_every_n: 2 ou um modelo menor.")
        else:
            print(f"  [LENTO] p95 por quadro = {p95:.1f} ms: abaixo de 15 fps.")
            print("          Considere CoreML/GPU, resolucao menor ou detector mais leve.")
        print("  (Desempenho sob carga real, com descarte: lprts eval --realtime --repeat 5)")

        return 0


if __name__ == "__main__":
    sys.exit(main())
