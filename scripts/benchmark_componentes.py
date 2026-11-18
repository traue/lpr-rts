#!/usr/bin/env python3
"""Latencia de cada componente em cada runtime disponivel nesta maquina.

Complementa o ``lprts bench`` (que mede o pipeline montado): aqui cada peca e
medida isoladamente, em cada runtime, para montar o orcamento de tempo real de
uma configuracao antes de roda-la:

    custo_por_quadro ~ objetos + n_veiculos x placa_por_roi + leituras x ocr

* **Detector de objetos** (YOLO11n): pesos ``.pt`` em PyTorch/MPS e em CPU, e
  ``.mlpackage`` em CoreML (Neural Engine/GPU), na resolucao pedida.
* **Detector de placas**: o classico (quadro inteiro e por ROI) e, se houver
  ``data/models/plate_yolo.pt``, o YOLO de placas. Sem ele, um YOLO11n em
  ``imgsz=320`` sobre a ROI serve de PROXY de custo (mesma arquitetura de um
  detector de placas nano; nao mede acuracia).
* **OCR**: cada motor instalado, sobre um recorte de placa.

Cada medida descarta as primeiras chamadas (compilacao de kernels, alocacao) e
reporta p50/p95/p99/max. O resultado vai para JSON com a proveniencia completa
(commit, chip, energia, versoes) - feche outros programas e ligue na tomada.

Uso::

    python scripts/benchmark_componentes.py
    python scripts/benchmark_componentes.py --source video.mp4 --width 1920 --out runs/bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lprts.core.clock import percentiles  # noqa: E402
from lprts.utils.logging import setup_logging  # noqa: E402

MODELOS = Path("data/models")


def _quadro(source: str | None, largura: int) -> np.ndarray:
    """Um quadro de conteudo real: do video informado ou da foto de exemplo do ultralytics."""
    if source:
        cap = cv2.VideoCapture(source)
        for _ in range(30):              # pula o inicio (quadros pretos, fade)
            ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Nao foi possivel ler {source}")
    else:
        import ultralytics

        img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
        # A foto e vertical; recorta uma faixa 16:9 (formato de camera) com o onibus.
        h0, w0 = img.shape[:2]
        alt = int(w0 * 9 / 16)
        topo = max(0, min(h0 - alt, int(h0 * 0.35)))
        img = img[topo: topo + alt]
    altura = int(round(img.shape[0] * largura / img.shape[1]))
    return cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)


def _medir(funcao, n: int, aquecimento: int) -> dict:
    for _ in range(aquecimento):
        funcao()
    tempos = []
    for _ in range(n):
        t0 = time.perf_counter()
        funcao()
        tempos.append((time.perf_counter() - t0) * 1000.0)
    p = percentiles(tempos)
    return {"n": n, "mean_ms": p.mean, "p50_ms": p.p50, "p95_ms": p.p95, "p99_ms": p.p99,
            "max_ms": p.max, "stdev_ms": statistics.stdev(tempos) if n > 1 else 0.0}


def _linha(nome: str, r: dict | str) -> None:
    if isinstance(r, str):
        print(f"  {nome:<46} {r}")
        return
    print(f"  {nome:<46} {r['p50_ms']:7.2f} {r['p95_ms']:7.2f} {r['p99_ms']:7.2f} {r['max_ms']:7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="video de onde tirar o quadro (padrao: foto de exemplo)")
    ap.add_argument("--width", type=int, default=1280, help="largura do quadro de processamento")
    ap.add_argument("--n", type=int, default=200, help="repeticoes medidas por componente")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="runs/benchmark_componentes.json")
    args = ap.parse_args()
    setup_logging("ERROR")
    # Antes de qualquer sessao ONNX: sem isto o processo aborta ao encerrar
    # (telemetria do ONNX Runtime; ver lprts.runtime).
    from lprts.runtime import configure_all

    configure_all()

    from lprts.detect.backends import describe_environment, is_available, resolve_device
    from lprts.detect.classic_plate import ClassicPlateDetector
    from lprts.detect.yolo_detector import YOLODetector, build_object_detector
    from lprts.ocr.engines import available_engines, build_ocr_engine

    quadro = _quadro(args.source, args.width)
    h, w = quadro.shape[:2]
    # ROI tipica de um veiculo proximo (~25% da largura) e recorte de placa.
    roi = cv2.resize(quadro[h // 3: h // 3 + h // 3, w // 3: w // 3 + w // 4], (w // 4, h // 4))
    placa = np.full((64, 200, 3), 235, np.uint8)
    cv2.putText(placa, "ABC1D23", (10, 48), cv2.FONT_HERSHEY_DUPLEX, 1.4, (20, 20, 20), 3)

    resultados: dict[str, dict | str] = {}
    print(f"Quadro {w}x{h}, {args.n} repeticoes (+{args.warmup} de aquecimento)\n")
    print(f"  {'componente':<46} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}  (ms)")
    print("  " + "-" * 78)

    pt = MODELOS / "yolo11n.pt"
    ml = MODELOS / "yolo11n.mlpackage"
    variantes = []
    if pt.exists():
        if is_available("mps"):
            variantes.append(("objetos YOLO11n .pt  PyTorch/MPS", pt, "mps"))
        if is_available("cuda"):
            variantes.append(("objetos YOLO11n .pt  PyTorch/CUDA", pt, "cuda"))
        variantes.append(("objetos YOLO11n .pt  PyTorch/CPU", pt, "cpu"))
    if ml.exists():
        # O mesmo .mlpackage pelos dois caminhos: a diferenca e so a sobrecarga do
        # framework (preparo da imagem e pos-processamento), nao a inferencia.
        variantes.append(("objetos YOLO11n .mlpackage  CoreML direto", ml, "auto"))
        variantes.append(("objetos YOLO11n .mlpackage  CoreML ultralytics", ml, "ultralytics"))
    if not variantes:
        resultados["objetos"] = "sem pesos (python scripts/download_models.py --coreml)"
    for nome, pesos, dev in variantes:
        try:
            runtime = "ultralytics" if dev == "ultralytics" else "auto"
            dev = "auto" if dev == "ultralytics" else dev
            det = build_object_detector(pesos, device=resolve_device(dev), preset="trafego",
                                        runtime=runtime)
            r = _medir(lambda d=det: d.detect(quadro), args.n, args.warmup)
            r["describe"] = det.describe()
            r["n_detections"] = len(det.detect(quadro))
        except Exception as exc:  # noqa: BLE001 - um runtime ausente nao derruba a medida
            r = f"falhou: {type(exc).__name__}: {exc}"
        resultados[nome] = r
        _linha(nome, r)

    classico = ClassicPlateDetector()
    for nome, img in (("placas classico  quadro inteiro (so-placa)", quadro),
                      ("placas classico  por ROI de veiculo", roi)):
        resultados[nome] = _medir(lambda i=img: classico.detect(i), args.n, args.warmup)
        _linha(nome, resultados[nome])

    pesos_placa = MODELOS / "plate_yolo.pt"
    proxy = not pesos_placa.exists()
    pesos_placa = pt if proxy else pesos_placa
    if pesos_placa.exists():
        rotulo = "proxy YOLO11n imgsz=320" if proxy else "plate_yolo imgsz=320"
        casos = [(pesos_placa, d, "auto") for d in ("mps", "cpu") if is_available(d)]
        ml_placa = pesos_placa.with_suffix(".mlpackage")
        if ml_placa.exists():
            casos += [(ml_placa, "auto", "auto"), (ml_placa, "auto", "ultralytics")]
        for pesos, dev, runtime in casos:
            onde = (dev if pesos.suffix == ".pt" else
                    "CoreML direto" if runtime == "auto" else "CoreML ultralytics")
            nome = f"placas {rotulo}  por ROI  {onde}"
            try:
                det = YOLODetector(pesos, imgsz=320, device=resolve_device(dev), name="placa",
                                   runtime=runtime)
                r = _medir(lambda d=det: d.detect(roi), args.n, args.warmup)
                r["proxy"] = proxy
            except Exception as exc:  # noqa: BLE001
                r = f"falhou: {type(exc).__name__}: {exc}"
            resultados[nome] = r
            _linha(nome, r)

    for motor in available_engines():
        nome = f"OCR {motor}  por leitura"
        try:
            ocr = build_ocr_engine(motor)
            r = _medir(lambda o=ocr: o.read(placa), args.n, args.warmup)
            r["describe"] = ocr.describe()
        except Exception as exc:  # noqa: BLE001
            r = f"falhou: {type(exc).__name__}: {exc}"
        resultados[nome] = r
        _linha(nome, r)

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(
        {"frame_size": [w, h], "n": args.n, "warmup": args.warmup, "source": args.source or "bus.jpg",
         "results": resultados, "environment": describe_environment(include_freeze=False)},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nResultado: {destino}")
    print("Orcamento por quadro ~ objetos + n_veiculos x placa_por_ROI + (OCR e assincrono no tempo real)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
