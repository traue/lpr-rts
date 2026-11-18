#!/usr/bin/env python3
"""Diagnóstico: o tempo das redes muda quando as chamadas são espaçadas a 30 quadros/s?

Uso, na máquina da bateria, na tomada e com os outros programas fechados::

    python scripts/artigo/diagnostico_ritmo.py --source data/videos/originais/IMG_0834.MOV

Por que existe. O artigo compara o tempo de serviço do pipeline em tempo real com o custo
das redes medido à parte (``scripts/benchmark_componentes.py``). O benchmark chama cada
rede sem pausa. No pipeline em tempo real, o processo dorme entre um quadro e o seguinte.
Se o processador desacelera nessas pausas, a mesma chamada leva mais tempo, e a diferença
entre as duas medidas não é código fora das redes.

O que mede. Um "quadro" é uma chamada do detector de veículos mais ``--rois`` chamadas do
detector de placas, com os mesmos modelos Core ML do perfil de tempo real, em três regimes:

* **A, sem pausa:** as chamadas se seguem, como no benchmark de componentes;
* **B, 30 quadros/s:** um quadro a cada 1/30 s, com o processo dormindo no intervalo;
* **C, 30 quadros/s com a CPU ocupada:** como B, com outros processos em laço.

A e B rodam duas vezes, para mostrar se a medida deriva. Não grava banco nem imagens: o
resultado, sem placa, vai para ``docs/artigo/resultados/diagnostico_ritmo.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

FPS = 30.0
AQUECIMENTO = 30
SAIDA = RAIZ / "docs" / "artigo" / "resultados" / "diagnostico_ritmo.json"
MODELOS = RAIZ / "data" / "models"


def _ocupar(parar) -> None:
    """Laço que mantém um núcleo ocupado até ``parar`` ser sinalizado."""
    x = 0
    while not parar.is_set():
        for _ in range(100_000):
            x += 1


def _quadro(source: str, largura: int) -> np.ndarray:
    """Um quadro de conteúdo real, reduzido à largura de processamento."""
    cap = cv2.VideoCapture(source)
    ok, img = False, None
    for _ in range(30):                  # pula o início (quadros pretos, fade)
        ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise SystemExit(f"Não foi possível ler {source}")
    altura = int(round(img.shape[0] * largura / img.shape[1]))
    return cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)


def medir(quadro_fn, quadros: int, periodo: float | None, aquecimento: int = AQUECIMENTO) -> dict:
    """Roda ``quadro_fn`` repetidas vezes e devolve os tempos de cada parte, em ms.

    ``quadro_fn()`` devolve (tempo dos veículos, tempo por ROI de placa, tempo total), em
    segundos. Com ``periodo``, cada quadro começa no seu instante e o processo dorme até lá.
    O sono fica fora da medida.
    """
    veic, roi, total = [], [], []
    inicio = time.perf_counter()
    for k in range(quadros + aquecimento):
        if periodo is not None:
            falta = inicio + k * periodo - time.perf_counter()
            if falta > 0:
                time.sleep(falta)
        tv, tr, tt = quadro_fn()
        if k >= aquecimento:
            veic.append(tv * 1000)
            roi.append(tr * 1000)
            total.append(tt * 1000)
    return {"veiculos_ms": float(np.mean(veic)), "placa_por_roi_ms": float(np.mean(roi)),
            "quadro_ms": float(np.mean(total)), "quadro_p50_ms": float(np.median(total)),
            "quadro_p99_ms": float(np.percentile(total, 99)), "quadros": quadros}


def razoes(regimes: dict) -> dict:
    """Quantas vezes o quadro a 30 quadros/s custa o do regime sem pausa."""
    a = np.mean([r["quadro_ms"] for n, r in regimes.items() if n.startswith("A")])
    b = np.mean([r["quadro_ms"] for n, r in regimes.items() if n.startswith("B")])
    c = np.mean([r["quadro_ms"] for n, r in regimes.items() if n.startswith("C")])
    return {"B_sobre_A": float(b / a), "C_sobre_A": float(c / a)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="vídeo de onde tirar o quadro")
    ap.add_argument("--rois", type=int, default=5, help="buscas de placa por quadro (o n médio do V1 é 4,8)")
    ap.add_argument("--quadros", type=int, default=300, help="quadros medidos por regime")
    ap.add_argument("--ocupados", type=int, default=4, help="processos em laço no regime C")
    ap.add_argument("--out", type=Path, default=SAIDA)
    args = ap.parse_args(argv)

    from lprts.runtime import configure_all

    configure_all()
    from lprts.detect.backends import resolve_device
    from lprts.detect.yolo_detector import YOLODetector, build_object_detector

    quadro = _quadro(args.source, 1280)
    h, w = quadro.shape[:2]
    # ROI típica de um veículo próximo, como em scripts/benchmark_componentes.py
    roi = cv2.resize(quadro[h // 3: h // 3 + h // 3, w // 3: w // 3 + w // 4], (w // 4, h // 4))
    det_obj = build_object_detector(MODELOS / "yolo11n.mlpackage", device=resolve_device("auto"),
                                    preset="trafego", runtime="auto")
    det_placa = YOLODetector(MODELOS / "plate_yolo.mlpackage", imgsz=320, device=resolve_device("auto"),
                             name="placa", runtime="auto")

    def um_quadro() -> tuple[float, float, float]:
        t0 = time.perf_counter()
        det_obj.detect(quadro)
        t1 = time.perf_counter()
        for _ in range(args.rois):
            det_placa.detect(roi)
        t2 = time.perf_counter()
        return t1 - t0, (t2 - t1) / max(args.rois, 1), t2 - t0

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    energia = (subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()
               or [""])[0].strip()
    if "Battery" in energia:
        print("AVISO: o computador está na bateria. A bateria do artigo rodou na tomada.")

    ordem = [("A sem pausa", None, 0), ("B 30 quadros/s", 1 / FPS, 0),
             ("C 30 quadros/s, CPU ocupada", 1 / FPS, args.ocupados),
             ("A sem pausa, de novo", None, 0), ("B 30 quadros/s, de novo", 1 / FPS, 0)]
    regimes = {}
    for nome, periodo, ocupados in ordem:
        parar = mp.Event()
        procs = [mp.Process(target=_ocupar, args=(parar,), daemon=True) for _ in range(ocupados)]
        for p in procs:
            p.start()
        if procs:
            time.sleep(1.0)
        regimes[nome] = medir(um_quadro, args.quadros, periodo)
        parar.set()
        for p in procs:
            p.join(timeout=5)
        r = regimes[nome]
        print(f"{nome:30s} veículos {r['veiculos_ms']:6.2f} ms | placa por ROI {r['placa_por_roi_ms']:5.2f} ms"
              f" | quadro (1 + {args.rois}) {r['quadro_ms']:6.2f} ms, p99 {r['quadro_p99_ms']:.2f}", flush=True)
    rz = razoes(regimes)
    print(f"\nA 30 quadros/s, o quadro custa {rz['B_sobre_A']:.2f} vez o do regime sem pausa; "
          f"com a CPU ocupada, {rz['C_sobre_A']:.2f}.")

    resultado = {"chip": chip, "energia": energia, "python": platform.python_version(),
                 "rois_por_quadro": args.rois, "ocupados_no_regime_c": args.ocupados,
                 "regimes": regimes, "razoes": rz}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"gravado: {args.out}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn")
    raise SystemExit(main())
