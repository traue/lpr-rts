#!/usr/bin/env python3
"""Treina um detector YOLO de placas com dados proprios (Apple Silicon: MPS).

Formato esperado (padrao YOLO)::

    dataset/
      images/train/*.jpg      labels/train/*.txt
      images/val/*.jpg        labels/val/*.txt

Cada ``.txt`` tem uma linha por placa::

    0 <cx> <cy> <w> <h>        # normalizados em [0,1], classe 0 = placa

E o arquivo YAML do dataset::

    path: /caminho/para/dataset
    train: images/train
    val: images/val
    names: {0: placa}

Datasets publicos brasileiros uteis: **UFPR-ALPR** (4.500 imagens, 150 veiculos)
e **RodoSol-ALPR** (20.000 imagens, carros e motos, Mercosul e antigo). Ambos
exigem pedido de acesso academico aos autores.

Uso::

    python scripts/train_plate_detector.py --data data/placas.yaml --epochs 100
    python scripts/train_plate_detector.py --data data/placas.yaml --device mps --batch 16
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def resolve_device(preference: str) -> str:
    """Escolhe o dispositivo de treino disponivel."""
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if (
            platform.system() == "Darwin"
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="YAML do dataset no formato YOLO")
    ap.add_argument("--model", default="yolo11n.pt", help="pesos iniciais (transfer learning)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 ajusta automaticamente")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | 0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=25, help="parada antecipada")
    ap.add_argument("--name", default="plate_detector")
    ap.add_argument("--out", default="data/models/plate_yolo.pt", help="destino do melhor peso")
    ap.add_argument("--export-coreml", action="store_true", help="exportar para CoreML ao final")
    ap.add_argument("--export-imgsz", type=int, default=320,
                    help="entrada fixa do .mlpackage; use o detector.plate_imgsz da configuracao")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERRO: ultralytics nao instalado.\n  pip install '.[dl]'")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERRO: YAML do dataset nao encontrado: {data_path}")
        print("\nCrie um arquivo assim:\n")
        print("  path: /caminho/absoluto/para/dataset")
        print("  train: images/train")
        print("  val: images/val")
        print("  names:\n    0: placa")
        return 1

    device = resolve_device(args.device)
    print("=" * 66)
    print("LPR-RTS - treino do detector de placas")
    print("=" * 66)
    print(f"  dataset .....: {data_path}")
    print(f"  modelo base .: {args.model}")
    print(f"  epocas ......: {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}")
    print(f"  dispositivo .: {device}")
    if device == "mps":
        print("  (Apple GPU via Metal. Se houver instabilidade, use --device cpu.)")
    print()

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        project="runs/train",
        # Aumentos adequados a placas: variacao de cor/iluminacao ajuda muito;
        # espelhamento horizontal NAO - inverteria a ordem dos caracteres e
        # ensinaria o modelo a aceitar geometrias que nao existem.
        fliplr=0.0,
        flipud=0.0,
        degrees=7.0,
        translate=0.10,
        scale=0.45,
        shear=3.0,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file() or best.stat().st_size == 0:
        print(f"\nERRO: best.pt ausente ou vazio em {results.save_dir}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Prepare e valide todos os artefatos pedidos antes de substituir o modelo
    # anterior. Uma falha de exportacao conserva o conjunto que ja funcionava.
    with TemporaryDirectory(prefix=".lprts-train-", dir=out.parent) as directory:
        staged = Path(directory) / out.name
        shutil.copy2(best, staged)
        exported: Path | None = None
        if args.export_coreml:
            print("Exportando para CoreML...")
            try:
                # Sem NMS embutido (ver download_models.py): iou/conf do YAML continuam
                # valendo. A entrada do .mlpackage e fixa: exporte no tamanho em que o
                # detector de placas roda (detector.plate_imgsz), nao no do treino.
                path = YOLO(str(staged)).export(
                    format="coreml", imgsz=args.export_imgsz, nms=False, half=True
                )
                exported = Path(path)
                if not exported.is_dir() or not any(
                    f.is_file() and f.stat().st_size > 0 for f in exported.rglob("*")
                ):
                    raise RuntimeError(f"artefato CoreML ausente ou vazio: {exported}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ERRO na exportacao: {exc}")
                return 1

        # Backups no mesmo filesystem permitem restaurar o conjunto se uma
        # substituicao falhar; so sao descartados depois de publicar tudo.
        backups: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for old in (out, out.with_suffix(".mlpackage"), out.with_suffix(".onnx")):
                if old.exists():
                    backup = Path(directory) / (old.name + ".previous")
                    old.replace(backup)
                    backups.append((old, backup))
            staged.replace(out)
            published.append(out)
            if exported is not None:
                target = out.with_suffix(".mlpackage")
                exported.replace(target)
                published.append(target)
        except OSError:
            for path in published:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            for old, backup in reversed(backups):
                backup.replace(old)
            raise
    print(f"\nMelhor peso copiado para: {out}")
    print("\nTeste com:\n  lprts run --source video.mp4 --plate-detector yolo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
