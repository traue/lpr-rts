#!/usr/bin/env python3
"""Baixa e prepara os modelos neurais usados pelo LPR-RTS.

O que e baixado, e de onde:

* **YOLO11n / YOLO11s (deteccao de veiculos)** - pesos oficiais da Ultralytics,
  pre-treinados no COCO, que ja contem as classes car/motorcycle/bus/truck.
  Nao e preciso treinar nada para detectar veiculos.
* **Detector de placas** - nao existe peso oficial universal. Com ``--plate``,
  baixa um YOLO11 de placas de terceiros (Hugging Face, revisao fixada e sha256
  conferido) para ``data/models/plate_yolo.pt``: serve para testar o pipeline
  neural, nao para resultados da tese (ver ``PLATE_MODELS``). O caminho para
  resultados e treinar com dados brasileiros (``scripts/train_plate_detector.py``).
  Sem peso de placa, o sistema usa o detector classico automaticamente.
* **Exportacao CoreML** (opcional, apenas em Apple Silicon) - converte os ``.pt``
  (veiculos e placas) para ``.mlpackage``, que roda no Neural Engine. E o caminho
  mais rapido no Mac.

Tudo roda localmente depois do download; nenhuma inferencia sai da maquina.

Uso::

    python scripts/download_models.py                 # basico (YOLO11n)
    python scripts/download_models.py --model yolo11s # modelo maior, mais preciso
    python scripts/download_models.py --plate morsetech-n   # + detector de placas
    python scripts/download_models.py --plate morsetech-n --coreml  # tudo em CoreML (Mac)
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

MODELS_DIR = Path("data/models")
ULTRALYTICS_MODELS = {
    "yolo11n": "o mais rapido; recomendado para tempo real em CPU/ANE",
    "yolo11s": "equilibrado; melhor precisao com custo moderado",
    "yolo11m": "mais preciso; exige GPU para manter 30 fps",
}

# Detectores de placa de terceiros. YOLO11 ajustado no dataset publico
# "License Plate Recognition" da Roboflow (~10 mil imagens, placas de varios
# paises; poucas brasileiras). Licenca AGPL-3.0, a mesma do ultralytics. O
# proprio autor avisa que as metricas publicadas estao infladas (as mesmas
# imagens aparecem em treino e teste): serve para exercitar o pipeline neural e
# como ponto de comparacao, nao como resultado. Revisao fixada e sha256
# conferido, porque o mesmo nome de arquivo no Hugging Face pode mudar de
# conteudo - e a sessao grava so o sha256 dos pesos.
PLATE_REPO = "morsetechlab/yolov11-license-plate-detection"
PLATE_REVISION = "251a30d7daedca065f56e04b0af04052c907c68f"
PLATE_MODELS = {
    "morsetech-n": ("license-plate-finetune-v1n.pt",
                    "0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2",
                    "YOLO11n, 5 MB; o mais rapido"),
    "morsetech-s": ("license-plate-finetune-v1s.pt",
                    "95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381",
                    "YOLO11s, 19 MB; equilibrado"),
    "morsetech-m": ("license-plate-finetune-v1m.pt",
                    "d691f8d5e7709d2065b18a0e2bd75deaf08f6115cc0295cc3de7d0fd2eabcdae",
                    "YOLO11m, 41 MB; mais preciso, mais lento"),
}
PLATE_TARGET = MODELS_DIR / "plate_yolo.pt"


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _artifact_exists(path: str | Path | None) -> bool:
    """Um caminho retornado pelo exportador nao comprova que o artefato existe."""
    if path is None:
        return False
    p = Path(path)
    return (p.is_file() and p.stat().st_size > 0) or (
        p.is_dir() and any(f.is_file() and f.stat().st_size > 0 for f in p.rglob("*"))
    )


def _export_staged(weights: Path, format: str, imgsz: int, **kwargs) -> Path:
    """So substitui uma exportacao existente depois de concluir a nova."""
    from ultralytics import YOLO

    with TemporaryDirectory(prefix=".lprts-export-", dir=weights.parent) as directory:
        staged = Path(directory) / weights.name
        shutil.copy2(weights, staged)
        result = YOLO(str(staged)).export(format=format, imgsz=imgsz, **kwargs)
        if not _artifact_exists(result):
            raise RuntimeError(f"artefato {format} ausente ou vazio: {result}")
        path = Path(result)
        target = weights.with_suffix(".mlpackage" if format == "coreml" else ".onnx")
        backup = Path(directory) / (target.name + ".previous")
        if target.exists():
            target.replace(backup)
        try:
            path.replace(target)
        except OSError:
            if backup.exists():
                backup.replace(target)
            raise
    return target


def download_vehicle_model(name: str) -> Path | None:
    """Baixa os pesos COCO da Ultralytics (o download e feito pela propria lib)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERRO: ultralytics nao instalado.\n  pip install '.[dl]'")
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / f"{name}.pt"
    if _artifact_exists(target):
        print(f"  ja existe: {target}")
        return target

    print(f"  baixando {name}.pt ...")
    try:
        model = YOLO(f"{name}.pt")      # baixa para o cache da ultralytics
    except Exception as exc:  # noqa: BLE001
        print(f"  ERRO no download: {exc}")
        return None
    source = Path(getattr(model, "ckpt_path", "") or f"{name}.pt")
    if source.exists() and source.resolve() != target.resolve():
        shutil.move(str(source), str(target))
    if not _artifact_exists(target):
        print(f"  AVISO: nao foi possivel localizar o arquivo baixado de {name}")
        return None
    print(f"  OK: {target}")
    return target


def export_coreml(weights: Path, imgsz: int = 640) -> Path | None:
    """Exporta para CoreML (.mlpackage) - Neural Engine no Apple Silicon."""
    if not _is_apple_silicon():
        print("  (pulado: exportacao CoreML so faz sentido em Apple Silicon)")
        return None
    print(f"  exportando {weights.name} para CoreML (imgsz={imgsz}) ...")
    try:
        # SEM NMS embutido: com nms=True o modelo fixa os limiares da exportacao
        # e o ultralytics ignora o iou/conf configurados, sem avisar. Sem ele, o
        # NMS roda no pos-processamento com os parametros do YAML - mesma
        # latencia medida (M5 Pro: p50 5,1 ms nos dois; p99 5,7 x 6,8 ms).
        path = _export_staged(weights, "coreml", imgsz, nms=False, half=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERRO na exportacao CoreML: {exc}")
        print("  Dica: requer 'pip install coremltools'.")
        return None
    print(f"  OK: {path}")
    return Path(path)


def export_onnx(weights: Path, imgsz: int = 640) -> Path | None:
    """Exporta para ONNX - runtime portatil, latencia previsivel."""
    print(f"  exportando {weights.name} para ONNX (imgsz={imgsz}) ...")
    try:
        path = _export_staged(weights, "onnx", imgsz, opset=12, simplify=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERRO na exportacao ONNX: {exc}")
        return None
    print(f"  OK: {path}")
    return Path(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def download_plate_model(name: str, force: bool = False) -> Path | None:
    """Baixa um detector de placas de terceiros para ``data/models/plate_yolo.pt``.

    Nao sobrescreve um ``plate_yolo.pt`` diferente (pode ser um modelo treinado
    por voce) sem ``--force``. Ao substituir, apaga as exportacoes do peso
    anterior: um ``plate_yolo.mlpackage`` antigo seria usado no lugar do novo
    ``.pt`` (ver ``lprts.factory.coreml_weights``).
    """
    arquivo, sha_esperado, _ = PLATE_MODELS[name]
    if PLATE_TARGET.exists():
        if _sha256(PLATE_TARGET) == sha_esperado:
            print(f"  ja existe: {PLATE_TARGET} ({name})")
            return PLATE_TARGET
        if not force:
            print(f"  {PLATE_TARGET} ja existe e e outro modelo (treinado por voce?).\n"
                  "  Nada foi alterado; use --force para substitui-lo.")
            return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{PLATE_REPO}/resolve/{PLATE_REVISION}/{arquivo}"
    parcial = PLATE_TARGET.with_suffix(".part")
    print(f"  baixando {arquivo} de {PLATE_REPO} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, parcial.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except OSError as exc:
        parcial.unlink(missing_ok=True)
        print(f"  ERRO no download: {exc}")
        return None
    sha = _sha256(parcial)
    if sha != sha_esperado:
        parcial.unlink(missing_ok=True)
        print(f"  ERRO: sha256 {sha} difere do esperado {sha_esperado}; arquivo descartado.")
        return None

    for antigo in (PLATE_TARGET.with_suffix(".mlpackage"), PLATE_TARGET.with_suffix(".onnx")):
        if antigo.exists():
            print(f"  removendo exportacao do peso anterior: {antigo}")
            if antigo.is_dir():
                shutil.rmtree(antigo)
            else:
                antigo.unlink()
    parcial.replace(PLATE_TARGET)
    print(f"  OK: {PLATE_TARGET}  (sha256 {sha[:12]}..., licenca AGPL-3.0)")
    print("  Metricas publicadas pelo autor estao infladas (treino/teste contaminados):\n"
          "  meca no seu video com `lprts eval` antes de tirar conclusoes.")
    return PLATE_TARGET


def prepare_plate_detector() -> None:
    """Explica como obter o detector de placas (nao ha peso oficial universal)."""
    if PLATE_TARGET.exists():
        print(f"  ja existe: {PLATE_TARGET}")
        return
    print(
        "\n  Nao ha um modelo oficial e universal de deteccao de placas.\n"
        "  Voce tem tres caminhos:\n\n"
        "  (a) Usar o detector CLASSICO (funciona agora, sem download):\n"
        "        lprts run --source video.mp4 --plate-detector classic\n\n"
        "  (b) Baixar um YOLO de placas de terceiros (para testar):\n"
        "        python scripts/download_models.py --plate morsetech-n\n\n"
        "  (c) TREINAR um detector com dados brasileiros (melhor resultado):\n"
        "        python scripts/train_plate_detector.py --data data/plates.yaml\n"
        "      Datasets publicos uteis: UFPR-ALPR, RodoSol-ALPR, OpenALPR-BR,\n"
        "      ou anotacoes proprias no formato YOLO.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="yolo11n", choices=sorted(ULTRALYTICS_MODELS),
                    help="modelo de deteccao de veiculos")
    ap.add_argument("--plate", choices=sorted(PLATE_MODELS),
                    help="baixar um detector de placas de terceiros para plate_yolo.pt")
    ap.add_argument("--force", action="store_true",
                    help="substituir um plate_yolo.pt diferente do pedido")
    ap.add_argument("--coreml", action="store_true", help="exportar tambem para CoreML (Mac)")
    ap.add_argument("--onnx", action="store_true", help="exportar tambem para ONNX")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="entrada do detector de veiculos (detector.imgsz)")
    ap.add_argument("--plate-imgsz", type=int, default=320,
                    help="entrada do detector de placas (detector.plate_imgsz)")
    args = ap.parse_args()

    print("=" * 66)
    print("LPR-RTS - preparacao de modelos")
    print("=" * 66)
    print(f"\nDestino: {MODELS_DIR.resolve()}\n")

    print(f"[1/3] Detector de veiculos ({args.model}: {ULTRALYTICS_MODELS[args.model]})")
    weights = download_vehicle_model(args.model)
    failed = not _artifact_exists(weights)

    print("\n[2/3] Detector de placas")
    if args.plate:
        print(f"  {args.plate}: {PLATE_MODELS[args.plate][2]}")
        plate = download_plate_model(args.plate, force=args.force)
        failed |= not _artifact_exists(plate)
    else:
        prepare_plate_detector()
        plate = PLATE_TARGET if PLATE_TARGET.exists() else None

    # O .mlpackage tem entrada fixa e so e usado se ela for igual ao imgsz da
    # configuracao: cada detector e exportado no seu tamanho.
    print("\n[3/3] Exportacoes opcionais")
    exportar = [(w, sz) for w, sz in ((weights, args.imgsz), (plate, args.plate_imgsz)) if w]
    for w, sz in exportar:
        if args.coreml:
            failed |= not _artifact_exists(export_coreml(w, sz))
        if args.onnx:
            failed |= not _artifact_exists(export_onnx(w, sz))
    if not (args.coreml or args.onnx):
        print("  (nenhuma solicitada; use --coreml e/ou --onnx)")

    print("\n" + "=" * 66)
    print("Proximos passos:")
    print("  lprts doctor                     # conferir o ambiente e os modelos")
    print("  lprts demo                       # rodar tudo com video sintetico")
    print("  lprts run --source video.mp4     # rodar no seu video")
    print("=" * 66)
    if failed:
        print("ERRO: a preparacao solicitada nao foi concluida.")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
