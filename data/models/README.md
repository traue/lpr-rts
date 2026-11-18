# Modelos

Pesos de rede neural. Esta pasta é ignorada pelo git (arquivos grandes).

## Baixar

```bash
python scripts/download_models.py --plate morsetech-n --coreml
```

## Arquivos esperados

| Arquivo | Origem | Necessário? |
|---|---|---|
| `yolo11n.pt` | Ultralytics (COCO) | opcional — sem ele, modo só-placa |
| `plate_yolo.pt` | treinado por você **ou** `--plate` (terceiros) | opcional — sem ele, detector clássico |
| `yolo11n.mlpackage` | exportado para CoreML (sem NMS embutido) | opcional — no Apple Silicon, `device: auto` o usa no lugar do `.pt` (latência p99 do pipeline ~2× menor) |
| `plate_yolo.mlpackage` | exportado para CoreML com entrada de 320 px (`detector.plate_imgsz`) | opcional — idem, para o detector de placas |

## Detector de placas de terceiros (`--plate`)

YOLO11 ajustado por [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
no dataset público *License Plate Recognition* da Roboflow (~10 mil imagens, vários
países, poucas placas brasileiras). Licença **AGPL-3.0**. O autor avisa que as
métricas publicadas estão **infladas** (as mesmas imagens aparecem em treino e
teste): use para testar o pipeline neural e como ponto de comparação, não como
resultado da tese.

O script fixa a revisão do repositório e confere o sha256. Cada sessão grava o
sha256 dos pesos (`sessions.environment_json`); por ele se identifica o modelo:

| Opção | Arquivo de origem | sha256 (início; completo em `scripts/download_models.py`) |
|---|---|---|
| `morsetech-n` | `license-plate-finetune-v1n.pt` | `0aec75976c56eb6f` |
| `morsetech-s` | `license-plate-finetune-v1s.pt` | `95e50c25ab7066dd` |
| `morsetech-m` | `license-plate-finetune-v1m.pt` | `d691f8d5e7709d20` |

Um `plate_yolo.pt` diferente (treinado por você) não é sobrescrito sem `--force`; ao
substituir, as exportações do peso anterior são apagadas.

**Entrada de 320 px e modo só-placa.** O detector de placas roda no recorte de cada
veículo, em 320 px. Sem detector de veículos (`--no-vehicle-detector`, ou o fallback
automático), ele roda no quadro inteiro e 320 px perde as placas pequenas: no vídeo
sintético, a revocação das caixas cai de 0,57 (640 px) para 0,08. Nesse modo use
`detector.plate_imgsz: 640` e exporte com `--plate-imgsz 640`.

Um `.mlpackage` exportado com `nms=True` (versões anteriores do script) fixa os
limiares da exportação e o `iou` configurado deixa de valer: reexporte com o
comando acima. O detector avisa e grava os parâmetros efetivos.

**O sistema roda sem nenhum destes arquivos**, usando o detector morfológico e o OCR
por gabaritos como baseline. Veja a seção 8 do README para treinar o detector de
placas com dados brasileiros.
