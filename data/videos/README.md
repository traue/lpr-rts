# Vídeos

Coloque aqui seus vídeos de teste. **Vídeos reais não vão para o git** (são
grandes e mostram placas de pessoas reais); só os sintéticos gerados pelo
projeto (`sintetico*.mp4`, com `_gt.csv` e `_meta.json`) são versionados.

## Vídeo sintético com ground truth

`sintetico.mp4` (400 quadros, 4 veículos, semente 42) acompanha o repositório.
O gerador é determinístico no conteúdo: o comando abaixo — o mesmo do `./start.sh`
e do `make synthetic` — recria a mesma anotação e os mesmos quadros desenhados (com a
mesma versão do OpenCV e a mesma fonte; ambas ficam no `_meta.json`). Os bytes do mp4
podem diferir um pouco entre máquinas (codificação: 0,03 de diferença média por pixel
num Apple M4 Pro), e isso basta para mudar 2 dos 9 veículos do baseline clássico. Por
isso o **vídeo versionado é a referência**: o `lprts demo` o usa quando ele existe.

```bash
python scripts/make_synthetic_dataset.py --out data/videos/sintetico.mp4 \
    --frames 400 --vehicles 4
```

Gera também `sintetico_gt.csv` (anotações exatas: placa, objeto, `track_id`,
`difficult`) e `sintetico_meta.json` (parâmetros, fonte usada, commit do código).
Sintéticos servem para ablação e teste; **os resultados do artigo precisam de
vídeo real**.

## Vídeos reais (celular, câmera)

Vídeos de iPhone (`.MOV`, HEVC, 4K, inclusive na vertical) funcionam direto: a
rotação gravada no arquivo é aplicada e o processamento acontece a 1280 px de
largura (`source.resize_width`). Pela interface, abra o arquivo em **Abrir...** — a
caixa **Veículos (YOLO)** é marcada sozinha. Pela linha de comando:

```bash
lprts run --source data/videos/originais/IMG_0834.MOV            # com interface
lprts run --source data/videos/originais/IMG_0834.MOV --headless # so grava o banco
```

Para o YOLO de placas, rode antes `python scripts/download_models.py --plate
morsetech-n --coreml`. **Vídeos reais ficam fora do git** (`*.mov`, `*.mp4`): mostram
placas de pessoas reais (dado pessoal, LGPD). Pelo mesmo motivo, pseudonimize as
placas antes de compartilhar uma exportação (receita em `docs/DADOS.md`).

## Organização da pasta (vídeos reais do artigo)

Nada daqui, além deste README e dos sintéticos, vai para o git: os vídeos, os gabaritos e as
exportações do CVAT estão no Zenodo, em <https://doi.org/10.5281/zenodo.24351664>.
Baixados, eles voltam para estas pastas.

| Pasta | O que tem |
|---|---|
| `originais/` | os vídeos como saíram da câmera (`IMG_*.MOV`) |
| `cortes/` | os trechos avaliados dos vídeos longos (`IMG_1179c.MOV`, `IMG_1184b.MOV`: os primeiros 360 s) |
| `gabaritos/` | o ground truth de cada trecho avaliado (`*_gt.csv`), gerado pelo `scripts/cvat_para_gt.py` |
| `cvat/revisado/` | a exportação do CVAT revisada pelo autor, de onde sai cada gabarito (`IMG_*_cvat_revisado.zip`) |
| `cvat/preanotacao/` | a pré-anotação gerada a partir das detecções do sistema, como foi importada no CVAT antes da revisão |
| `cvat/historico/` | a anotação à mão do autor e estados intermediários, para a proveniência |

O passo a passo de cada vídeo (refinar, converter, conferir) está em
[`docs/COLETA.md`](../../docs/COLETA.md).

## Fontes de vídeo aceitas

| Fonte | Exemplo |
|---|---|
| Arquivo | `--source data/videos/rodovia.mp4` |
| Webcam | `--source 0` |
| RTSP | `--source rtsp://usuario:senha@192.168.1.50/stream` |
| Pasta de imagens | `--source data/videos/sequencia/` |

## Recomendações para captura

Protocolo completo para o artigo (filmagem, registro de campo e anotação no CVAT) em
[`docs/COLETA.md`](../../docs/COLETA.md).

- **Resolução**: o OCR só tenta placas com ao menos **45 px de largura na imagem
  processada** (`ocr.min_plate_width`, a 1280 px de largura): ~68 px num vídeo 1080p e
  ~135 px num 4K. Confira num teste de 1 minuto em cada local.
- **Ângulo**: até ~30° de desvio frontal. Acima disso a perspectiva distorce os
  caracteres além do que a retificação corrige.
- **Obturador**: prefira tempo curto (1/500 s ou menos) — desfoque de movimento é o
  fator que mais degrada o OCR em veículos rápidos.
- **Taxa de quadros**: 30 fps dá margem para a fusão multi-frame acumular leituras.
