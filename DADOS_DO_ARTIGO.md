# Dados do artigo

Os vídeos, os gabaritos, as exportações do CVAT e os bancos das sessões do artigo ficam fora do
git: somam alguns gigabytes e contêm placas reais. Eles estão aqui:

<!-- Cole abaixo o endereço da pasta dos dados (Google Drive, Zenodo ou outro). -->
**Endereço dos dados:** <COLE AQUI O ENDEREÇO>

Na mesma pasta, o arquivo `SHA256SUMS` traz o sha256 de cada arquivo. Para conferir o que foi
baixado, rode `shasum -a 256 -c SHA256SUMS` dentro da pasta.

## O que há na pasta

| Pasta | O que tem | Para onde vai, neste repositório |
|---|---|---|
| `originais/` | os quatro vídeos como saíram da câmera (`IMG_1179.MOV`, `IMG_1184.MOV`, `IMG_0834.MOV`, `IMG_0833.MOV`) | `data/videos/originais/` |
| `cortes/` | os trechos avaliados dos dois vídeos longos, os primeiros 360 s (`IMG_1179c.MOV`, `IMG_1184b.MOV`) | `data/videos/cortes/` |
| `gabaritos/` | o ground truth de cada trecho avaliado (`IMG_1179c_gt.csv`, `IMG_1184b_gt.csv`, `IMG_0834_gt.csv`, `IMG_0833_gt.csv`) | `data/videos/gabaritos/` |
| `cvat/` | as exportações do CVAT: a revisada pelo autor (`revisado/`), a pré-anotação (`preanotacao/`) e o histórico (`historico/`) | `data/videos/cvat/` |
| `runs/artigo/` | os bancos SQLite das 160 sessões da bateria, os relatórios de cada rodada e o `bench_componentes.json` de cada vídeo | `runs/artigo/` |

## Os quatro vídeos

São vídeos de rua de São Paulo, gravados com um iPhone 17 em 4K HEVC, a 30 quadros/s, com a
câmera na calçada, a 1 m de altura e a 1 m da pista.

| Vídeo | Arquivo | Trecho avaliado | Veículos | Na métrica | Difíceis |
|---|---|---|---|---|---|
| V1 | `IMG_1179` (Av. Paulista, de dia) | os primeiros 360 s (`cortes/IMG_1179c.MOV`) | 137 | 94 | 43 |
| V2 | `IMG_1184` (Av. Paulista, à noite) | os primeiros 360 s (`cortes/IMG_1184b.MOV`) | 106 | 9 | 97 |
| V3 | `IMG_0834` (R. da Consolação, de dia) | o vídeo inteiro, 92 s | 74 | 56 | 18 |
| V4 | `IMG_0833` (R. da Consolação, de dia, na vertical) | o vídeo inteiro, 27 s | 16 | 15 | 1 |

Ficam na métrica as placas legíveis com ao menos 50 px de largura no quadro original. Difícil é a
placa cujo texto não se afirma com certeza. Dos 333 veículos, 37 foram anotados à mão pelo autor
e 296 pré-anotados a partir das detecções do sistema e revisados por ele.

## Como reproduzir

1. Instale o projeto (README, seção 4) e baixe os modelos:
   `python scripts/download_models.py --plate morsetech-n --coreml`.
2. Baixe os dados e ponha cada pasta no lugar indicado na tabela acima.
3. Para refazer a análise a partir dos bancos publicados, rode `python scripts/artigo/analise.py`.
   Ele grava o `docs/artigo/resultados/analise.json` e as Figs. 3 a 5 em `docs/artigo/figuras/`.
   O resultado deve bater com o `analise.json` versionado neste repositório.
4. Para refazer a Fig. 2, que sai só dos gabaritos, rode `python scripts/artigo/figuras.py fig2`.
5. Para refazer a bateria a partir dos vídeos, rode, para cada vídeo:

   ```bash
   scripts/bateria_artigo.sh data/videos/cortes/IMG_1179c.MOV data/videos/gabaritos/IMG_1179c_gt.csv V1 50
   scripts/bateria_artigo.sh data/videos/cortes/IMG_1184b.MOV data/videos/gabaritos/IMG_1184b_gt.csv V2 50
   scripts/bateria_artigo.sh data/videos/originais/IMG_0834.MOV data/videos/gabaritos/IMG_0834_gt.csv V3 50
   scripts/bateria_artigo.sh data/videos/originais/IMG_0833.MOV data/videos/gabaritos/IMG_0833_gt.csv V4 50
   ```

   São 40 rodadas por vídeo, cerca de 10 h no total. As medidas de tempo dependem da máquina: as
   do artigo são de um Apple M4 Pro (12 núcleos, 24 GB), na tomada, com os outros programas
   fechados e o Power Nap desligado. O cabeçalho do script lista os cuidados.

## Proveniência

Este repositório é uma cópia selecionada do repositório de desenvolvimento, sem o histórico dele.
Cada sessão grava o commit do código em que rodou (`sessions.code_version`). Esses commits são do
repositório de desenvolvimento:

- `a523d3f` (15/11/2026): o plano de análise registrado antes da bateria
  ([`docs/artigo/plano_analise_registrado.md`](docs/artigo/plano_analise_registrado.md));
- `6669cbe` (15 e 16/11/2026): a bateria;
- `6399c13` (16/11/2026): o V4 e parte do V3, refeitos depois de o computador dormir entre
  rodadas.

O código do sistema (`src/`, `configs/`, `scripts/bateria_artigo.sh` e os scripts de conversão
das anotações) é idêntico ao desses dois commits da bateria. Os scripts de análise
(`scripts/artigo/`) mudaram depois da bateria, e são os desta publicação que geraram o
`analise.json`.

## Licença e condições de uso

<!-- Licença proposta; troque se preferir outra. -->
Os dados estão sob a licença **Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0
Internacional** (CC BY-NC-SA 4.0): <https://creativecommons.org/licenses/by-nc-sa/4.0/deed.pt-br>.

Os vídeos, os gabaritos e os bancos trazem placas de veículos reais, que tratamos como dados
pessoais (Lei 13.709/2018, a LGPD). Eles são publicados porque a reprodução do artigo depende
deles. Ao usá-los, você se compromete a:

- usá-los só para pesquisa e ensino;
- não tentar identificar os donos ou os condutores dos veículos;
- não cruzá-los com outras bases para esse fim;
- não redistribuir os vídeos fora destas condições.

Pedidos de remoção de um trecho ou de uma placa vão para o autor, pelo contato do artigo.
