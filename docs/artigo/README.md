# Materiais de reprodução do artigo

Esta pasta tem o que é preciso para conferir e refazer os números do artigo "LPR-RTS: leitura de
placas veiculares em tempo real brando em hardware de consumo". O texto do artigo não está aqui.

| Arquivo | O que é |
|---|---|
| `resultados/analise.json` | todos os números do artigo, gerados por `scripts/artigo/analise.py` a partir dos bancos da bateria e dos gabaritos, sem texto de placa |
| `resultados/diagnostico_ritmo.json` | a medida do ritmo das chamadas das redes (`scripts/artigo/diagnostico_ritmo.py`), feita na máquina da bateria |
| `resultados/diagnostico_anotacao.json` | os tempos menores ou iguais a zero e a comparação entre os veículos anotados à mão e os pré-anotados (`scripts/artigo/diagnostico_anotacao.py`) |
| `plano_analise_registrado.md` | o plano de análise como foi registrado em 15/11/2026, antes da bateria (commit `a523d3f` do repositório de desenvolvimento) |

Os dados estão no Zenodo, em <https://doi.org/10.5281/zenodo.24351664>, e os passos para refazer tudo, na seção "Reproduzir o
artigo" do [README](../../README.md).
Cada sessão grava o commit do código em que rodou (`sessions.code_version`). São commits do
repositório de desenvolvimento, que não é público (`6669cbe` na bateria e `6399c13` nas rodadas
refeitas); o código do sistema deste repositório é idêntico ao deles.
As figuras que os scripts refazem vão para `figuras/`, nesta pasta, que fica fora do git.

## O que mudou em relação ao plano registrado

O plano foi seguido, com as mudanças abaixo. O artigo declara as principais.

- Emenda antes da bateria (15/11, commit `6669cbe`): na revisão do autor, duas placas do V3
  ganharam texto, e uma moto ficou legível. O V3 passou a 56 veículos legíveis, e o teto foi
  recalculado para 52%, 71% e 96% nas três resoluções.
- Sono do computador: na primeira bateria (15 e 16/11), o Mac dormiu entre rodadas e, em
  DarkWake, tarefas do macOS contaminaram algumas medidas de tempo.
  - Uma sessão é considerada suspensa quando a duração no relógio de parede passa a do relógio
    monotônico em mais de 1 s. As 6 sessões suspensas que restaram, de 160, saem das análises de
    tempo.
  - O V4 inteiro e, no V3, a linha de base em tempo real e a variante sem a O4 foram refeitos em
    16/11 (commit `6399c13`), com o Power Nap desligado e a tela acesa. As rodadas antigas ficam
    em `runs/artigo/V*/descartadas/`, nos dados publicados.
  - A acurácia não mudou: em tempo real, as cinco repetições deram o mesmo resultado por veículo.
- Latência nas comparações: usa-se a mediana do p99 entre as repetições, que resiste a um desvio
  isolado numa delas.
- Fila: a análise passou das três resoluções às 28 condições em tempo real. Como nenhuma utilização
  passa de 0,64, não há regime de descarte a relacionar com ela.
- Modelo misto: não se aplica. Com as repetições idênticas por veículo, a variância entre elas é
  nula. Usa-se o McNemar exato sobre a decisão da maioria das repetições, com IC por *bootstrap*
  estratificado por vídeo.
- Acerto por largura: a logística registrada não descreve os dados, porque força o acerto a chegar
  a 100%, e acima do mínimo do OCR ele para em cerca de 89%. O modelo usado é um degrau: zero
  abaixo do mínimo do OCR, convertido em largura anotada, e um patamar acima, com IC de Wilson. A
  logística registrada continua no `analise.json`.
- Teto da revocação: além do teto registrado, há um teto corrigido. Ele usa a largura da caixa
  detectada, que é a que o OCR usa (mediana de 1,05 vez a anotada), e 5 quadros em vez de 6,
  porque, com o OCR a cada 2 quadros, as 3 leituras cabem em t, t+2 e t+4. O registrado fica como
  está.
- Análises exploratórias, fora do plano e feitas depois da bateria, sem nova medida do sistema:
  - o ritmo das chamadas das redes (`diagnostico_ritmo.json`);
  - os tempos menores ou iguais a zero e o efeito da pré-anotação (`diagnostico_anotacao.json`);
  - no `analise.json`:
    - a detecção em tempo real dividida pela sem prazo (`f_exploratorias`);
    - as placas publicadas erradas por veículo;
    - a faixa entre repetições;
    - o serviço por repetição.

Os comentários de alguns scripts citam documentos do repositório de desenvolvimento que não fazem
parte desta publicação. O plano que eles citam é o desta pasta.
