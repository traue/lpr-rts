# Plano de análise do artigo (antes da bateria)

Escrito em 2026-11-15, antes de qualquer resultado da bateria, a pedido do autor: "as
imagens devem ser bem claras, mostrar uma boa análise e evolução de análise delas
(exemplo, tirar uma integral e interpretar...)", "precisa ter mais imagens no artigo, mais
gráficos, nem que isso sacrifique um pouco dos textos". O commit deste arquivo, anterior aos
bancos em `runs/artigo/`, é o registro de que as análises foram decididas antes dos dados.

Cada questão segue a mesma progressão: **descrever** os dados → **derivar** uma grandeza
com significado físico (uma integral, uma taxa) → **modelar** → **inferir** com incerteza →
**interpretar**. Código: `scripts/artigo/figuras.py` (figuras) e, depois da bateria,
`scripts/artigo/analise.py`.

## A. Os dados, antes do sistema (só o gabarito)

- **A1. Tamanho das placas** (Figura `dados`, pronta). Maior largura da placa de cada
  veículo, por vídeo, legível ou difícil, com o escopo (50 px) e o mínimo do OCR.
- **A2. Janela de leitura** (Figura `janela`, pronta). Para cada veículo legível e cada
  limiar w do OCR, T(w) = ∫ 1[largura(t) ≥ w] dt: quanto tempo a placa fica grande o bastante
  para o OCR tentar. Com o OCR a cada 2 quadros e 3 leituras para a primeira decisão, um
  veículo só pode ser lido se T(w) ≥ 6 quadros (0,2 s). A fração dos legíveis que cumpre isso
  é o **teto da revocação** previsto só pela anotação. Previsões registradas (teto, sem prazo,
  para as três resoluções de processamento da bateria):

  | Vídeo | 960 px | 1280 px (perfil) | 1920 px |
  |---|---|---|---|
  | V1 Paulista, dia (94 legíveis) | 15% | 36% | 73% |
  | V2 Paulista, noite (9) | 44% | 56% | 100% |
  | V3 Consolação, dia (55) | 53% | 73% | 98% |
  | V4 Consolação, vertical (15) | 60% | 93% | 100% |

## B. QP1 — onde vai o tempo

- **B1.** Latência por etapa (captura, fila, detecção, rastreamento, OCR, gravação) em p50,
  p95 e p99, por vídeo, nas cinco repetições de `base_rt` (tabela `frames`).
- **B2. Modelo de custo.** Tempo de serviço por quadro = a + b·n, com n = veículos com busca
  de placa no quadro; mínimos quadrados com IC por bootstrap em blocos de quadros. Comparar
  a e b com o custo isolado do detector de objetos e do detector de placas
  (`bench_componentes.json`): a soma dos componentes explica o observado?
- **B3. Fila.** Utilização ρ = λ·E[S] (taxa de chegada × tempo médio de serviço) e a lei de
  Little, L = λW, com L = (1/T)∫N(t)dt, o número médio de quadros no sistema
  [@little1961proof]. Relacionar ρ com a taxa de descarte nas três resoluções: quando ρ
  passa de 1, a fila só se mantém descartando.

## C. QP2 — cada otimização

Diferenças pareadas por veículo (com menos sem): Δ revocação com IC 95% por bootstrap sobre
veículos [@efron1979bootstrap], McNemar sem prazo [@mcnemar1947note], regressão logística
mista nas repetições em tempo real [@bates2015lme4], Holm [@holm1979simple]; Δ latência p99 e
Δ descarte. Figura em floresta no lugar da Tabela 1.

## D. QP3 — tempo real brando × sem prazo

- **D1.** Revocação por vídeo, sem prazo e em tempo real (as cinco repetições como pontos).
- **D2. Tempo até a decisão correta.** Curvas de Kaplan–Meier [@kaplan1958nonparametric]
  (veículo sem decisão correta censurado na saída da cena) e o tempo médio restrito
  RMST(τ) = ∫₀^τ S(t) dt [@royston2013restricted]: a área sob a curva é o tempo médio, até τ,
  em que o veículo segue sem placa correta publicada. Diferença de RMST com IC por bootstrap.

## E. Acerto × tamanho da placa (liga A a C e D)

Regressão logística P(placa correta | maior largura) por condição [@agresti2013categorical],
com faixa por bootstrap e a largura de 50% de acerto (w50). Sobrepor o teto de A2: a
diferença entre o teto e o observado é o que se perde por leitura errada, e não por falta de
tamanho.

## Figuras previstas (até 4 páginas sem referências; o texto cede espaço)

1. Pipeline (pronta). 2. Dados (A1, pronta). 3. Janela e teto (A2, pronta).
4. Tempo: B1 + B2 + B3. 5. Otimizações: C. 6. Tempo real × sem prazo: D1 + D2.
7. Acerto × largura: E.

## Referências novas (verificar no Stage 2.5)

- Little, J. D. C. (1961). A proof for the queuing formula: L = λW. *Operations Research*,
  9(3), 383–387. doi:10.1287/opre.9.3.383
- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations.
  *Journal of the American Statistical Association*, 53(282), 457–481.
  doi:10.1080/01621459.1958.10501452
- Royston, P., & Parmar, M. K. B. (2013). Restricted mean survival time: an alternative to
  the hazard ratio for the design and analysis of randomized trials with a time-to-event
  outcome. *BMC Medical Research Methodology*, 13, 152. doi:10.1186/1471-2288-13-152
- Agresti, A. (2013). *Categorical data analysis* (3rd ed.). Wiley.
