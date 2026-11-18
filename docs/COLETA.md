# Coleta: filmagem e anotação

Para quem vai filmar e anotar os vídeos de avaliação. O resultado é, por vídeo:

- o arquivo do vídeo;
- uma linha no registro de campo;
- o **gabarito** (*ground truth*): a placa certa de cada veículo, quadro a quadro.

O sistema é corrigido contra esse gabarito (revocação e precisão por veículo, ver
[AVALIACAO.md](AVALIACAO.md)).

São três etapas, nesta ordem:

1. filmar, preenchendo o registro de campo;
2. anotar o gabarito no CVAT, no computador, depois da filmagem;
3. converter a exportação do CVAT e conferir.

Na rua você **não** anota placas: só o registro de campo. As placas são marcadas
depois, assistindo ao vídeo.

---

## 0. Antes de sair

- **Ética e LGPD.** Placa é dado pessoal. Consulte o comitê de ética da instituição sobre
  filmar via pública, antes de começar. Vídeos, gabaritos e exportações ficam só no
  computador: `data/videos/` está fora do git. Ver [DADOS.md](DADOS.md) §6.
- **Equipamento:**
  - celular (iPhone em 4K a 30 quadros/s);
  - tripé com suporte de celular;
  - bateria extra;
  - espaço livre: o 4K em HEVC ocupa uns 170 MB por minuto.
- **Piloto.** Antes da coleta grande, faça a cadeia inteira com 1–2 minutos de um vídeo
  que já existe: anotar, converter, rodar o `lprts eval`. É onde aparecem os problemas.

## 1. Filmagem

**Ajustes da câmera:**
- Filme **na horizontal**. A vertical funciona no sistema, mas complica a anotação.
- 4K a 30 quadros/s. Desligue o Vídeo HDR (Ajustes > Câmera > Gravar Vídeo).
- Trave foco e exposição: segure o dedo na tela até aparecer "Bloqueio AE/AF".
- Obturador curto: 1/500 s ou mais rápido de dia; à noite, o mais curto que a luz
  permitir. É o que mais evita placa borrada. A câmera nativa do iPhone não deixa
  escolher o obturador; um app com controle manual (por exemplo, o Blackmagic Camera,
  gratuito) deixa.

**Posição:**
- Tripé na calçada, a ~1–1,5 m de altura, fora da via.
- Câmera apontada **ao longo** da via, pegando a frente ou a traseira dos carros, com no
  máximo ~30° de desvio. Não aponte de lado, perpendicular à rua.
- Os carros devem passar a uns 5–15 m.
- Enquadre **só** a pista que vai medir. Placa que aparece no quadro e dá para ler tem
  de ir para o gabarito (§3.6). As que ficam longe demais (pista oposta, carros parados
  ao fundo) não precisam de caixa, mas obrigam a avaliar com uma largura mínima
  declarada (`--scope-min-width`, [AVALIACAO.md](AVALIACAO.md) §3): sem elas no
  enquadramento, nada disso é preciso.
- No 4K, a placa precisa chegar a **uns 135 px de largura**, que é o mínimo que o OCR
  tenta ler (45 px na imagem processada a 1280 px). Grave 1 minuto de teste em cada local
  e confira; se a placa ficar pequena, aproxime ou use o zoom 2×.

**Durante a gravação:**
- Vídeo contínuo de 10–20 minutos por local e condição.
- Não mexa, não dê zoom e não toque no celular.
- Use o modo "Não perturbe", para não chegarem notificações.
- Se algo acontecer (câmera tremeu, alguém tapou a vista), anote o minuto no registro.

**Quantidade:**
- O que conta é veículo com placa anotada: no mínimo 50, idealmente 200 ou mais.
  Com 200 veículos, o intervalo de confiança de 95% de uma revocação de 80% fica em
  ±5,5 pontos.
- Pelo menos 3 locais, de dia e à noite (chuva, se der). Por exemplo: 3 locais × 2
  condições, com ~35 veículos em cada vídeo.

## 2. Registro de campo

Uma linha por vídeo. O modelo é [modelos/registro_de_campo.csv](modelos/registro_de_campo.csv),
que abre no Excel ou no Numbers. Guarde a cópia preenchida em
`data/videos/registro_de_campo.csv`.

| Coluna | O que escrever | Exemplo |
|---|---|---|
| `arquivo` | nome do arquivo do vídeo | `V1.MOV` |
| `data`, `hora_inicio`, `hora_fim` | quando gravou | `2026-11-10`, `17:05`, `17:22` |
| `local`, `latitude`, `longitude` | onde | `Av. Exemplo, 100 (sentido centro)` |
| `altura_camera_m`, `distancia_faixa_m` | posição da câmera, aproximada | `1,2`, `4` |
| `angulo_via_graus` | desvio em relação ao eixo da via | `20` |
| `placas` | `dianteiras` ou `traseiras` | `traseiras` |
| `resolucao`, `fps` | da gravação | `3840x2160`, `30` |
| `obturador`, `iso`, `zoom` | se o app mostrar | `1/500`, `auto`, `1x` |
| `luz` | `dia`, `entardecer` ou `noite` | `entardecer` |
| `clima` | `limpo`, `nublado` ou `chuva` | `nublado` |
| `trafego` | `baixo`, `medio` ou `alto` | `medio` |
| `observacoes` | o que aconteceu | `ônibus parou na frente 05:30–06:10` |

Esses dados descrevem os vídeos no artigo: local, condição, duração e resolução.

## 3. Anotação no CVAT

### 3.1 O que é o CVAT

O CVAT é um programa gratuito e de código aberto para marcar objetos em vídeo. Você
desenha uma caixa em volta da placa num quadro, avança alguns quadros e ajusta a caixa.
O CVAT preenche os quadros do meio sozinho (**interpolação**) e mantém a mesma
identidade do veículo do começo ao fim (uma **trilha**). Ele roda no navegador, mas
instalado no seu computador.

**Não use a versão na nuvem (app.cvat.ai):** os vídeos têm placas e não devem sair da
sua máquina.

### 3.2 Instalar (uma vez)

1. Instale o **Docker Desktop** para Mac (docker.com) e abra-o.
   - As imagens do CVAT são só para processador Intel. Num Mac com chip Apple, elas
     rodam por emulação, mais devagar mas funcionando.
   - Se o Docker Desktop tiver a opção "Use Rosetta for x86_64/amd64 emulation" em
     Settings > General, deixe ligada.
2. No Terminal:

   ```bash
   git clone https://github.com/cvat-ai/cvat
   cd cvat
   docker compose up -d
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   ```

   - A primeira subida baixa alguns GB.
   - O último comando pede um nome de usuário e uma senha.
3. Abra o **Google Chrome**, o único navegador suportado, em
   <http://localhost:8080> e entre com esse usuário.
4. Para desligar: `docker compose stop`, dentro da pasta `cvat`. Para religar:
   `docker compose up -d`.
5. Anote a versão do CVAT, que aparece em "About": ela vai para o método do artigo.

### 3.3 Criar o projeto (uma vez)

Em **Projects > +**, crie o projeto `LPR-RTS` com **um** rótulo, `placa`, e três
atributos:

| Atributo | Tipo no CVAT | Valores | "Mutable" |
|---|---|---|---|
| `texto` | Text | — (a placa digitada) | **não** |
| `tipo` | Select | `carro`, `motocicleta`, `caminhao`, `onibus` (padrão `carro`) | **não** |
| `dificil` | Checkbox | padrão desmarcado | **sim** |

Os nomes precisam ser exatamente esses: é o que o conversor procura. "Mutable" quer dizer
que o valor pode mudar ao longo da trilha. Só o `dificil` muda, porque uma placa pode
ficar legível no meio do caminho.

### 3.4 Criar a tarefa (uma por vídeo)

1. Em **Tasks > + > Create a new task**, dê o nome do vídeo (por exemplo, `V1`),
   escolha o projeto `LPR-RTS` e envie o arquivo do vídeo, pela opção "My computer".
2. Em **Advanced configuration**, **não** preencha "Start frame", "Stop frame" nem
   "Frame step". A tarefa precisa ser o vídeo inteiro, com a mesma numeração de quadros
   que o LPR-RTS usa.
3. Clique em **Submit** e abra o *job* da tarefa.

### 3.5 Anotar um veículo

1. Vá ao **primeiro** quadro em que a placa aparece, mesmo pequena ou cortada pela
   borda, e não ao quadro em que você reparou nela. Se ali ainda não dá para ler, marque
   `dificil` e desmarque no quadro em que ela fica legível.
2. Na barra da esquerda, escolha o retângulo, o rótulo `placa` e **Track**, não "Shape".
   Desenhe uma caixa justa em volta da **placa**, não do carro.
3. No painel da direita, preencha `texto` (os 7 caracteres, por exemplo `ABC1D23`) e
   `tipo`.
4. Avance alguns quadros (`F` avança um quadro, `V` avança vários; `D` e `C` voltam) e
   arraste a caixa para a posição nova da placa, acertando também o tamanho. Cada ajuste
   cria um quadro-chave, e o CVAT interpola entre eles **em linha reta**.
   - De quantos em quantos quadros: de 10 em 10, mais ou menos. No piloto de 2026-11,
     com um quadro-chave a cada 10 quadros o refino do §4 recuperou 91% das posições
     que tinham sido marcadas à mão; a cada 15, 86%; a cada 20, 77%.
   - Nos quadros **do meio** a caixa escapa da placa. O refino do §4 a põe de volta,
     desde que os seus quadros-chave estejam certos: capriche neles. Onde o refino não
     conseguir, ele avisa, e aí o ajuste é à mão.
5. Siga a placa **até ela sair do quadro**, não até ficar pequena: quando não der mais
   para ler, marque `dificil` e continue. No quadro em que ela sai, marque a trilha como
   **fora de cena** ("outside", o ícone de olho no painel do objeto). Sem essa marca, o
   CVAT deixa a caixa parada até o fim do vídeo. O conversor corta no último ajuste e
   avisa.
6. Se algo passa na frente (um pedestre, outro veículo), marque fora de cena no quadro
   em que a placa some e **desmarque** no quadro em que ela volta, na mesma trilha. Não
   encerre a trilha na oclusão.
7. Repita para o próximo veículo, sempre com uma trilha nova.
8. Salve com frequência (`Ctrl+S`). `F1` mostra os atalhos.

### 3.6 Regras do gabarito

- **Anote todos os veículos com placa visível, não só os fáceis.** Se o sistema ler
  certo um veículo que ficou fora do gabarito, a leitura conta como erro. Vale para os
  veículos que já estavam em cena no primeiro quadro e para os que passam ao longe na
  pista medida.
- **Placas longe demais para qualquer leitura não precisam de caixa** (a pista oposta,
  os carros parados ao fundo). Quem as tira da conta é a avaliação, com
  `--scope-min-width` ([AVALIACAO.md](AVALIACAO.md) §3): abaixo da largura declarada,
  o que não tem anotação é ignorado. No piloto de 2026-11, numa avenida com a pista
  oposta no enquadramento, 48 dos 82 eventos do perfil de avaliação caíram nessas
  placas, todas com menos de 50 px no 4K.
- **O vídeo que vai para a avaliação é anotado inteiro.** Se ele for longo demais (num
  local movimentado, 2–3 minutos já dão dezenas de veículos), corte o arquivo a partir do
  início, sem recodificar, e use o arquivo cortado no CVAT e no `lprts`
  ([artigo/CONTINUAR.md](artigo/CONTINUAR.md) §3). Esse corte mantém a numeração dos
  quadros: o que já foi anotado no começo do vídeo continua valendo. Se a tarefa do
  CVAT já é o vídeo inteiro, não precisa refazê-la: o conversor grava só o trecho do
  corte com `--ate` (§4).
- **Digite a placa lendo a imagem, sem olhar o que o sistema leu.** Senão o gabarito
  herda os erros do sistema. Na Mercosul (LLLNLNN), o 4º caractere é número e o 5º é
  letra: `0` e `I`, nunca `D` e `1`, nessas posições.
- **`dificil` é a placa que você não consegue ler ou cujo texto não dá para afirmar com
  certeza** (suja, danificada, borrada, encoberta). Se ela ficar legível depois,
  desmarque a partir daquele quadro. Difícil o tempo todo, o veículo sai da métrica,
  como no protocolo PASCAL VOC: o que o sistema fizer com ele não conta como acerto nem
  como erro. Pode ficar sem texto ou com o seu melhor palpite; o conversor avisa do
  segundo caso só para você confirmar que foi de propósito.
- **`dificil` só existe numa caixa.** Placa sem caixa não é difícil: para a avaliação
  ela não existe, e o que o sistema ler ali é erro (a não ser abaixo da largura mínima,
  acima).
- **Distância sozinha não é motivo.** Se você leu a placa com segurança, ela não é
  difícil, por menor que esteja. Cada veículo excluído por `dificil` tem de ser
  declarado no artigo, com o critério.
- **Um veículo, uma trilha.** Se a trilha se partiu, por exemplo depois de um ônibus
  passar na frente, junte as duas com **Merge** (`M`). Senão a avaliação conta dois
  veículos.
- **Concordância.** Anote uns 20% dos veículos duas vezes, numa segunda tarefa com o
  mesmo vídeo, sem olhar a primeira (outra pessoa, ou você em outro dia).

### 3.7 Exportar

Na tarefa, **Actions > Export task dataset**:

- formato **CVAT for video 1.1**;
- **desmarque** "Save images": as imagens ocupariam muito espaço e têm placas.

O resultado é um `.zip`. A exportação feita pelo *job* (`job_<n>_annotations_<data>.zip`)
também serve, desde que a tarefa tenha um *job* só.

## 4. Refinar, converter e conferir

```bash
python scripts/refinar_caixas.py data/videos/cvat/revisado/V1_cvat_revisado.zip \
    --video data/videos/originais/V1.MOV --out data/videos/cvat/V1_refinado.zip
python scripts/cvat_para_gt.py data/videos/cvat/V1_refinado.zip \
    --out data/videos/gabaritos/V1_gt.csv --video data/videos/originais/V1.MOV
rm data/videos/cvat/V1_refinado.zip   # intermediário
```

**O refino** (`refinar_caixas.py`) põe de volta em cima da placa as caixas dos quadros
que você não ajustou:
- em cada vão entre dois quadros-chave, segue a placa no vídeo, de um quadro-chave até o
  outro e de volta, pela semelhança da imagem;
- só aceita o resultado que **fecha**: saindo de um quadro-chave, tem de chegar na caixa
  que você pôs no seguinte. Senão o vão fica como estava, e o script avisa;
- não muda os seus quadros-chave, o texto nem o `dificil`, não cria trilhas e não usa o
  detector do LPR-RTS: o gabarito não pode depender do sistema que ele avalia;
- a saída é outra exportação no mesmo formato. Dá para subi-la de volta ao CVAT e
  continuar dali: dentro do job, **Menu > Upload annotations**, formato **CVAT 1.1**,
  modo **Replace**. O CVAT troca as anotações do job pelas do arquivo e só guarda os
  quadros-chave, por isso o refino grava como quadro-chave cada caixa que reposiciona.
  Exporte antes, para ter para onde voltar.

É anotação semiautomática: declare no método do artigo. No piloto de 2026-11, a fração
de caixas interpoladas em cima da placa foi de 88,8% para 95,8%.

Se a tarefa do CVAT é o vídeo inteiro e a avaliação é num corte dele (§3.6), passe o
arquivo cortado em `--video` e acrescente `--ate N` ao conversor, com N igual ao número
de quadros do corte: ele grava só os quadros 0 a N−1.

**O conversor:**
- interpola os quadros entre os ajustes;
- respeita o "fora de cena";
- normaliza o texto;
- grava o CSV que o `lprts eval` lê ([AVALIACAO.md](AVALIACAO.md) §1).

**Ele recusa, e não grava nada:**
- trilha sem texto e sem `dificil`;
- texto que muda dentro da trilha.

**Ele avisa:**
- texto fora do padrão Mercosul (LLLNLNN) ou antigo (LLLNNNN), provavelmente um erro de
  digitação;
- a mesma placa em duas trilhas;
- trilha sem marca de saída;
- trilha com texto e todos os quadros marcados `dificil`;
- vãos entre quadros-chave em que a caixa interpolada deve escapar da placa (uma
  estimativa, pela curvatura do movimento entre três quadros-chave seguidos);
- com `--video`, número de quadros ou orientação diferentes dos do vídeo.

Corrija no CVAT, exporte de novo e repita.

**No piloto**, confira com o próprio sistema:

```bash
lprts eval -s data/videos/originais/V1.MOV --gt data/videos/gabaritos/V1_gt.csv -c configs/tempo_real.yaml \
    --offline --db runs/piloto.db --out runs/piloto.json
```

Com a anotação ainda pela metade, acrescente `--max-frames N`, com N no ponto até onde
tudo já foi anotado: o que o sistema ler depois dali contaria como erro. Isso só vale
offline e para conferir; em tempo real o `--max-frames` conta quadros processados e
passa do trecho. Para os números do artigo, o arquivo é cortado (§3.6).

O que olhar no relatório:

- **Veículos anotados que o sistema publicou certo.** Um número muito baixo costuma ser
  erro de anotação: caixa no carro em vez da placa, ou tarefa com outra numeração de
  quadros.
- **"Eventos sem veículo anotado"** e tentativas **"sem placa anotada"**. Muitos deles
  querem dizer gabarito incompleto: veículo que faltou, trilha que começa tarde ou
  termina cedo, placas ao fundo. `piloto_eventos.csv` traz o quadro de cada um.
- **Confusões que se repetem num veículo só** (em `piloto_veiculos.csv`, `fused_text`
  diferente de `plate_text` por um caractere): reveja a imagem. Pode ser o OCR, pode ser
  o texto digitado.

Use o piloto para achar o que faltou, mas digite cada placa lendo a imagem. Depois,
apague `runs/piloto*`: o banco, o relatório e as tabelas têm placas.

## 5. Onde fica cada coisa

| O quê | Onde | Vai para o git? |
|---|---|---|
| Vídeos | `data/videos/originais/` e `cortes/` | não (dado pessoal) |
| Gabaritos (`*_gt.csv`) | `data/videos/gabaritos/` | não (têm placas) |
| Exportações do CVAT | `data/videos/cvat/revisado/` (pré-anotação em `cvat/preanotacao/`) | não (têm placas); entram no pacote de dados |
| Registro de campo | `data/videos/registro_de_campo.csv` | não, por padrão |
| Anotações dentro do CVAT | volumes do Docker | não. Exporte antes de apagar: `docker compose down -v` apaga tudo |

Deixe o disco do Mac com o FileVault ligado. Nada disso vai para serviços de nuvem.
