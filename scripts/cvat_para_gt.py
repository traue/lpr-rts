#!/usr/bin/env python3
"""Converte a exportacao "CVAT for video 1.1" do CVAT no ground truth do LPR-RTS.

Uso::

    python scripts/cvat_para_gt.py exportacao.zip --out data/videos/gabaritos/V1_gt.csv
    python scripts/cvat_para_gt.py annotations.xml --out data/videos/gabaritos/V1_gt.csv \\
        --video data/videos/originais/V1.MOV
    python scripts/cvat_para_gt.py exportacao.zip --out data/videos/gabaritos/V1a_gt.csv \\
        --video data/videos/cortes/V1a.MOV --ate 3391

Como anotar no CVAT para este script: ``docs/COLETA.md``. Em resumo, um rotulo
``placa`` (retangulo) anotado como **trilha** - uma por veiculo - com os
atributos ``texto`` (a placa lida), ``tipo`` (carro, motocicleta, caminhao,
onibus) e ``dificil`` (caixa de selecao, que pode mudar ao longo da trilha).

O que o script faz:

* escreve uma linha ``kind=plate`` por quadro em que a placa aparece, com o
  ``track_id`` da trilha do CVAT, que e a identidade do veiculo na avaliacao;
* interpola linearmente os quadros entre duas caixas da trilha: a exportacao pode
  trazer so os quadros-chave, e a avaliacao pareia leitura e anotacao por quadro;
* respeita "fora de cena" (outside): nenhum quadro de uma caixa marcada fora ate
  a proxima visivel, e entre uma caixa visivel e uma marcada fora a caixa fica
  parada, como no CVAT;
* nao estende a trilha alem do ultimo quadro-chave. Sem a marca de saida, o CVAT
  deixaria a caixa parada ate o fim do video;
* normaliza o texto (maiusculas, sem hifen) e confere o padrao Mercosul
  (LLLNLNN) ou antigo (LLLNNNN);
* com ``--ate N``, grava so os quadros 0 a N-1: o gabarito de um corte do video
  feito a partir do inicio, quando a tarefa do CVAT e o video inteiro. O corte
  sem recodificar mantem a numeracao dos quadros (``docs/artigo/CONTINUAR.md``).

Erros (saida 1, nada e gravado): trilha nao dificil sem texto, texto que muda
dentro da trilha, nenhuma trilha com o rotulo. Avisos: texto fora do padrao, a
mesma placa em duas trilhas (o mesmo veiculo partido em dois?), trilha sem a
marca de saida, trilha com texto e todos os quadros dificeis, vaos entre
quadros-chave em que a caixa interpolada deve escapar da placa e, com
``--video``, numero de quadros ou orientacao diferentes dos do video como o
LPR-RTS o le.

O arquivo de saida tem textos de placa (dado pessoal): fica fora do git
(``data/videos/`` e ignorado) e nao vai para o repositorio publico.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lprts.eval.groundtruth import GroundTruth, GTAnnotation, save_ground_truth  # noqa: E402
from lprts.ocr.plates_br import is_valid, normalize  # noqa: E402

TIPOS = ("carro", "motocicleta", "caminhao", "onibus")
# Grafias aceitas no atributo "tipo" alem das canonicas.
SINONIMOS = {"moto": "motocicleta", "caminhão": "caminhao", "ônibus": "onibus",
             "onibús": "onibus", "car": "carro", "motorcycle": "motocicleta",
             "truck": "caminhao", "bus": "onibus"}
VERDADEIRO = ("1", "true", "sim", "yes")
# Desvio da caixa, em larguras dela, que derruba o IoU com a placa para 0,5 - o
# limiar com que a avaliacao casa leitura e anotacao: (1 - f) / (1 + f) = 0,5.
DESVIO_MAX = 1 / 3


@dataclass(slots=True)
class Caixa:
    """Uma caixa da trilha, como veio do XML."""

    quadro: int
    caixa: tuple[float, float, float, float]
    fora: bool
    chave: bool
    atributos: dict[str, str]


@dataclass(slots=True)
class Conversao:
    """Resultado da conversao: anotacoes, problemas e numeros para conferir."""

    anotacoes: list[GTAnnotation] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    trilhas: int = 0
    interpoladas: int = 0
    dificeis: int = 0
    cortadas: int = 0          # com --ate: trilhas que atravessam o corte
    fora_do_corte: int = 0     # com --ate: trilhas inteiras depois dele
    quadros_no_cvat: int | None = None
    tamanho_no_cvat: tuple[int, int] | None = None


def ler_xml(caminho: Path) -> ET.Element:
    """Raiz do ``annotations.xml``, de um ``.xml`` ou de dentro do ``.zip`` exportado."""
    if zipfile.is_zipfile(caminho):
        with zipfile.ZipFile(caminho) as z:
            nomes = [n for n in z.namelist() if n.endswith("annotations.xml")]
            if not nomes:
                raise ValueError(f"{caminho} nao tem annotations.xml: exporte em 'CVAT for video 1.1'.")
            return ET.fromstring(z.read(min(nomes, key=len)))
    return ET.parse(caminho).getroot()


def _inteiro(elemento: ET.Element | None) -> int | None:
    if elemento is None or not (elemento.text or "").strip():
        return None
    return int(float(elemento.text.strip()))


def ler_meta(raiz: ET.Element) -> tuple[int | None, tuple[int, int] | None]:
    """Numero de quadros e tamanho (largura, altura) da tarefa ou do trabalho exportado."""
    for no in ("meta/task", "meta/job", "meta/project/tasks/task"):
        base = raiz.find(no)
        if base is None:
            continue
        quadros = _inteiro(base.find("size"))
        # A exportacao de um trabalho (job) traz o tamanho direto em <meta>, ao lado
        # de <job>, e nao dentro dele: sem olhar ali, a conferencia de orientacao
        # era pulada em silencio.
        for origem in (base, raiz.find("meta")):
            if origem is None:
                continue
            largura = _inteiro(origem.find("original_size/width"))
            altura = _inteiro(origem.find("original_size/height"))
            if largura and altura:
                return quadros, (largura, altura)
        return quadros, None
    return None, None


def ler_trilha(no: ET.Element) -> list[Caixa]:
    """As caixas de uma ``<track>``, em ordem de quadro."""
    caixas = []
    for box in no.findall("box"):
        atributos = {a.get("name", ""): (a.text or "").strip() for a in box.findall("attribute")}
        caixas.append(Caixa(
            quadro=int(box.get("frame", "0")),
            caixa=(float(box.get("xtl", "0")), float(box.get("ytl", "0")),
                   float(box.get("xbr", "0")), float(box.get("ybr", "0"))),
            fora=box.get("outside", "0") == "1",
            # Exportacoes sem o atributo: toda caixa vale como quadro-chave.
            chave=box.get("keyframe", "1") == "1",
            atributos=atributos,
        ))
    return sorted(caixas, key=lambda c: c.quadro)


def expandir(caixas: list[Caixa]) -> tuple[list[Caixa], int, bool]:
    """Caixas visiveis quadro a quadro: interpola os vazios e corta no ultimo quadro-chave.

    Devolve ``(visiveis, interpoladas, sem_saida)``. ``sem_saida`` diz que a trilha
    nao termina numa caixa marcada fora de cena.
    """
    chaves = [c for c in caixas if c.chave] or caixas
    ultimo = max(c.quadro for c in chaves)
    uteis = [c for c in caixas if c.quadro <= ultimo]
    sem_saida = not uteis[-1].fora
    visiveis: list[Caixa] = []
    interpoladas = 0
    for a, b in zip(uteis, [*uteis[1:], None], strict=True):
        if a.fora:
            continue
        visiveis.append(a)
        if b is None:
            continue
        for q in range(a.quadro + 1, b.quadro):
            if b.fora:
                caixa = a.caixa
            else:
                t = (q - a.quadro) / (b.quadro - a.quadro)
                caixa = tuple(pa + t * (pb - pa) for pa, pb in zip(a.caixa, b.caixa, strict=True))
            visiveis.append(Caixa(q, caixa, False, False, a.atributos))  # type: ignore[arg-type]
            interpoladas += 1
    return visiveis, interpoladas, sem_saida


def _centro(c: Caixa) -> tuple[float, float]:
    return (c.caixa[0] + c.caixa[2]) / 2, (c.caixa[1] + c.caixa[3]) / 2


def _desvios(chaves: list[Caixa]) -> list[tuple[int, int, float]]:
    """Vaos de uma sequencia de quadros-chave visiveis com desvio acima de ``DESVIO_MAX``."""
    # Aceleracao aparente (px/quadro^2) em cada quadro-chave do meio: o erro da
    # reta A->C em B e a * g1 * g2 / 2. Trincas com vao de um quadro ficam de fora:
    # nelas o tremor da mao pesa mais que o movimento.
    aceleracao: dict[int, float] = {}
    for a, b, c in zip(chaves, chaves[1:], chaves[2:], strict=False):
        g1, g2 = b.quadro - a.quadro, c.quadro - b.quadro
        if g1 * g2 < 4:
            continue
        t = g1 / (g1 + g2)
        (ax, ay), (bx, by), (cx, cy) = _centro(a), _centro(b), _centro(c)
        erro = math.hypot(bx - (ax + t * (cx - ax)), by - (ay + t * (cy - ay)))
        aceleracao[b.quadro] = 2 * erro / (g1 * g2)
    achados = []
    for a, b in zip(chaves, chaves[1:], strict=False):
        vao = b.quadro - a.quadro
        vizinhas = [aceleracao[q] for q in (a.quadro, b.quadro) if q in aceleracao]
        if vao < 2 or not vizinhas:
            continue
        largura = min(a.caixa[2] - a.caixa[0], b.caixa[2] - b.caixa[0])
        # No meio de um vao de g quadros, a reta erra a * g^2 / 8.
        desvio = max(vizinhas) * vao * vao / 8 / max(largura, 1e-9)
        if desvio >= DESVIO_MAX:
            achados.append((a.quadro, b.quadro, desvio))
    return achados


def desvios_da_interpolacao(caixas: list[Caixa]) -> list[tuple[int, int, float]]:
    """Vaos entre quadros-chave em que a caixa interpolada deve escapar da placa.

    A interpolacao e uma reta; a placa nao anda em linha reta a velocidade constante
    (perspectiva, freada, moto costurando). Tres quadros-chave seguidos A, B, C medem
    essa curvatura: e o quanto a reta A->C erraria a caixa que o anotador pos em B.
    Com aceleracao constante o erro cresce com o quadrado do vao, o que da o desvio
    esperado no meio de cada vao vizinho. Devolve ``(quadro inicial, quadro final,
    desvio em larguras da caixa)`` dos vaos que passam de ``DESVIO_MAX``.

    E uma estimativa, para dizer onde olhar. No piloto de 2026-11 (22 trilhas, video
    4K de avenida), 65% dos quadros interpolados dos vaos apontados tinham IoU < 0,5
    com a placa detectada pelo sistema, contra 8% dos demais.
    """
    achados: list[tuple[int, int, float]] = []
    sequencia: list[Caixa] = []
    for c in caixas:
        if c.fora:
            # Fora de cena nao ha interpolacao: a curvatura nao atravessa a lacuna.
            achados.extend(_desvios(sequencia))
            sequencia = []
        elif c.chave:
            sequencia.append(c)
    achados.extend(_desvios(sequencia))
    return achados


def _tipo(valor: str, tid: int, avisos: list[str]) -> str:
    bruto = valor.strip().lower()
    tipo = SINONIMOS.get(bruto, bruto) or "carro"
    if tipo not in TIPOS:
        avisos.append(f"trilha {tid}: tipo {valor!r} desconhecido; gravado como 'carro'.")
        return "carro"
    return tipo


def converter(raiz: ET.Element, rotulo: str = "placa", attr_texto: str = "texto",
              attr_tipo: str = "tipo", attr_dificil: str = "dificil",
              ate: int | None = None) -> Conversao:
    """Converte a arvore XML exportada pelo CVAT nas anotacoes do LPR-RTS.

    ``ate`` limita o gabarito aos quadros ``0..ate-1`` (um corte do video a partir
    do inicio): trilhas inteiras depois dele saem, e as que o atravessam terminam
    no ultimo quadro do corte, sem o aviso de falta da marca de saida.
    """
    res = Conversao()
    res.quadros_no_cvat, res.tamanho_no_cvat = ler_meta(raiz)
    if ate is not None and res.quadros_no_cvat:
        res.quadros_no_cvat = min(res.quadros_no_cvat, ate)
    outros: dict[str, int] = {}
    textos: dict[str, list[int]] = {}
    for no in raiz.findall("track"):
        label = no.get("label", "")
        if label != rotulo:
            outros[label] = outros.get(label, 0) + 1
            continue
        tid = int(no.get("id", "0"))
        caixas = ler_trilha(no)
        if not caixas:
            continue
        visiveis, n_interp, sem_saida = expandir(caixas)
        cortada = False
        if ate is not None:
            # Expande a trilha inteira e so entao corta: a caixa do ultimo quadro do
            # corte sai da interpolacao rumo ao quadro-chave que ficou depois dele.
            cortada = any(c.quadro >= ate for c in caixas)
            visiveis = [c for c in visiveis if c.quadro < ate]
            exportados = {c.quadro for c in caixas}
            n_interp = sum(1 for c in visiveis if c.quadro not in exportados)
        if not visiveis:
            res.fora_do_corte += cortada
            continue
        res.trilhas += 1
        res.cortadas += cortada
        res.interpoladas += n_interp
        if sem_saida and not cortada:
            res.avisos.append(
                f"trilha {tid}: termina no quadro {visiveis[-1].quadro} sem a marca de saida "
                "(fora de cena); nao foi estendida ate o fim do video. Confira se a placa sai ali.")
        lidos = {normalize(c.atributos.get(attr_texto, "")) for c in visiveis} - {""}
        dificil = [c.atributos.get(attr_dificil, "").lower() in VERDADEIRO for c in visiveis]
        if len(lidos) > 1:
            res.erros.append(f"trilha {tid}: o texto muda dentro da trilha ({', '.join(sorted(lidos))}).")
            continue
        texto = lidos.pop() if lidos else ""
        if not texto and not all(dificil):
            res.erros.append(
                f"trilha {tid} (quadros {visiveis[0].quadro}-{visiveis[-1].quadro}): sem texto e "
                "sem a marca 'dificil'. Digite a placa ou marque a trilha como dificil.")
            continue
        if texto and not is_valid(texto):
            res.avisos.append(f"trilha {tid}: {texto!r} nao segue o padrao Mercosul nem o antigo.")
        if texto:
            textos.setdefault(texto, []).append(tid)
        if all(dificil):
            res.dificeis += 1
            if texto:
                res.avisos.append(
                    f"trilha {tid}: tem o texto {texto} mas todos os quadros estao marcados "
                    "'dificil', e o veiculo sai da metrica. Se o texto e um palpite, esta "
                    "certo assim; se voce leu a placa com seguranca em algum quadro, "
                    "desmarque nele.")
        desvios = [d for d in desvios_da_interpolacao(caixas) if ate is None or d[0] < ate]
        if desvios:
            vaos = ", ".join(f"{a}-{b}" for a, b, _ in desvios)
            pior = max(d for _, _, d in desvios)
            res.avisos.append(
                f"trilha {tid}: a caixa interpolada deve escapar da placa entre os "
                f"quadros-chave {vaos} (desvio estimado de ate {pior:.1f} largura(s) da caixa). "
                "Crie quadros-chave no meio desses vaos.")
        tipo = _tipo(visiveis[0].atributos.get(attr_tipo, ""), tid, res.avisos)
        for c, dif in zip(visiveis, dificil, strict=True):
            res.anotacoes.append(GTAnnotation(
                frame_index=c.quadro, bbox=tuple(round(v, 1) for v in c.caixa),  # type: ignore[arg-type]
                plate_text=texto, object_class=tipo, track_id=tid, difficult=dif, kind="plate",
            ))
    for texto, ids in textos.items():
        if len(ids) > 1:
            res.avisos.append(
                f"a placa {texto} esta em {len(ids)} trilhas ({', '.join(map(str, ids))}): se for "
                "o mesmo veiculo, junte as trilhas no CVAT; senao a avaliacao conta dois veiculos.")
    for label, n in sorted(outros.items()):
        res.avisos.append(f"{n} trilha(s) com o rotulo {label!r} ignorada(s) (so '{rotulo}' entra).")
    if not res.trilhas and not res.erros:
        res.erros.append(f"nenhuma trilha com o rotulo {rotulo!r}: anotou como trilha (Track)?")
    res.anotacoes.sort(key=lambda a: (a.frame_index, a.track_id or 0))
    return res


def conferir_video(res: Conversao, video: Path) -> list[str]:
    """Compara numero de quadros e orientacao do CVAT com o video lido pelo LPR-RTS."""
    from lprts.capture.source import VideoSource

    try:
        fonte = VideoSource(str(video)).open()
    except (FileNotFoundError, RuntimeError) as exc:
        return [f"nao consegui abrir o video para conferir: {exc}"]
    try:
        # Dimensoes do video girado (como o player mostra): o espaco do ground truth.
        largura, altura, quadros = fonte.width, fonte.height, fonte.frame_count
    finally:
        fonte.close()
    avisos = []
    if res.tamanho_no_cvat and res.tamanho_no_cvat != (largura, altura):
        if res.tamanho_no_cvat == (altura, largura):
            avisos.append(
                f"orientacao diferente: o CVAT viu {res.tamanho_no_cvat[0]}x{res.tamanho_no_cvat[1]} "
                f"e o LPR-RTS ve {largura}x{altura}. As caixas nao batem; filme na horizontal "
                "ou envie ao CVAT o video ja girado.")
        else:
            avisos.append(f"tamanho diferente: CVAT {res.tamanho_no_cvat}, video {(largura, altura)}.")
    if res.quadros_no_cvat and quadros and abs(res.quadros_no_cvat - quadros) > 1:
        avisos.append(
            f"numero de quadros diferente: CVAT {res.quadros_no_cvat}, video {quadros}. A tarefa "
            "do CVAT precisa ser o video inteiro, sem 'start frame', 'stop frame' nem 'frame step'.")
    if res.anotacoes and quadros and max(a.frame_index for a in res.anotacoes) >= quadros:
        avisos.append("ha caixas depois do ultimo quadro do video: confira se a tarefa e deste video.")
    return avisos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", type=Path, help="exportacao do CVAT (.zip ou annotations.xml)")
    ap.add_argument("--out", type=Path, required=True, help="CSV de ground truth a gravar")
    ap.add_argument("--video", type=Path, help="video anotado, para conferir quadros e orientacao")
    ap.add_argument("--rotulo", default="placa", help="rotulo das placas no CVAT")
    ap.add_argument("--ate", type=int, metavar="N",
                    help="grava so os quadros 0 a N-1: gabarito de um corte do video a partir do inicio")
    args = ap.parse_args(argv)
    if args.ate is not None and args.ate < 1:
        ap.error("--ate deve ser >= 1")

    res = converter(ler_xml(args.entrada), rotulo=args.rotulo, ate=args.ate)
    if args.video:
        res.avisos.extend(conferir_video(res, args.video))
    for aviso in res.avisos:
        print(f"AVISO: {aviso}")
    if res.erros:
        for erro in res.erros:
            print(f"ERRO: {erro}")
        print(f"\nNada foi gravado: corrija {len(res.erros)} erro(s) no CVAT e exporte de novo.")
        return 1
    save_ground_truth(GroundTruth(video=args.out.stem, annotations=res.anotacoes), args.out)
    quadros = len({a.frame_index for a in res.anotacoes})
    print(f"\n{args.out}: {res.trilhas} veiculos, {len(res.anotacoes)} caixas "
          f"({res.interpoladas} interpoladas) em {quadros} quadros; "
          f"{res.dificeis} veiculo(s) inteiro(s) marcado(s) como dificil.")
    if args.ate is not None:
        print(f"Corte no quadro {args.ate}: {res.cortadas} trilha(s) terminam no ultimo quadro do "
              f"corte e {res.fora_do_corte} ficaram de fora, depois dele.")
    print("O arquivo tem textos de placa: fica so nesta maquina (data/videos/ esta fora do git).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
