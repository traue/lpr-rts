#!/usr/bin/env python3
"""Reposiciona, no video, as caixas interpoladas de uma exportacao "CVAT for video 1.1".

Uso::

    python scripts/refinar_caixas.py data/videos/cvat/revisado/V1_cvat_revisado.zip \\
        --video data/videos/originais/V1.MOV --out data/videos/cvat/V1_refinado.zip
    python scripts/cvat_para_gt.py data/videos/cvat/V1_refinado.zip \\
        --out data/videos/gabaritos/V1_gt.csv --video data/videos/originais/V1.MOV

Por que existe: o CVAT interpola a caixa em linha reta entre dois quadros-chave, e a
placa nao anda em linha reta a velocidade constante (perspectiva, freada, moto
costurando). Nos quadros do meio a caixa escapa da placa, e ali a avaliacao deixa de
casar leitura e anotacao. Ajustar a mao todo quadro de todo veiculo rapido custa horas.

O que o script faz, em cada vao entre dois quadros-chave visiveis seguidos (A e B):

* segue a placa de A para frente e de B para tras, quadro a quadro, procurando no
  quadro seguinte o recorte da placa do quadro anterior (correlacao cruzada
  normalizada), em volta de onde a interpolacao linear a poria;
* so aceita o rastreio que **fecha**: saindo de A, ele tem de chegar na caixa que o
  anotador pos em B (a menos de ``--fechamento`` larguras), e vice-versa. Senao o vao
  fica como estava, na interpolacao linear, e o script avisa;
* grava as caixas reposicionadas como quadros-chave. Os quadros-chave do anotador nao
  mudam, e o tamanho da caixa continua o interpolado entre eles.

A saida e outra exportacao no mesmo formato: serve para o ``cvat_para_gt.py`` e para
subir de volta ao CVAT (Upload annotations).

O que ele nao faz: nao usa o detector de placas do LPR-RTS (o gabarito nao pode
depender do sistema que ele avalia), nao cria nem estende trilhas e nao mexe em texto
ou em ``dificil``. E anotacao semiautomatica: declare no metodo do artigo.

Conferido no piloto de 2026-11 (35 trilhas, 4K de avenida, 2.673 quadros
interpolados): a fracao com IoU >= 0,5 contra a placa detectada pelo sistema foi de
88,8% para 95,8%, e nenhum quadro piorou mais de 0,2 de IoU. O detector entrou so
nessa conferencia, nunca na posicao das caixas.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LARGURA_DE_TRABALHO = 64.0   # px: placas maiores sao reduzidas ate esta largura
MARGEM_X, MARGEM_Y = 3.0, 1.5   # recorte guardado: larguras e alturas em volta da reta
BUSCA_X, BUSCA_Y = 0.9, 0.7     # janela de busca por passo: larguras e alturas


@dataclass(slots=True)
class Vao:
    """Um vao entre dois quadros-chave visiveis seguidos de uma trilha."""

    trilha: int
    a: int
    b: int
    caixa_a: np.ndarray
    caixa_b: np.ndarray
    modelo: ET.Element                      # caixa de A: molde das que faltarem no XML
    no: ET.Element                          # a <track>
    escala: float = 1.0
    recortes: dict[int, tuple[np.ndarray, int, int]] = field(default_factory=dict)

    def reta(self, quadro: int) -> np.ndarray:
        """A caixa da interpolacao linear no quadro."""
        return self.caixa_a + (self.caixa_b - self.caixa_a) * (quadro - self.a) / (self.b - self.a)


@dataclass(slots=True)
class Resultado:
    """O que aconteceu com cada vao: para o resumo e para os testes."""

    vaos: int = 0
    aceitos: Counter = field(default_factory=Counter)      # "ambos" | "frente" | "tras"
    recusados: list[tuple[int, int, int, float, float]] = field(default_factory=list)
    sem_video: int = 0                                      # vaos depois do fim do video
    caixas: int = 0                                         # caixas reposicionadas
    corrigidos: int = 0     # vaos em que a caixa andou mais de 1/3 de largura


def _caixa(box: ET.Element) -> np.ndarray:
    return np.array([float(box.get(k, "0")) for k in ("xtl", "ytl", "xbr", "ybr")])


def _centro(caixa: np.ndarray) -> np.ndarray:
    return np.array([(caixa[0] + caixa[2]) / 2, (caixa[1] + caixa[3]) / 2])


def listar_vaos(raiz: ET.Element, rotulo: str | None = None) -> list[Vao]:
    """Vaos de pelo menos um quadro interpolado, de todas as trilhas (ou so de ``rotulo``)."""
    vaos = []
    for no in raiz.findall("track"):
        if rotulo is not None and no.get("label") != rotulo:
            continue
        sequencia: list[ET.Element] = []
        sequencias = [sequencia]
        for box in sorted(no.findall("box"), key=lambda b: int(b.get("frame", "0"))):
            if box.get("outside", "0") == "1":
                # Fora de cena nao ha o que seguir: o vao nao atravessa a lacuna.
                sequencia = []
                sequencias.append(sequencia)
            elif box.get("keyframe", "1") == "1":
                sequencia.append(box)
        for chaves in sequencias:
            for box_a, box_b in zip(chaves, chaves[1:], strict=False):
                a, b = int(box_a.get("frame", "0")), int(box_b.get("frame", "0"))
                if b - a < 2:
                    continue
                caixa_a, caixa_b = _caixa(box_a), _caixa(box_b)
                largura = max(caixa_a[2] - caixa_a[0], caixa_b[2] - caixa_b[0])
                vaos.append(Vao(int(no.get("id", "0")), a, b, caixa_a, caixa_b, box_a, no,
                                escala=min(1.0, LARGURA_DE_TRABALHO / max(largura, 1e-9))))
    return vaos


def _guardar(vao: Vao, quadro: int, cinza: np.ndarray) -> None:
    """Guarda o pedaco do quadro em volta da reta, ja na escala de trabalho."""
    c = vao.reta(quadro)
    largura = max(vao.caixa_a[2] - vao.caixa_a[0], vao.caixa_b[2] - vao.caixa_b[0])
    altura = max(vao.caixa_a[3] - vao.caixa_a[1], vao.caixa_b[3] - vao.caixa_b[1])
    mx, my = max(60.0, MARGEM_X * largura), max(30.0, MARGEM_Y * altura)
    alto, largo = cinza.shape
    x1, y1 = int(max(0, np.floor(c[0] - mx))), int(max(0, np.floor(c[1] - my)))
    x2, y2 = int(min(largo, np.ceil(c[2] + mx))), int(min(alto, np.ceil(c[3] + my)))
    recorte = cinza[y1:y2, x1:x2]
    if vao.escala < 1.0 and recorte.size:
        recorte = cv2.resize(recorte, None, fx=vao.escala, fy=vao.escala,
                             interpolation=cv2.INTER_AREA)
    vao.recortes[quadro] = (recorte, x1, y1)


def _placa(vao: Vao, quadro: int, centro: np.ndarray) -> np.ndarray | None:
    """Recorte da placa (tamanho da reta) centrado em ``centro``; ``None`` se sai do guardado."""
    recorte, ox, oy = vao.recortes[quadro]
    reta = vao.reta(quadro)
    w = max(4, int(round((reta[2] - reta[0]) * vao.escala)))
    h = max(4, int(round((reta[3] - reta[1]) * vao.escala)))
    x = int(round((centro[0] - ox) * vao.escala - w / 2))
    y = int(round((centro[1] - oy) * vao.escala - h / 2))
    if x < 0 or y < 0 or x + w > recorte.shape[1] or y + h > recorte.shape[0]:
        return None
    return recorte[y:y + h, x:x + w]


def _seguir(vao: Vao, ordem: list[int]) -> tuple[dict[int, np.ndarray], bool]:
    """Centro da placa em cada quadro de ``ordem``, partindo do quadro-chave ``ordem[0]``.

    O segundo valor diz se todo passo teve o que casar. Sem isso o caminho devolvido
    e so a reta, que "fecha" por construcao: nao pode valer como rastreio.
    """
    centros = {ordem[0]: _centro(vao.reta(ordem[0]))}
    casou = True
    for anterior, quadro in zip(ordem, ordem[1:], strict=False):
        # Movimento previsto: o da interpolacao linear entre os dois quadros.
        previsto = centros[anterior] + _centro(vao.reta(quadro)) - _centro(vao.reta(anterior))
        modelo = _placa(vao, anterior, centros[anterior])
        centros[quadro] = previsto
        if modelo is None or float(modelo.std()) < 2.0:
            casou = False                    # saiu do recorte, ou nao ha textura para casar
            continue
        recorte, ox, oy = vao.recortes[quadro]
        mh, mw = modelo.shape
        cx, cy = (previsto[0] - ox) * vao.escala, (previsto[1] - oy) * vao.escala
        x1 = int(max(0, np.floor(cx - mw / 2 - BUSCA_X * mw)))
        y1 = int(max(0, np.floor(cy - mh / 2 - BUSCA_Y * mh)))
        x2 = int(min(recorte.shape[1], np.ceil(cx + mw / 2 + BUSCA_X * mw)))
        y2 = int(min(recorte.shape[0], np.ceil(cy + mh / 2 + BUSCA_Y * mh)))
        busca = recorte[y1:y2, x1:x2]
        if busca.shape[0] < mh or busca.shape[1] < mw:
            casou = False
            continue
        _, _, _, onde = cv2.minMaxLoc(cv2.matchTemplate(busca, modelo, cv2.TM_CCOEFF_NORMED))
        centros[quadro] = np.array([(x1 + onde[0] + mw / 2) / vao.escala + ox,
                                    (y1 + onde[1] + mh / 2) / vao.escala + oy])
    return centros, casou


def resolver(vao: Vao, fechamento: float) -> tuple[str, float, float, dict[int, np.ndarray]]:
    """Segue a placa nos dois sentidos e devolve ``(aceito, erro_frente, erro_tras, centros)``.

    ``aceito`` e "ambos", "frente", "tras" ou "" (nenhum sentido fechou; sem centros).
    Os erros de fechamento estao em larguras da caixa.
    """
    quadros = list(range(vao.a, vao.b + 1))
    (frente, casou_f), (tras, casou_t) = _seguir(vao, quadros), _seguir(vao, quadros[::-1])
    largura = min(vao.caixa_a[2] - vao.caixa_a[0], vao.caixa_b[2] - vao.caixa_b[0])
    centro_a, centro_b = _centro(vao.caixa_a), _centro(vao.caixa_b)
    erro_f = float(np.linalg.norm(frente[vao.b] - centro_b)) / max(largura, 1e-9)
    erro_t = float(np.linalg.norm(tras[vao.a] - centro_a)) / max(largura, 1e-9)
    fecha_f, fecha_t = casou_f and erro_f <= fechamento, casou_t and erro_t <= fechamento
    if not (fecha_f or fecha_t):
        return "", erro_f, erro_t, {}
    centros = {}
    for q in quadros[1:-1]:
        peso = (q - vao.a) / (vao.b - vao.a)            # 0 em A, 1 em B
        if fecha_f and fecha_t:
            if float(np.linalg.norm(frente[q] - tras[q])) <= 0.3 * largura:
                centros[q] = frente[q] * (1 - peso) + tras[q] * peso
            else:
                # Os sentidos discordam: vale o que partiu do quadro-chave mais perto.
                centros[q] = frente[q] if peso < 0.5 else tras[q]
        elif fecha_f:
            # Um sentido so: o erro de fechamento e repartido ao longo do vao.
            centros[q] = frente[q] - (frente[vao.b] - centro_b) * peso
        else:
            centros[q] = tras[q] - (tras[vao.a] - centro_a) * (1 - peso)
    return ("ambos" if fecha_f and fecha_t else "frente" if fecha_f else "tras"), erro_f, erro_t, centros


def _gravar(vao: Vao, centros: dict[int, np.ndarray]) -> tuple[int, float]:
    """Escreve as caixas no XML; devolve ``(caixas gravadas, maior deslocamento em larguras)``."""
    existentes = {int(b.get("frame", "0")): b for b in vao.no.findall("box")}
    largura = min(vao.caixa_a[2] - vao.caixa_a[0], vao.caixa_b[2] - vao.caixa_b[0])
    maior = 0.0
    for q, centro in centros.items():
        reta = vao.reta(q)
        w, h = reta[2] - reta[0], reta[3] - reta[1]
        maior = max(maior, float(np.linalg.norm(centro - _centro(reta))) / max(largura, 1e-9))
        box = existentes.get(q)
        if box is None:
            # Exportacao so com quadros-chave: a caixa nasce igual a de A (atributos inclusive).
            box = ET.fromstring(ET.tostring(vao.modelo))
            box.set("frame", str(q))
            box.set("outside", "0")
            vao.no.append(box)
        box.set("xtl", f"{centro[0] - w / 2:.2f}")
        box.set("ytl", f"{centro[1] - h / 2:.2f}")
        box.set("xbr", f"{centro[0] + w / 2:.2f}")
        box.set("ybr", f"{centro[1] + h / 2:.2f}")
        box.set("keyframe", "1")
    if centros:
        ordenadas = sorted(vao.no.findall("box"), key=lambda b: int(b.get("frame", "0")))
        for box in ordenadas:
            vao.no.remove(box)
        vao.no.extend(ordenadas)
    return len(centros), maior


def refinar(raiz: ET.Element, quadros, fechamento: float = 0.25,
            rotulo: str | None = None) -> Resultado:
    """Refina ``raiz`` no lugar. ``quadros`` itera ``(indice, imagem BGR)`` em ordem.

    Cada vao so guarda os recortes enquanto esta aberto: a memoria nao cresce com o
    video. Vaos que passam do fim de ``quadros`` ficam como estavam.
    """
    res = Resultado()
    vaos = listar_vaos(raiz, rotulo)
    res.vaos = len(vaos)
    por_quadro: dict[int, list[Vao]] = defaultdict(list)
    for vao in vaos:
        for q in range(vao.a, vao.b + 1):
            por_quadro[q].append(vao)
    ultimo = max(por_quadro, default=-1)
    resolvidos = 0
    for indice, imagem in quadros:
        if indice > ultimo:
            break
        abertos = por_quadro.get(indice)
        if not abertos:
            continue
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem
        for vao in abertos:
            _guardar(vao, indice, cinza)
            if indice != vao.b:
                continue
            resolvidos += 1
            if len(vao.recortes) == vao.b - vao.a + 1:      # nenhum quadro faltou na leitura
                aceito, erro_f, erro_t, centros = resolver(vao, fechamento)
                if aceito:
                    res.aceitos[aceito] += 1
                    gravadas, maior = _gravar(vao, centros)
                    res.caixas += gravadas
                    res.corrigidos += maior >= 1 / 3
                else:
                    res.recusados.append((vao.trilha, vao.a, vao.b, erro_f, erro_t))
            else:
                res.sem_video += 1
            vao.recortes.clear()
    res.sem_video += len(vaos) - resolvidos
    return res


def ler_exportacao(caminho: Path) -> ET.ElementTree:
    """A arvore do ``annotations.xml``, de um ``.xml`` ou de dentro do ``.zip`` exportado."""
    if zipfile.is_zipfile(caminho):
        with zipfile.ZipFile(caminho) as z:
            nomes = [n for n in z.namelist() if n.endswith("annotations.xml")]
            if not nomes:
                raise ValueError(f"{caminho} nao tem annotations.xml: exporte em 'CVAT for video 1.1'.")
            return ET.ElementTree(ET.fromstring(z.read(min(nomes, key=len))))
    return ET.parse(caminho)


def gravar_exportacao(arvore: ET.ElementTree, saida: Path) -> None:
    """Grava ``.xml`` ou, com o sufixo ``.zip``, um zip com ``annotations.xml`` (como o CVAT)."""
    saida.parent.mkdir(parents=True, exist_ok=True)
    if saida.suffix.lower() == ".zip":
        conteudo = ET.tostring(arvore.getroot(), encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(saida, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("annotations.xml", conteudo)
    else:
        arvore.write(saida, encoding="utf-8", xml_declaration=True)


def quadros_do_video(video: Path):
    """``(indice, imagem)`` na ordem e com a numeracao em que o LPR-RTS le o video."""
    from lprts.capture.source import VideoSource

    fonte = VideoSource(str(video)).open()
    try:
        for cru in fonte.raw_frames():
            yield cru.index, cru.image
    finally:
        fonte.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", type=Path, help="exportacao do CVAT (.zip ou annotations.xml)")
    ap.add_argument("--video", type=Path, required=True, help="o video anotado (ou o corte dele)")
    ap.add_argument("--out", type=Path, required=True, help="exportacao refinada a gravar (.zip ou .xml)")
    ap.add_argument("--rotulo", default="placa", help="rotulo das placas no CVAT")
    ap.add_argument("--fechamento", type=float, default=0.25,
                    help="erro aceito ao chegar no quadro-chave seguinte, em larguras da caixa")
    args = ap.parse_args(argv)
    if not args.video.exists():
        ap.error(f"video nao encontrado: {args.video}")

    arvore = ler_exportacao(args.entrada)
    res = refinar(arvore.getroot(), quadros_do_video(args.video), args.fechamento, args.rotulo)
    for trilha, a, b, erro_f, erro_t in res.recusados:
        print(f"AVISO: trilha {trilha}, quadros-chave {a}-{b}: o rastreio nao fechou "
              f"(erro de {erro_f:.2f} e {erro_t:.2f} largura(s)); ficou a interpolacao linear. "
              "Confira esse vao no CVAT.")
    if res.sem_video:
        print(f"AVISO: {res.sem_video} vao(s) depois do ultimo quadro do video ficaram como estavam.")
    gravar_exportacao(arvore, args.out)
    aceitos = sum(res.aceitos.values())
    print(f"\n{args.out}: {aceitos} de {res.vaos} vaos refinados, {res.caixas} caixas reposicionadas; "
          f"em {res.corrigidos} vao(s) a caixa estava a mais de 1/3 de largura da placa.")
    print("O arquivo tem textos de placa: fica so nesta maquina (data/videos/ esta fora do git).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
