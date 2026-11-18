#!/usr/bin/env python3
"""Gera um video sintetico de transito com placas brasileiras + ground truth.

Existe por tres motivos:

1. **Executar na primeira tentativa.** Permite rodar o sistema inteiro sem
   possuir video de transito nem pesos treinados.
2. **Testar de ponta a ponta.** O ground truth e exato por construcao, entao o
   comando ``lprts eval`` tem contra o que medir desde o primeiro minuto.
3. **Estudar fatores isoladamente.** Desfoque, ruido, iluminacao, velocidade e
   tempo de obturador sao parametros, o que permite curvas de degradacao
   controladas.

Velocidade e borrao de movimento (``--speed``, ``--shutter-ms``)
-----------------------------------------------------------------

``--speed S`` multiplica a velocidade de todos os veiculos. Cada veiculo tem o
seu proprio gerador aleatorio (semente + indice), entao **os mesmos veiculos -
mesma placa, cor, faixa e ordem de chegada - aparecem em qualquer velocidade**,
nas mesmas configuracoes relativas, so que S vezes mais rapido: o desenho e
pareado por veiculo. Com ``--total-vehicles N`` o video termina quando o
N-esimo veiculo sai de cena, e toda condicao tem exatamente os mesmos N.

``--shutter-ms E`` simula a exposicao da camera: cada quadro e a media de
subquadros renderizados ao longo dos E ms, **centrados no instante anotado**
(a caixa do ground truth fica no centro do rastro). O borrao, portanto, cresce
com a velocidade aparente do veiculo, como numa camera real. ``0`` = obturador
instantaneo (sem borrao). Referencias: 1/1000 s = 1 ms (camera de LPR
dedicada), 1/250 s = 4 ms, 1/60 s = 16,7 ms (exposicao automatica com pouca
luz). Tem de caber no periodo do quadro.

Por veiculo, o ``_meta.json`` registra velocidade, quadros visiveis e
legiveis, velocidade aparente da placa (px/quadro) e extensao do borrao (px) -
covariaveis para modelar a acuracia por veiculo.

O ground truth (``<saida>_gt.csv``) segue o formato de ``lprts.eval.groundtruth``:

* uma linha ``kind=plate`` por placa por quadro e uma ``kind=object`` por
  veiculo/pedestre por quadro, com ``track_id`` (a placa e o veiculo que a
  porta tem o mesmo id);
* ``difficult=1`` quando a caixa nao deve contar na metrica: placa com menos de
  ``--min-plate-width`` px, cortada pela borda da imagem, ou mais da metade
  coberta por um veiculo mais proximo. Caixas sao recortadas aos limites da
  imagem.

Os parametros da geracao (semente, fonte, fatores de degradacao, versao do
codigo) vao para ``<saida>_meta.json``.

Aviso metodologico: numeros obtidos aqui **nao substituem** avaliacao em video
real. Placas sinteticas nao tem sujeira, reflexo, deformacao de para-choque nem
compressao de camera. E a fonte importa: o OCR por gabaritos do projeto usa as
fontes Hershey SIMPLEX, DUPLEX e TRIPLEX - desenhar as placas com uma delas
favorece esse motor (vazamento). O padrao (``--font auto``) usa DejaVu Sans
Bold quando disponivel, senao Hershey COMPLEX - nenhuma das duas esta no banco
de gabaritos.

Uso::

    python scripts/make_synthetic_dataset.py --out data/videos/sintetico.mp4 \\
        --frames 600 --vehicles 6 --blur 1 --noise 4
    python scripts/make_synthetic_dataset.py --out /tmp/v3x.mp4 \\
        --total-vehicles 60 --speed 3 --shutter-ms 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

import cv2
import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

FONTES_HERSHEY = {
    "hershey-simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "hershey-duplex": cv2.FONT_HERSHEY_DUPLEX,
    "hershey-complex": cv2.FONT_HERSHEY_COMPLEX,
    "hershey-triplex": cv2.FONT_HERSHEY_TRIPLEX,
}
# Fontes que o OCR por gabaritos usa como modelo (ver lprts.ocr.template_ocr).
FONTES_DO_BASELINE = {"hershey-simplex", "hershey-duplex", "hershey-triplex"}


def random_plate(rng: random.Random, layout: str = "mix") -> str:
    """Sorteia uma placa valida no layout Mercosul ou antigo."""
    if layout == "mix":
        layout = rng.choice(["mercosul", "legacy"])
    letters = "".join(rng.choice(LETTERS) for _ in range(3))
    if layout == "mercosul":
        return f"{letters}{rng.choice(DIGITS)}{rng.choice(LETTERS)}{rng.choice(DIGITS)}{rng.choice(DIGITS)}"
    return letters + "".join(rng.choice(DIGITS) for _ in range(4))


def _dejavu_path() -> Path | None:
    try:
        import matplotlib

        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return p if p.exists() else None
    except ImportError:
        return None


def resolve_font(nome: str) -> str:
    """Escolhe a fonte efetiva (registrada nos metadados)."""
    if nome != "auto":
        return nome
    try:
        import PIL  # noqa: F401
    except ImportError:
        return "hershey-complex"
    return "dejavu" if _dejavu_path() is not None else "hershey-complex"


def _texto_dejavu(plate: np.ndarray, text: str, band: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    h, w = plate.shape[:2]
    caminho = _dejavu_path()
    alvo_w, alvo_h = int(w * 0.86), int((h - band) * 0.62)
    tamanho = 10
    fonte = ImageFont.truetype(str(caminho), tamanho)
    for tamanho in range(10, 400, 2):
        fonte = ImageFont.truetype(str(caminho), tamanho)
        x0, y0, x1, y1 = fonte.getbbox(text)
        if (x1 - x0) > alvo_w or (y1 - y0) > alvo_h:
            fonte = ImageFont.truetype(str(caminho), max(10, tamanho - 2))
            break
    img = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = fonte.getbbox(text)
    tw, th = x1 - x0, y1 - y0
    org = ((w - tw) // 2 - x0, band + (h - band - th) // 2 - y0)
    draw.text(org, text, font=fonte, fill=(20, 20, 20))
    plate[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def render_plate(text: str, font: str = "hershey-complex", width: int = 400,
                 height: int = 130) -> np.ndarray:
    """Desenha uma placa no padrao visual brasileiro (fundo branco, tarja azul)."""
    plate = np.full((height, width, 3), 250, dtype=np.uint8)
    # Tarja azul superior do padrao Mercosul.
    band = int(height * 0.22)
    cv2.rectangle(plate, (0, 0), (width, band), (120, 40, 20), -1)
    cv2.putText(
        plate, "BRASIL", (int(width * 0.36), int(band * 0.78)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.rectangle(plate, (2, 2), (width - 3, height - 3), (25, 25, 25), 3)

    if font == "dejavu":
        _texto_dejavu(plate, text, band)
        return plate
    face = FONTES_HERSHEY[font]
    scale, thickness = 2.2, 6
    (tw, th), _ = cv2.getTextSize(text, face, scale, thickness)
    if tw > width * 0.9:
        scale *= width * 0.9 / tw
        (tw, th), _ = cv2.getTextSize(text, face, scale, thickness)
    org = ((width - tw) // 2, band + (height - band + th) // 2)
    cv2.putText(plate, text, org, face, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return plate


class Pedestrian:
    """Pedestre ou ciclista atravessando a cena lateralmente."""

    def __init__(self, rng: random.Random, width: int, height: int, track_id: int) -> None:
        self.track_id = track_id
        self.is_cyclist = rng.random() < 0.35
        self.object_class = "bicicleta" if self.is_cyclist else "pessoa"
        self.width, self.height = width, height
        self.color = tuple(int(c) for c in rng.choices(range(60, 220), k=3))
        self.speed = rng.uniform(0.004, 0.011) * (1.8 if self.is_cyclist else 1.0)
        self.direction = rng.choice([-1, 1])
        self.t = rng.uniform(-0.4, 0.2)
        # Faixa de profundidade: define a escala aparente e a altura na imagem.
        self.depth = rng.uniform(0.45, 0.95)

    def step(self) -> None:
        self.t += self.speed

    @property
    def done(self) -> bool:
        return self.t > 1.3

    def geometry(self, dt: float = 0.0) -> tuple[int, int, int, int] | None:
        """Caixa em ``t + dt`` quadros (``dt`` fracionario: subquadros da exposicao)."""
        t = self.t + dt * self.speed
        if not 0.0 <= t <= 1.25:
            return None
        horizon = self.height * 0.42
        cy = int(horizon + (self.height - horizon) * self.depth)
        altura = int(self.height * 0.06 + self.height * 0.16 * self.depth)
        largura = int(altura * (0.85 if self.is_cyclist else 0.42))
        if altura < 16:
            return None
        percurso = t if self.direction > 0 else (1.0 - t)
        cx = int(percurso * (self.width + 2 * largura)) - largura
        return (cx - largura // 2, cy - altura, cx + largura // 2, cy)

    def draw(self, scene: np.ndarray, dt: float = 0.0) -> None:
        geom = self.geometry(dt)
        if geom is None:
            return
        x1, y1, x2, y2 = geom
        altura = y2 - y1
        if self.is_cyclist:
            raio = max(3, altura // 5)
            cv2.circle(scene, (x1 + raio, y2 - raio), raio, (30, 30, 30), 2)
            cv2.circle(scene, (x2 - raio, y2 - raio), raio, (30, 30, 30), 2)
            cv2.line(scene, (x1 + raio, y2 - raio), (x2 - raio, y2 - raio), (40, 40, 40), 2)
            cv2.rectangle(scene, (x1 + raio // 2, y1), ((x1 + x2) // 2 + raio, y2 - raio),
                          self.color, -1)
        else:
            cabeca = max(3, altura // 6)
            cv2.circle(scene, ((x1 + x2) // 2, y1 + cabeca), cabeca, self.color, -1)
            cv2.rectangle(scene, (x1, y1 + 2 * cabeca), (x2, y2), self.color, -1)
        cv2.rectangle(scene, (x1, y1), (x2, y2), (20, 20, 20), 1)


class Vehicle:
    """Veiculo sintetico que se aproxima da camera com escala perspectiva."""

    def __init__(self, rng: random.Random, width: int, height: int, lane: int, n_lanes: int,
                 track_id: int, font: str, speed_factor: float = 1.0) -> None:
        self.track_id = track_id
        self.plate_text = random_plate(rng)
        self.plate_img = render_plate(self.plate_text, font)
        self.color = tuple(int(c) for c in rng.choices(range(40, 210), k=3))
        self.width, self.height = width, height
        self.lane_x = (lane + 0.5) / n_lanes
        # Fracao da cena por quadro. O fator multiplica DEPOIS do sorteio: os
        # mesmos veiculos em qualquer velocidade.
        self.base_speed = rng.uniform(0.006, 0.013)
        self.speed = self.base_speed * speed_factor
        self.t = rng.uniform(-0.9, 0.0)             # posicao inicial (negativa = ainda fora)
        self.object_class = rng.choice(
            ["carro", "carro", "carro", "caminhao", "onibus", "motocicleta"]
        )

    def step(self) -> None:
        self.t += self.speed

    @property
    def done(self) -> bool:
        return self.t > 1.15

    def geometry(
        self, dt: float = 0.0
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        """``(caixa_do_veiculo, caixa_da_placa)`` em ``t + dt`` quadros, ou None fora de cena."""
        t = self.t + dt * self.speed
        if not 0.0 <= t <= 1.1:
            return None
        veh_w = int(self.width * (0.05 + 0.25 * (t**1.6)))
        veh_h = int(veh_w * 0.78)
        if veh_w < 24 or veh_h < 18:
            return None

        horizon = self.height * 0.42
        cy = int(horizon + (self.height - horizon) * (t**1.5))
        center = self.width * 0.5
        cx = int(center + (self.lane_x - 0.5) * self.width * (0.30 + 0.55 * t))

        x1, y1 = cx - veh_w // 2, cy - veh_h
        vehicle_box = (x1, y1, x1 + veh_w, y1 + veh_h)

        plate_w = max(12, int(veh_w * 0.34))
        plate_h = max(5, int(plate_w * 130 / 400))
        px = cx - plate_w // 2
        py = y1 + int(veh_h * 0.70)
        return vehicle_box, (px, py, px + plate_w, py + plate_h)


def draw_background(width: int, height: int, offset: float) -> np.ndarray:
    """Fundo de rodovia: ceu, asfalto com gradiente e marcas de faixa rolando."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * 0.42)
    img[:horizon] = (185, 170, 150)                       # ceu
    for y in range(horizon, height):
        shade = 55 + int(45 * (y - horizon) / max(1, height - horizon))
        img[y, :] = (shade, shade, shade)                 # asfalto
    cv2.line(img, (0, horizon), (width, horizon), (120, 110, 100), 2)

    vanish = (width // 2, horizon)
    for lane in (0.25, 0.5, 0.75):
        base_x = int(width * lane)
        for k in range(14):
            t0 = ((k + offset) % 14) / 14.0
            t1 = t0 + 0.035
            if t1 > 1.0:
                continue
            p0 = (
                int(vanish[0] + (base_x - vanish[0]) * (t0**1.5)),
                int(vanish[1] + (height - vanish[1]) * (t0**1.5)),
            )
            p1 = (
                int(vanish[0] + (base_x - vanish[0]) * (t1**1.5)),
                int(vanish[1] + (height - vanish[1]) * (t1**1.5)),
            )
            cv2.line(img, p0, p1, (215, 215, 210), max(1, int(1 + 4 * t0)))
    return img


def draw_vehicle(scene: np.ndarray, vehicle: Vehicle, geom) -> None:
    """Carroceria, para-brisa e placa de um veiculo na geometria dada."""
    (vx1, vy1, vx2, vy2), plate_box = geom
    cv2.rectangle(scene, (vx1, vy1), (vx2, vy2), vehicle.color, -1)
    cv2.rectangle(scene, (vx1, vy1), (vx2, vy2), (18, 18, 18), 2)
    # Para-brisa, para dar contraste interno como num veiculo real.
    cv2.rectangle(
        scene,
        (vx1 + int((vx2 - vx1) * 0.14), vy1 + int((vy2 - vy1) * 0.10)),
        (vx2 - int((vx2 - vx1) * 0.14), vy1 + int((vy2 - vy1) * 0.42)),
        (60, 55, 50), -1,
    )
    paste_plate(scene, vehicle.plate_img, plate_box)


def render_scene(width: int, height: int, frame_index: int, dt: float,
                 ordem: list[Vehicle], pedestrians: list[Pedestrian]) -> np.ndarray:
    """A cena ``dt`` quadros depois do instante ``frame_index`` (algoritmo do pintor).

    A ordem de desenho dos veiculos e a do instante anotado, para que todos os
    subquadros de uma exposicao tenham as mesmas oclusoes.
    """
    scene = draw_background(width, height, (frame_index + dt) * 0.35)
    # Pedestres e ciclistas ao fundo, antes dos veiculos.
    for pessoa in sorted(pedestrians, key=lambda p: p.depth):
        pessoa.draw(scene, dt)
    for vehicle in ordem:
        geom = vehicle.geometry(dt)
        if geom is not None:
            draw_vehicle(scene, vehicle, geom)
    return scene


def exposure_offsets(ordem: list[Vehicle], exposure: float) -> list[float]:
    """Instantes dos subquadros, centrados em 0, ao longo de ``exposure`` quadros.

    Um subquadro por pixel de deslocamento do veiculo mais rapido na cena
    (2 a 64), para que o rastro seja continuo e nao uma sequencia de copias.
    """
    if exposure <= 0:
        return [0.0]
    desloc = 0.0
    for v in ordem:
        a, b = v.geometry(-exposure / 2), v.geometry(exposure / 2)
        if a is None or b is None:
            continue
        desloc = max(desloc, *(abs(p - q) for p, q in zip(a[0] + a[1], b[0] + b[1], strict=True)))
    k = int(min(64, max(2, math.ceil(desloc) + 1)))
    return [exposure * (i / (k - 1) - 0.5) for i in range(k)]


def paste_plate(scene: np.ndarray, plate: np.ndarray, box: tuple[int, int, int, int]) -> None:
    """Cola a placa na cena, recortando o que sair dos limites."""
    x1, y1, x2, y2 = box
    h, w = scene.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    resized = cv2.resize(plate, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    scene[sy1:sy2, sx1:sx2] = resized[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1]


def _clip(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    c = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    return c if c[2] > c[0] and c[3] > c[1] else None


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _fracao_coberta(box: tuple[int, int, int, int],
                    cobertores: list[tuple[int, int, int, int]]) -> float:
    """Fracao de ``box`` coberta pela uniao dos retangulos ``cobertores``."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or not cobertores:
        return 0.0
    mascara = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for cx1, cy1, cx2, cy2 in cobertores:
        a, b = max(x1, cx1), max(y1, cy1)
        c, d = min(x2, cx2), min(y2, cy2)
        if c > a and d > b:
            mascara[b - y1 : d - y1, a - x1 : c - x1] = True
    return float(mascara.mean())


def _code_version() -> str:
    from .provenance import code_version

    return code_version()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/videos/sintetico.mp4", help="video de saida (.mp4)")
    ap.add_argument("--gt", default="", help="CSV de ground truth (padrao: <out>_gt.csv)")
    ap.add_argument("--frames", type=int, default=None,
                    help="numero de quadros (padrao 600; com --total-vehicles, limite opcional)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--vehicles", type=int, default=5, help="veiculos simultaneos")
    ap.add_argument("--total-vehicles", type=int, default=0,
                    help="encerra quando o N-esimo veiculo sai de cena (0 = pelo numero de quadros)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="multiplica a velocidade dos veiculos (mesmos veiculos em qualquer valor)")
    ap.add_argument("--shutter-ms", type=float, default=0.0,
                    help="tempo de exposicao em ms: borrao de movimento (0 = sem borrao)")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--pedestrians", type=int, default=2,
                    help="pedestres/ciclistas simultaneos (0 desliga)")
    ap.add_argument("--blur", type=int, default=0, help="desfoque gaussiano (0 = nenhum, impar)")
    ap.add_argument("--noise", type=float, default=0.0, help="desvio-padrao do ruido gaussiano")
    ap.add_argument("--brightness", type=float, default=1.0, help="fator multiplicativo de brilho")
    ap.add_argument("--font", default="auto",
                    choices=["auto", "dejavu", *sorted(FONTES_HERSHEY)],
                    help="fonte das placas (evite as do baseline: simplex/duplex/triplex)")
    ap.add_argument("--min-plate-width", type=int, default=40,
                    help="placas mais estreitas que isto sao anotadas como difficult")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.speed <= 0:
        ap.error("--speed precisa ser positivo")
    exposicao = args.shutter_ms / 1000.0 * args.fps          # em quadros
    if not 0.0 <= exposicao <= 1.0:
        ap.error(f"--shutter-ms precisa caber no periodo do quadro (0 a {1000 / args.fps:.1f} ms)")
    if args.frames is None:
        args.frames = 0 if args.total_vehicles > 0 else 600
    if args.frames <= 0 and args.total_vehicles <= 0:
        ap.error("informe --frames ou --total-vehicles")

    font = resolve_font(args.font)
    if font in FONTES_DO_BASELINE:
        print(f"AVISO: a fonte '{font}' esta no banco de gabaritos do OCR classico; "
              "os resultados desse motor neste video serao otimistas.")

    # Um gerador por objeto (semente + indice): o k-esimo veiculo e sempre o
    # mesmo, qualquer que seja a velocidade, o obturador ou o numero de pedestres.
    def rng_de(tipo: str, k: int) -> random.Random:
        return random.Random(f"{args.seed}:{tipo}:{k}")

    np_rng = np.random.default_rng(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path = Path(args.gt) if args.gt else out_path.with_name(out_path.stem + "_gt.csv")
    meta_path = out_path.with_name(out_path.stem + "_meta.json")

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        print(f"ERRO: nao foi possivel abrir o gravador de video em {out_path}")
        return 1

    vehicles: list[Vehicle] = []
    pedestrians: list[Pedestrian] = []
    gt_rows: list[tuple] = []
    W, H = args.width, args.height
    # Ids: veiculos 1, 3, 5... e pedestres 2, 4, 6... - estaveis entre condicoes.
    n_veic = n_pede = 0
    veiculos_info: dict[int, dict] = {}
    limite = args.frames if args.frames > 0 else 10**7
    frame_index = 0

    while frame_index < limite:
        while len(vehicles) < args.vehicles and (
            args.total_vehicles <= 0 or n_veic < args.total_vehicles
        ):
            v = Vehicle(rng_de("veiculo", n_veic), W, H, n_veic % args.lanes, args.lanes,
                        2 * n_veic + 1, font, args.speed)
            vehicles.append(v)
            veiculos_info[v.track_id] = {
                "track_id": v.track_id, "plate_text": v.plate_text,
                "object_class": v.object_class, "lane": n_veic % args.lanes,
                "speed_scene_per_frame": v.speed, "base_speed": v.base_speed,
                "frames_visible": 0, "frames_legible": 0, "_px": [], "_ultimo": None,
            }
            n_veic += 1
        if args.total_vehicles > 0 and n_veic >= args.total_vehicles and not vehicles:
            break                                     # o ultimo veiculo saiu de cena

        while len(pedestrians) < args.pedestrians:
            pedestrians.append(Pedestrian(rng_de("pedestre", n_pede), W, H, 2 * n_pede + 2))
            n_pede += 1

        # Geometria de todos os veiculos, na ordem de desenho (do mais distante
        # para o mais proximo): o que vem depois cobre o que veio antes.
        visible = [(v, vehicle_geom) for v in sorted(vehicles, key=lambda v: v.t)
                   if (vehicle_geom := v.geometry()) is not None]
        ordem = [v for v, _ in visible]
        geoms = [geom for _, geom in visible]
        caixas_veic = [g[0] for g in geoms]

        offsets = exposure_offsets(ordem, exposicao)
        if len(offsets) == 1:
            scene = render_scene(W, H, frame_index, 0.0, ordem, pedestrians)
        else:
            soma = np.zeros((H, W, 3), dtype=np.float32)
            for dt in offsets:
                soma += render_scene(W, H, frame_index, dt, ordem, pedestrians)
            scene = np.clip(np.rint(soma / len(offsets)), 0, 255).astype(np.uint8)

        for pessoa in sorted(pedestrians, key=lambda p: p.depth):
            geom = pessoa.geometry()
            if geom is None:
                continue
            visivel = _clip(geom, W, H)
            if visivel is None:
                continue
            cortada = _area(visivel) < 0.5 * _area(geom)
            coberta = _fracao_coberta(visivel, caixas_veic) > 0.5
            gt_rows.append((frame_index, *visivel, "", pessoa.object_class, pessoa.track_id,
                            "object", int(cortada or coberta or (geom[2] - geom[0]) < 10)))

        for i, (vehicle, ((vx1, vy1, vx2, vy2), plate_box)) in enumerate(zip(ordem, geoms,
                                                                               strict=True)):
            info = veiculos_info[vehicle.track_id]
            info["frames_visible"] += 1
            mais_proximos = caixas_veic[i + 1 :]
            caixa_v = (vx1, vy1, vx2, vy2)
            vis_v = _clip(caixa_v, W, H)
            if vis_v is not None:
                dif_v = (_area(vis_v) < 0.5 * _area(caixa_v)
                         or _fracao_coberta(vis_v, mais_proximos) > 0.5)
                gt_rows.append((frame_index, *vis_v, "", vehicle.object_class,
                                vehicle.track_id, "object", int(dif_v)))

            vis_p = _clip(plate_box, W, H)
            if vis_p is None or (plate_box[2] - plate_box[0]) < 12:
                continue
            truncada = _area(vis_p) < _area(plate_box)
            ocluida = _fracao_coberta(vis_p, mais_proximos) > 0.5
            pequena = (plate_box[2] - plate_box[0]) < args.min_plate_width
            dificil = truncada or ocluida or pequena
            gt_rows.append((frame_index, *vis_p, vehicle.plate_text, vehicle.object_class,
                            vehicle.track_id, "plate", int(dificil)))
            centro = ((plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2)
            if not dificil:
                info["frames_legible"] += 1
                if info["_ultimo"] is not None and info["_ultimo"][0] == frame_index - 1:
                    info["_px"].append(math.dist(centro, info["_ultimo"][1]))
            info["_ultimo"] = (frame_index, centro)

        if args.brightness != 1.0:
            scene = np.clip(scene.astype(np.float32) * args.brightness, 0, 255).astype(np.uint8)
        if args.blur and args.blur >= 3:
            k = args.blur if args.blur % 2 == 1 else args.blur + 1
            scene = cv2.GaussianBlur(scene, (k, k), 0)
        if args.noise > 0:
            noisy = scene.astype(np.float32) + np_rng.normal(0, args.noise, scene.shape)
            scene = np.clip(noisy, 0, 255).astype(np.uint8)

        writer.write(scene)
        frame_index += 1

        for vehicle in vehicles:
            vehicle.step()
        vehicles = [v for v in vehicles if not v.done]
        for pessoa in pedestrians:
            pessoa.step()
        pedestrians = [p for p in pedestrians if not p.done]

    writer.release()
    n_quadros = frame_index
    if args.total_vehicles > 0 and (n_veic < args.total_vehicles or vehicles):
        print(f"AVISO: --frames {args.frames} encerrou o video antes de os "
              f"{args.total_vehicles} veiculos sairem de cena.")

    por_veiculo = []
    for info in veiculos_info.values():
        px = info.pop("_px")
        info.pop("_ultimo")
        mediana = statistics.median(px) if px else None
        info["plate_px_per_frame_median"] = mediana
        # Extensao do rastro na exposicao: deslocamento por quadro x fracao exposta.
        info["motion_blur_px_median"] = mediana * exposicao if mediana is not None else None
        por_veiculo.append(info)

    with gt_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_index", "x1", "y1", "x2", "y2", "plate_text", "object_class",
                    "track_id", "kind", "difficult"])
        w.writerows(gt_rows)

    placas = [r for r in gt_rows if r[8] == "plate"]
    unicas = len({r[5] for r in placas})
    validas = len({r[7] for r in placas if not r[9]})
    legiveis = [v["frames_legible"] for v in por_veiculo if v["frames_legible"] > 0]
    meta = {
        "generator": "scripts/make_synthetic_dataset.py",
        "code_version": _code_version(),
        "args": vars(args),
        "n_frames": n_quadros,
        "exposure_fraction_of_frame": exposicao,
        "frames_legible_median": statistics.median(legiveis) if legiveis else 0,
        "font_used": font,
        "video": str(out_path),
        "ground_truth": str(gt_path),
        "n_annotations": len(gt_rows),
        "n_plate_boxes": len(placas),
        "n_plate_boxes_difficult": sum(1 for r in placas if r[9]),
        "n_vehicles_with_plate": unicas,
        "n_vehicles_with_non_difficult_plate": validas,
        "opencv": cv2.__version__,
        "vehicles": por_veiculo,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Video ......: {out_path}  ({n_quadros} quadros, {W}x{H} @ {args.fps} fps, "
          f"velocidade {args.speed}x, obturador {args.shutter_ms} ms)")
    print(f"Ground truth: {gt_path}  ({len(gt_rows)} anotacoes, {unicas} veiculos com placa, "
          f"{meta['n_plate_boxes_difficult']} caixas de placa difficult)")
    print(f"Metadados ..: {meta_path}  (fonte: {font})")
    print("\nProximo passo:")
    print(f"  lprts run --source {out_path} --no-vehicle-detector --plate-detector classic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
