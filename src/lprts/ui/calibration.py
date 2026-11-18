"""Calibracao interativa: linhas virtuais e homografia, em janela OpenCV.

Usa OpenCV em vez de Qt de proposito: a calibracao e feita uma vez por camera e
precisa funcionar mesmo sem a interface grafica completa instalada.

Controles::

    modo LINHA (tecla L)  clique em 2 pontos para criar uma linha de contagem
    modo SOLO  (tecla H)  clique em 4 pontos no chao, em sentido horario,
                          comecando pelo canto proximo-esquerdo
    U  desfaz o ultimo ponto      R  reinicia
    S  salva e sai                Q/ESC sai sem salvar

Para a homografia sao pedidas as coordenadas reais (em metros) dos 4 pontos.
Uma referencia pratica no Brasil: a marca de faixa tracejada tem 2 m de traco e
4 m de intervalo; a largura de faixa de rodovia e 3,5 m.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from ..utils.logging import get_logger

log = get_logger(__name__)

MODE_LINE = "linha"
MODE_GROUND = "solo"


class _State:
    def __init__(self) -> None:
        self.mode = MODE_LINE
        self.pending: list[tuple[int, int]] = []
        self.lines: list[dict] = []
        self.ground: list[tuple[int, int]] = []


def _on_mouse(event: int, x: int, y: int, flags: int, state: object | None) -> None:
    if not isinstance(state, _State) or event != cv2.EVENT_LBUTTONDOWN:
        return
    if state.mode == MODE_LINE:
        state.pending.append((x, y))
        if len(state.pending) == 2:
            state.lines.append(
                {
                    "name": f"linha_{len(state.lines) + 1}",
                    "p1": list(state.pending[0]),
                    "p2": list(state.pending[1]),
                }
            )
            state.pending.clear()
    elif len(state.ground) < 4:
        state.ground.append((x, y))


def _render(base: np.ndarray, state: _State) -> np.ndarray:
    canvas = base.copy()
    for line in state.lines:
        p1, p2 = tuple(line["p1"]), tuple(line["p2"])
        cv2.line(canvas, p1, p2, (200, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line["name"], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 120, 255), 1)

    for i, pt in enumerate(state.ground):
        cv2.circle(canvas, pt, 6, (80, 255, 120), -1)
        cv2.putText(canvas, str(i + 1), (pt[0] + 8, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2)
    if len(state.ground) == 4:
        cv2.polylines(canvas, [np.array(state.ground, np.int32)], True, (80, 255, 120), 2)

    for pt in state.pending:
        cv2.circle(canvas, pt, 5, (60, 180, 255), -1)

    banner = [
        f"modo: {state.mode.upper()}   (L=linha  H=solo)",
        f"linhas: {len(state.lines)}   pontos de solo: {len(state.ground)}/4",
        "U=desfazer  R=reiniciar  S=salvar e sair  Q=sair",
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 74), (30, 28, 26), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    for i, text in enumerate(banner):
        cv2.putText(canvas, text, (12, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def _ask_world_points() -> list[list[float]] | None:
    """Pergunta no terminal as coordenadas reais dos 4 pontos, em metros."""
    print("\nInforme as coordenadas REAIS dos 4 pontos no plano do solo, em metros.")
    print("Origem sugerida: ponto 1 = (0, 0). Eixo X = lateral, Eixo Y = ao longo da via.")
    print("Exemplo para uma faixa de 3,5 m x 20 m:  0 0 / 3.5 0 / 3.5 20 / 0 20\n")
    world: list[list[float]] = []
    for i in range(4):
        while True:
            raw = input(f"  ponto {i + 1} (x y em metros): ").strip().replace(",", " ")
            parts = raw.split()
            if len(parts) == 2:
                try:
                    world.append([float(parts[0]), float(parts[1])])
                    break
                except ValueError:
                    pass
            print("  entrada invalida - digite dois numeros, ex.: 3.5 0")
    return world


def run_calibration(source: str, out_path: Path, frame_index: int = 0) -> int:
    """Abre a janela de calibracao e grava o YAML resultante."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f"ERRO: nao foi possivel abrir a fonte: {source}")
        return 1
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, base = cap.read()
    cap.release()
    if not ok or base is None:
        print("ERRO: nao foi possivel ler o quadro de referencia.")
        return 1

    state = _State()
    window = "LPR-RTS - Calibracao"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse, state)
    print(__doc__)

    while True:
        cv2.imshow(window, _render(base, state))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("Calibracao cancelada.")
            return 0
        if key == ord("l"):
            state.mode = MODE_LINE
            state.pending.clear()
        elif key == ord("h"):
            state.mode = MODE_GROUND
            state.pending.clear()
        elif key == ord("u"):
            if state.mode == MODE_LINE:
                if state.pending:
                    state.pending.pop()
                elif state.lines:
                    state.lines.pop()
            elif state.ground:
                state.ground.pop()
        elif key == ord("r"):
            state = _State()
            cv2.setMouseCallback(window, _on_mouse, state)
        elif key == ord("s"):
            break

    cv2.destroyAllWindows()

    # Pontos e linhas estao em pixels da imagem ORIGINAL; a resolucao fica
    # registrada para que o pipeline recuse aplica-los a outra fonte.
    calibration: dict = {"image_size": [int(base.shape[1]), int(base.shape[0])]}
    if len(state.ground) == 4:
        world = _ask_world_points()
        if world:
            calibration.update({
                "image_points": [list(p) for p in state.ground],
                "world_points": world,
            })
    else:
        print(
            "\nAviso: menos de 4 pontos de solo marcados - a velocidade em km/h "
            "ficara indisponivel (o sistema reporta None em vez de um valor sem sentido)."
        )

    payload = {"analytics": {"lines": state.lines, "calibration": calibration}}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"\nCalibracao salva em: {out_path}")
    print(f"  {len(state.lines)} linha(s) virtual(is), homografia: "
          f"{'sim' if 'image_points' in calibration else 'nao'}, "
          f"resolucao {calibration['image_size'][0]}x{calibration['image_size'][1]}")
    print("\nCombine com um perfil (o ultimo --config sobrepoe os anteriores):")
    print(f"  lprts run --source {source} --config configs/avaliacao.yaml --config {out_path}")
    return 0
