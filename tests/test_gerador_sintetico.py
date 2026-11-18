"""Gerador de video sintetico: determinismo e desenho pareado de velocidade."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
GERADOR = RAIZ / "scripts" / "make_synthetic_dataset.py"


def _gerar(pasta: Path, nome: str, *args: str) -> tuple[Path, dict]:
    video = pasta / f"{nome}.mp4"
    proc = subprocess.run(
        [sys.executable, str(GERADOR), "--out", str(video), "--width", "480", "--height", "270",
         "--pedestrians", "1", *args],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    meta = json.loads(video.with_name(f"{nome}_meta.json").read_text(encoding="utf-8"))
    return video, meta


def _sha(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def test_mesma_semente_mesmos_bytes(tmp_path):
    """O video versionado em data/videos so faz sentido se for reproduzivel."""
    a, _ = _gerar(tmp_path, "a", "--frames", "60", "--vehicles", "3")
    b, _ = _gerar(tmp_path, "b", "--frames", "60", "--vehicles", "3")
    assert _sha(a) == _sha(b)
    assert _sha(tmp_path / "a_gt.csv") == _sha(tmp_path / "b_gt.csv")


def test_mesmos_veiculos_em_qualquer_velocidade(tmp_path):
    """Desenho pareado: a velocidade muda o tempo em cena, nao quais veiculos passam."""
    _, m1 = _gerar(tmp_path, "v1", "--total-vehicles", "6", "--vehicles", "3")
    _, m3 = _gerar(tmp_path, "v3", "--total-vehicles", "6", "--vehicles", "3", "--speed", "3")
    assert [v["plate_text"] for v in m1["vehicles"]] == [v["plate_text"] for v in m3["vehicles"]]
    assert [v["track_id"] for v in m1["vehicles"]] == [v["track_id"] for v in m3["vehicles"]]
    for a, b in zip(m1["vehicles"], m3["vehicles"], strict=True):
        assert b["speed_scene_per_frame"] == pytest.approx(3 * a["speed_scene_per_frame"])
        assert b["frames_visible"] < a["frames_visible"]
    assert m3["n_frames"] < m1["n_frames"] / 2


def test_obturador_borra_sem_mudar_o_ground_truth(tmp_path):
    """A caixa anotada e a do centro da exposicao: o borrao muda pixels, nao a verdade."""
    nitido, m0 = _gerar(tmp_path, "e0", "--total-vehicles", "4", "--speed", "2")
    borrado, m8 = _gerar(tmp_path, "e8", "--total-vehicles", "4", "--speed", "2",
                         "--shutter-ms", "16")
    assert _sha(tmp_path / "e0_gt.csv") == _sha(tmp_path / "e8_gt.csv")
    assert _sha(nitido) != _sha(borrado)
    assert m0["exposure_fraction_of_frame"] == 0.0
    assert m8["exposure_fraction_of_frame"] == pytest.approx(0.48)
    blur = [v["motion_blur_px_median"] for v in m8["vehicles"] if v["motion_blur_px_median"]]
    assert blur and all(b > 0 for b in blur)


@pytest.mark.parametrize("args", [
    ("--frames", "10", "--shutter-ms", "40"),       # exposicao maior que o quadro (33 ms)
    ("--frames", "10", "--speed", "0"),
])
def test_parametros_invalidos(tmp_path, args):
    proc = subprocess.run([sys.executable, str(GERADOR), "--out", str(tmp_path / "x.mp4"), *args],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 2


def test_variantes_do_experimento(tmp_path):
    """--variant nome=secao.chave:valor vira um YAML que o lprts eval aceita."""
    import importlib.util

    from lprts.config import AppConfig

    spec = importlib.util.spec_from_file_location("curva", RAIZ / "scripts" / "curva_velocidade.py")
    curva = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(curva)
    assert curva.variante("padrao=", tmp_path) == ("padrao", None)
    nome, yaml_path = curva.variante(
        "rev=fusion.publish_policy:revise,fusion.exit_min_reads:1", tmp_path)
    cfg = AppConfig.load_many([yaml_path])
    assert nome == "rev"
    assert (cfg.fusion.publish_policy, cfg.fusion.exit_min_reads) == ("revise", 1)
