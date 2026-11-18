"""Relatorio descritivo para o artigo (`lprts report`)."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
SINTETICO = RAIZ / "data" / "videos" / "sintetico.mp4"

pytestmark = pytest.mark.skipif(not SINTETICO.exists(), reason="video sintetico ausente")


@pytest.fixture(scope="module")
def banco(tmp_path_factory):
    """Uma sessao curta do video sintetico, em modo so-placa (rapido)."""
    from typer.testing import CliRunner

    from lprts.cli import app

    pasta = tmp_path_factory.mktemp("relatorio")
    perfil = pasta / "saida.yaml"
    perfil.write_text(f"analytics:\n  crops_dir: {pasta / 'crops'}\n", encoding="utf-8")
    db = pasta / "r.db"
    r = CliRunner().invoke(app, [
        "eval", "--source", str(SINTETICO), "--gt", str(SINTETICO.with_name("sintetico_gt.csv")),
        "--config", str(perfil), "--no-vehicle-detector", "--plate-detector", "classic",
        "--ocr", "template", "--max-frames", "150", "--db", str(db),
        "--out", str(pasta / "r.json"),
    ])
    assert r.exit_code == 0, r.output
    return db


def test_gera_tabelas_e_figuras_sem_textos_de_placa(banco, tmp_path):
    from typer.testing import CliRunner

    from lprts.cli import app

    out = tmp_path / "rel"
    r = CliRunner().invoke(app, ["report", "--db", str(banco), "--out", str(out),
                                 "--min-dwell", "0"])
    assert r.exit_code == 0, r.output
    for nome in ("videos", "objetos", "funil", "ocr_largura", "desempenho"):
        with (out / f"{nome}.csv").open(encoding="utf-8") as fh:
            assert list(csv.DictReader(fh)), nome
    texto = (out / "relatorio.md").read_text(encoding="utf-8")
    assert "Funil das placas" in texto and "sintetico.mp4" in texto

    # Placa e dado pessoal: nenhum texto lido pode sair nas tabelas.
    with sqlite3.connect(banco) as con:
        placas = {r[0] for r in con.execute("SELECT text FROM reads WHERE text<>''")}
    assert placas
    saida = texto + "".join(p.read_text(encoding="utf-8") for p in out.glob("*.csv"))
    assert not [p for p in placas if p in saida]

    pytest.importorskip("matplotlib")
    assert (out / "fig_latencia.pdf").exists() and (out / "fig_composicao.png").exists()


def test_permanencia_minima_separa_fragmentos(banco):
    from lprts.analytics.study_report import collect

    todos = collect([banco], min_dwell_s=0.0).objetos
    longos = collect([banco], min_dwell_s=60.0).objetos
    assert sum(o["confirmadas"] for o in todos) > 0
    assert sum(o["confirmadas"] for o in longos) == 0
    assert [o["trilhas"] for o in todos] == [o["trilhas"] for o in longos]


def test_banco_inexistente_e_erro(tmp_path):
    from lprts.analytics.study_report import build_report

    with pytest.raises(FileNotFoundError):
        build_report([tmp_path / "nao.db"], tmp_path / "out")
