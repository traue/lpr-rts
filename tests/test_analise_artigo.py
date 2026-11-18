"""Funções estatísticas da análise da bateria do artigo (scripts/artigo/analise.py)."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("analise_artigo", RAIZ / "scripts" / "artigo" / "analise.py")
analise = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = analise
_spec.loader.exec_module(analise)


def test_kaplan_meier_e_rmst_num_exemplo_feito_a_mao():
    # 4 veículos: decisões em 1 s e 2 s; dois censurados (sem placa correta) em 3 s e 4 s.
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 0, 0])
    tempos, sob = analise.kaplan_meier(t, e)
    assert tempos.tolist() == [1.0, 2.0]
    assert sob.tolist() == pytest.approx([0.75, 0.5])
    # área sob S(t) até τ = 3: 1·1 + 0,75·1 + 0,5·1
    assert analise.rmst(t, e, 3.0) == pytest.approx(2.25)
    # τ antes do primeiro evento: a área é o próprio τ
    assert analise.rmst(t, e, 0.5) == pytest.approx(0.5)


def test_mcnemar_conta_so_os_pares_discordantes():
    com = np.array([1, 1, 1, 0, 0, 1])
    sem = np.array([1, 0, 0, 0, 1, 1])
    b, c, p = analise.mcnemar_exato(com, sem)
    assert (b, c) == (2, 1)
    assert p == pytest.approx(1.0)
    assert analise.mcnemar_exato(np.ones(5), np.ones(5)) == (0, 0, 1.0)


def test_holm_e_monotono_e_limitado_a_um():
    ajust = analise.holm([0.01, 0.04, 0.03, 0.5])
    assert ajust == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert analise.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_wilson_contem_a_proporcao_e_fica_entre_zero_e_um():
    lo, hi = analise.wilson(45, 50)
    assert lo < 0.9 < hi
    assert lo == pytest.approx(0.786, abs=0.002) and hi == pytest.approx(0.957, abs=0.002)
    lo, hi = analise.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.35


def test_logistica_recupera_os_parametros():
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, 4000)
    y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(0.5 + 2.0 * x)))).astype(float)
    b0, b1 = analise.logistica(x, y)
    assert b0 == pytest.approx(0.5, abs=0.15)
    assert b1 == pytest.approx(2.0, abs=0.2)


def test_ols_por_blocos_acha_a_reta_e_cerca_o_coeficiente():
    rng = np.random.default_rng(2)
    n = rng.integers(0, 7, 3000).astype(float)
    y = 6.0 + 1.5 * n + rng.normal(0, 0.5, n.size)
    coef, lo, hi = analise.ols_bootstrap_blocos(n, y, np.zeros(n.size, dtype=int))
    assert coef == pytest.approx([6.0, 1.5], abs=0.05)
    assert lo[1] < 1.5 < hi[1]


def test_faixa_entre_repeticoes_e_o_pior_caso_dos_dois_lados():
    # com − sem: a menor diferença usa o menor "com" e o maior "sem", e a maior, o contrário
    assert analise.faixa_entre_repeticoes([22.0, 23.0], [39.0, 97.0]) == pytest.approx([-75.0, -16.0])
    assert analise.faixa_entre_repeticoes([10.0], [10.0]) == pytest.approx([0.0, 0.0])
    lo, hi = analise.faixa_entre_repeticoes([19.3, 20.2, 19.7], [12.4, 12.5])
    assert lo <= hi and lo == pytest.approx(19.3 - 12.5) and hi == pytest.approx(20.2 - 12.4)


def _veiculos_csv(caminho: Path, linhas: list[tuple[int, int, int, int]]) -> None:
    # só as colunas sem texto de placa: veículo, difícil, publicada, publicada correta
    caminho.write_text("vehicle,excluded_difficult,published,published_correct\n"
                       + "".join(f"{v},{d},{p},{c}\n" for v, d, p, c in linhas), encoding="utf-8")


def test_placa_errada_por_veiculo_usa_a_maioria_das_repeticoes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(analise, "RUNS", tmp_path)
    (tmp_path / "V9").mkdir()
    # veículo 1: errada em 2 de 3 repetições; 2: em 1 de 3; 3: difícil (fora da métrica);
    # 4: nunca publicada (não é placa errada); 5: sempre correta
    reps = [
        [(1, 0, 1, 0), (2, 0, 1, 0), (3, 1, 1, 0), (4, 0, 0, 0), (5, 0, 1, 1)],
        [(1, 0, 1, 0), (2, 0, 1, 1), (3, 1, 1, 0), (4, 0, 0, 0), (5, 0, 1, 1)],
        [(1, 0, 1, 1), (2, 0, 1, 1), (3, 1, 1, 0), (4, 0, 0, 0), (5, 0, 1, 1)],
    ]
    for i, linhas in enumerate(reps, start=1):
        _veiculos_csv(tmp_path / "V9" / f"base_rt_rep{i}_veiculos.csv", linhas)
    err = analise._erradas_por_veiculo("V9", "base_rt")
    assert err.name == "base_rt"
    assert err.to_dict() == {1: True, 2: False, 4: False, 5: False}
    # sem prazo: um arquivo só, sem repetições
    _veiculos_csv(tmp_path / "V9" / "base_off_veiculos.csv", [(1, 0, 1, 0), (2, 0, 1, 1)])
    assert analise._erradas_por_veiculo("V9", "base_off").to_dict() == {1: True, 2: False}


def _banco(caminho: Path, inicio: str, fim: str, mono_s: float) -> Path:
    con = sqlite3.connect(caminho)
    con.execute("CREATE TABLE sessions (session_id TEXT, started_at TEXT, finished_at TEXT, frame_count INT, "
                "source_fps REAL, realtime INT, processing_width INT)")
    con.execute("CREATE TABLE frames (session_id TEXT, capture_t_s REAL, done_t_s REAL)")
    con.execute("INSERT INTO sessions VALUES ('s', ?, ?, 30, 30.0, 1, 1280)", (inicio, fim))
    con.executemany("INSERT INTO frames VALUES ('s', ?, ?)", [(0.0, 0.01), (mono_s - 0.01, mono_s)])
    con.commit()
    con.close()
    return caminho


def test_sessao_e_suspensa_quando_o_relogio_de_parede_passa_o_monotonico(tmp_path: Path):
    # O monotônico para durante o sono; o de parede, não.
    dormiu = _banco(tmp_path / "a.db", "2026-11-16T04:00:00+00:00", "2026-11-16T04:30:00+00:00", 60.0)
    lenta = _banco(tmp_path / "b.db", "2026-11-16T04:00:00+00:00", "2026-11-16T04:01:13+00:00", 73.0)
    assert bool(analise.sessoes(dormiu).suspensa.iloc[0]) is True
    # Mais lenta que o vídeo, mas sem sono: não é suspensão.
    assert bool(analise.sessoes(lenta).suspensa.iloc[0]) is False
