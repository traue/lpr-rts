"""Otimizacao multiobjetivo: Pareto, NSGA-II, espaco de busca e refusao."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from lprts.analytics.store import ResultStore
from lprts.core.types import PlateLayout, PlateRead
from lprts.eval.groundtruth import GroundTruth, GTAnnotation
from lprts.optimize.pareto import (
    Trial,
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_sort,
    nsga2_survivors,
    pareto_front,
)
from lprts.optimize.refusion import (
    TrackReads,
    attach_ground_truth,
    evaluate_fusion,
    grid_search,
    load_reads,
    make_objective,
    read_statistics,
)
from lprts.optimize.space import build_space, combined_space
from lprts.optimize.tuner import Tuner, available_backends

NOMES = ["exatidao", "latencia_ms"]
DIRS = ["max", "min"]


def _trials(dados):
    return [
        Trial(params={"i": i}, objectives={"exatidao": a, "latencia_ms": lat}, number=i)
        for i, (a, lat) in enumerate(dados)
    ]


class TestDominancia:
    def test_dominancia_estrita(self):
        assert dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]))

    def test_empate_nao_domina(self):
        assert not dominates(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_troca_nao_domina(self):
        assert not dominates(np.array([1.0, 5.0]), np.array([5.0, 1.0]))


class TestParetoFront:
    def test_identifica_a_fronteira(self):
        # (exatidao, latencia): (0.93,45) perde para (0.95,40) nos dois eixos
        trials = _trials([(0.95, 40), (0.90, 20), (0.85, 10), (0.93, 45), (0.80, 25)])
        frente = pareto_front(trials, NOMES, DIRS)
        assert {t.number for t in frente} == {0, 1, 2}

    def test_ponto_unico(self):
        assert len(pareto_front(_trials([(0.9, 10)]), NOMES, DIRS)) == 1

    def test_sem_trials(self):
        assert pareto_front([], NOMES, DIRS) == []

    def test_frentes_sucessivas(self):
        trials = _trials([(0.95, 40), (0.90, 20), (0.93, 45), (0.85, 50)])
        frentes = non_dominated_sort(trials, NOMES, DIRS)
        assert sum(len(f) for f in frentes) == len(trials)
        assert all(t.rank == 0 for t in frentes[0])

    def test_todos_nao_dominados(self):
        """Fronteira pura: uma unica frente."""
        trials = _trials([(0.9, 10), (0.8, 5), (0.95, 20)])
        assert len(non_dominated_sort(trials, NOMES, DIRS)) == 1


class TestCrowding:
    def test_extremos_recebem_infinito(self):
        trials = _trials([(0.9, 10), (0.8, 5), (0.95, 20), (0.85, 8)])
        crowding_distance(trials, NOMES, DIRS)
        assert sum(1 for t in trials if t.crowding == float("inf")) >= 2

    def test_poucos_pontos(self):
        trials = _trials([(0.9, 10), (0.8, 5)])
        crowding_distance(trials, NOMES, DIRS)
        assert all(t.crowding == float("inf") for t in trials)

    def test_selecao_preserva_o_tamanho(self):
        trials = _trials([(0.9 - i * 0.01, 10 + i) for i in range(20)])
        assert len(nsga2_survivors(trials, NOMES, DIRS, 8)) == 8


class TestHipervolume:
    def test_fronteira_melhor_tem_hipervolume_maior(self):
        pior = _trials([(0.70, 40), (0.60, 30)])
        melhor = _trials([(0.95, 20), (0.90, 10)])
        ref = [0.0, 60.0]
        assert hypervolume(melhor, NOMES, DIRS, ref) > hypervolume(pior, NOMES, DIRS, ref)

    def test_vazio_e_zero(self):
        assert hypervolume([], NOMES, DIRS) == 0.0

    def test_referencia_fixa_torna_estudos_comparaveis(self):
        """Regressao: a referencia derivada da propria fronteira mudava entre estudos.

        Uma fronteira estritamente melhor tem de ter hipervolume maior SEM que o
        chamador informe a referencia.
        """
        nomes, dirs = ["precisao", "revocacao"], ["max", "max"]

        def t(dados):
            return [Trial({"i": i}, {"precisao": p, "revocacao": r}, i)
                    for i, (p, r) in enumerate(dados)]

        pior = t([(0.6, 0.4), (0.5, 0.5)])
        melhor = t([(0.9, 0.8), (0.8, 0.9)])
        assert hypervolume(melhor, nomes, dirs) > hypervolume(pior, nomes, dirs)
        # Referencia (0, 0): o hipervolume de um ponto e o produto P x R.
        assert hypervolume(t([(0.5, 0.4)]), nomes, dirs) == pytest.approx(0.2)


class TestSearchSpace:
    @pytest.mark.parametrize("nome", ["fusion", "realtime", "tracker"])
    def test_espacos_conhecidos(self, nome):
        assert len(build_space(nome)) > 0

    def test_espaco_invalido(self):
        with pytest.raises(ValueError):
            build_space("inexistente")

    def test_amostragem_respeita_o_dominio(self):
        space = build_space("fusion")
        rng = random.Random(0)
        for _ in range(200):
            p = space.sample(rng)
            assert 1 <= p["fusion.min_reads"] <= 12
            assert 0.30 <= p["fusion.min_agreement"] <= 0.95
            assert p["fusion.mode"] in ("char", "string")

    def test_quantizacao_sem_ruido_de_ponto_flutuante(self):
        """Sem arredondar, 0.9+0.05 vira 0.9500000000000001 e polui a fronteira."""
        space = build_space("fusion")
        rng = random.Random(1)
        valores = {space.sample(rng)["fusion.min_agreement"] for _ in range(300)}
        assert all(len(str(v).split(".")[-1]) <= 3 for v in valores)

    def test_mutacao_permanece_no_dominio(self):
        space = build_space("fusion")
        rng = random.Random(2)
        p = space.sample(rng)
        for _ in range(100):
            p = space.mutate(p, rng, forca=0.9)
            assert 1 <= p["fusion.min_reads"] <= 12

    def test_espacos_combinados(self):
        combinado = combined_space("fusion", "tracker")
        assert len(combinado) == len(build_space("fusion")) + len(build_space("tracker"))


class TestTuner:
    def _objetivo(self, p):
        r = p["fusion.min_reads"]
        a = p["fusion.min_agreement"]
        return {
            "exatidao": 1 - math.exp(-0.45 * r) * (1.2 - a),
            "cobertura": math.exp(-0.10 * r) * (1.05 - 0.3 * a),
        }

    @pytest.mark.parametrize("backend", ["random", "nsga2"])
    def test_busca_produz_fronteira(self, backend):
        t = Tuner(build_space("fusion"), ["exatidao", "cobertura"], ["max", "max"], seed=3)
        r = t.optimize(self._objetivo, n_trials=40, backend=backend, population=10)
        assert len(r.trials) == 40
        assert len(r.front) >= 1
        assert r.hypervolume() > 0

    def test_nsga2_nao_perde_para_a_busca_aleatoria(self):
        """A busca aleatoria e a linha de base honesta que o otimizador deve bater."""
        space = build_space("fusion")
        objetivos, dirs = ["exatidao", "cobertura"], ["max", "max"]
        aleatorio = Tuner(space, objetivos, dirs, seed=5).optimize(
            self._objetivo, n_trials=80, backend="random"
        )
        nsga = Tuner(space, objetivos, dirs, seed=5).optimize(
            self._objetivo, n_trials=80, backend="nsga2", population=16
        )
        ref = [0.0, 0.0]
        assert nsga.hypervolume(ref) >= aleatorio.hypervolume(ref) * 0.95

    def test_falha_em_um_trial_nao_aborta(self):
        chamadas = {"n": 0}

        def instavel(p):
            chamadas["n"] += 1
            if chamadas["n"] % 5 == 0:
                raise RuntimeError("config invalida")
            return {"a": 1.0, "b": 2.0}

        t = Tuner(build_space("fusion"), ["a", "b"], ["max", "max"], seed=1)
        r = t.optimize(instavel, n_trials=20, backend="random")
        assert r.n_failed > 0 and len(r.trials) > 0

    def test_objetivos_e_direcoes_incompativeis(self):
        with pytest.raises(ValueError):
            Tuner(build_space("fusion"), ["a", "b"], ["max"])

    def test_relatorio_e_serializacao(self, tmp_path):
        t = Tuner(build_space("fusion"), ["exatidao", "cobertura"], ["max", "max"], seed=1)
        r = t.optimize(self._objetivo, n_trials=20, backend="random")
        assert "FRONTEIRA DE PARETO" in r.to_text()
        assert r.save(tmp_path / "estudo.json").exists()

    def test_backends_disponiveis(self):
        assert "nsga2" in available_backends() and "random" in available_backends()


class TestRefusion:
    def _tracks(self) -> list[TrackReads]:
        def leitura(texto, conf, valido=True):
            return PlateRead(
                text=texto, confidence=conf, valid_syntax=valido,
                layout=PlateLayout.MERCOSUL, sharpness=100.0,
            )

        return [
            TrackReads(1, "s", [leitura("ABC1D23", 0.9)] * 5, truth="ABC1D23"),
            TrackReads(2, "s", [leitura("XYZ9K88", 0.8)] * 4, truth="XYZ9K88"),
            TrackReads(3, "s", [leitura("QWE5R66", 0.4)] * 2, truth="QWE5R66"),
            TrackReads(4, "s", [leitura("AAA1A11", 0.5)] * 3, truth="BBB2B22"),  # erro
        ]

    def test_avaliacao_basica(self):
        m = evaluate_fusion(self._tracks(), min_reads=3, min_agreement=0.5, min_confidence=0.3)
        assert m.n_tracks == 4
        assert m.n_published == 3          # a trilha com 2 leituras nao publica
        assert m.n_correct == 2
        assert m.precision == pytest.approx(2 / 3)

    def test_min_reads_alto_reduz_publicacao(self):
        tracks = self._tracks()
        baixo = evaluate_fusion(tracks, min_reads=1, min_confidence=0.1, min_agreement=0.1)
        alto = evaluate_fusion(tracks, min_reads=5, min_confidence=0.1, min_agreement=0.1)
        assert alto.n_published < baixo.n_published

    def test_metricas_coerentes(self):
        m = evaluate_fusion(self._tracks(), min_reads=1, min_confidence=0.1, min_agreement=0.1)
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.publication_rate <= 1.0

    def test_grade_cobre_todo_o_espaco(self):
        r = grid_search(self._tracks(), range(1, 4), (0.5, 0.7), ("char", "string"))
        assert len(r) == 3 * 2 * 2

    def test_objetivo_para_o_tuner(self):
        obj = make_objective(self._tracks(), ["precisao", "revocacao"])
        vals = obj({"fusion.min_reads": 2, "fusion.min_agreement": 0.5})
        assert set(vals) == {"precisao", "revocacao"}

    def test_estatisticas(self):
        s = read_statistics(self._tracks())
        assert s["n_trilhas"] == 4 and s["n_leituras"] == 14
        assert s["com_ground_truth"] == 4

    def test_atribuicao_por_texto_parecido_e_recusada(self):
        """Regressao metodologica: a verdade nao pode depender da propria leitura."""
        tracks = [TrackReads(1, "s", [PlateRead(text="ABC1D28", confidence=0.9)])]
        with pytest.raises(TypeError):
            attach_ground_truth(tracks, {"ABC1D23", "XYZ9K88"})

    def test_ground_truth_pela_posicao(self):
        """A trilha recebe a placa do veiculo que ela cobre - mesmo lendo outra coisa.

        A leitura "XYZ9K88" e identica a placa de OUTRO veiculo; o pareamento
        antigo (texto mais parecido) a daria como correta.
        """
        gt = GroundTruth(annotations=[
            GTAnnotation(10, (100, 100, 200, 140), "ABC1D23", track_id=1),
            GTAnnotation(10, (500, 100, 600, 140), "XYZ9K88", track_id=2),
        ])
        leitura = PlateRead(text="XYZ9K88", confidence=0.9, frame_index=10,
                            bbox=(102, 101, 198, 139))
        tracks = attach_ground_truth([TrackReads(7, "s", [leitura])], gt)
        assert tracks[0].truth == "ABC1D23"
        assert tracks[0].vehicle == "id:1"

    def test_trilha_sem_placa_anotada_fica_sem_verdade(self):
        gt = GroundTruth(annotations=[GTAnnotation(10, (100, 100, 200, 140), "ABC1D23", track_id=1)])
        leitura = PlateRead(text="ABC1D23", confidence=0.9, frame_index=10,
                            bbox=(800, 500, 900, 540))      # um letreiro, longe da placa
        tracks = attach_ground_truth([TrackReads(3, "s", [leitura])], gt)
        assert tracks[0].truth == "" and tracks[0].vehicle == ""

    def test_revocacao_conta_veiculos_nunca_lidos(self):
        """O denominador e o numero de veiculos anotados, nao o de trilhas."""
        tracks = self._tracks()
        for i, t in enumerate(tracks):
            t.vehicle = f"id:{i}"
        com = evaluate_fusion(tracks, min_reads=1, min_confidence=0.1, min_agreement=0.1,
                              n_vehicles=10)
        assert com.n_vehicles == 10
        assert com.recall == pytest.approx(com.n_vehicles_correct / 10)

    def test_simulacao_publica_na_primeira_fusao_estavel(self):
        """A refusao replica a decisao online: publica na 1a vez que fica estavel.

        Tres leituras certas tornam a trilha estavel; leituras erradas depois nao
        mudam o que ja foi publicado (fundir tudo no fim daria outra resposta).
        """
        from lprts.fusion.voting import PlateVoter
        from lprts.optimize.refusion import simulate_online

        boas = [PlateRead(text="ABC1D23", confidence=0.9, valid_syntax=True,
                          layout=PlateLayout.MERCOSUL, frame_index=i) for i in range(3)]
        ruins = [PlateRead(text="XYZ9K88", confidence=0.95, valid_syntax=True,
                           layout=PlateLayout.MERCOSUL, frame_index=10 + i) for i in range(6)]
        texto, n = simulate_online(boas + ruins, PlateVoter(min_reads=3, min_agreement=0.6))
        assert texto == "ABC1D23" and n == 3

    def test_particao_por_veiculo(self):
        from lprts.optimize.refusion import split_by_vehicle

        tracks = []
        for v in range(10):
            for k in range(3):                  # 3 trilhas (fragmentos) por veiculo
                tracks.append(TrackReads(v * 10 + k, "s", [], truth="X", vehicle=f"id:{v}"))
        p = split_by_vehicle(tracks, {f"id:{v}" for v in range(12)}, holdout=0.25, seed=1)
        assert p.tune_vehicles.isdisjoint(p.holdout_vehicles)
        assert len(p.tune_vehicles) + len(p.holdout_vehicles) == 12   # inclui os nunca lidos
        # Nenhum veiculo aparece dos dois lados.
        assert {t.vehicle for t in p.tune_tracks}.isdisjoint({t.vehicle for t in p.holdout_tracks})

    def test_carrega_do_banco(self, tmp_path):
        db = tmp_path / "r.db"
        with ResultStore(db) as store:
            store.start_session("v.mp4")
            for i in range(6):
                store.add_read(
                    1, PlateRead(text="ABC1D23", confidence=0.9, valid_syntax=True,
                                 engine="template", sharpness=50.0),
                    i, i / 30.0,
                )
        tracks = load_reads(db)
        assert len(tracks) == 1 and len(tracks[0].reads) == 6

    def test_banco_sem_leituras(self, tmp_path):
        db = tmp_path / "vazio.db"
        with ResultStore(db) as store:
            store.start_session("v.mp4")
        assert load_reads(db) == []


# ---------------------------------------------------------------------------
# A varredura exaustiva precisa gravar o estudo, nao so imprimir a tabela
# ---------------------------------------------------------------------------

def test_tune_grid_grava_o_estudo_com_a_fronteira(tmp_path):
    """``lprts tune --grid -o arquivo.json`` tem de escrever o arquivo.

    A opcao ``--out`` existia e era ignorada no modo grade: a tabela ia para o
    terminal e o estudo se perdia, o que inviabiliza citar a varredura num
    artigo.
    """
    import json
    import sqlite3

    from typer.testing import CliRunner

    from lprts.analytics.store import _load_schema
    from lprts.cli import app

    banco = tmp_path / "leituras.db"
    con = sqlite3.connect(banco)
    con.executescript(_load_schema())
    con.execute(
        "INSERT INTO sessions (session_id, started_at, source_uri) "
        "VALUES ('s1', '2026-01-01T00:00:00Z', 'teste.mp4')"
    )
    for i in range(6):
        con.execute(
            "INSERT INTO reads (session_id, track_id, frame_index, raw_text, text, "
            "confidence, layout, valid_syntax, engine) "
            "VALUES ('s1', 1, ?, 'ABC1D23', 'ABC1D23', 0.9, 'mercosul', 1, 'teste')",
            (i,),
        )
    con.commit()
    con.close()

    # A particao de ajuste/validacao exige ground truth com caixas.
    gt = tmp_path / "gt.csv"
    gt.write_text(
        "frame_index,x1,y1,x2,y2,plate_text,object_class,track_id,kind,difficult\n"
        + "".join(
            f"{i},10,10,110,60,,carro,1,object,0\n"
            f"{i},30,25,90,50,ABC1D23,carro,1,plate,0\n"
            for i in range(6)
        ),
        encoding="utf-8",
    )

    saida = tmp_path / "estudo.json"
    resultado = CliRunner().invoke(
        app, ["tune", "--db", str(banco), "--gt", str(gt), "--grid", "-o", str(saida)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert saida.exists(), "o modo grade ignorou --out"

    estudo = json.loads(saida.read_text(encoding="utf-8"))
    assert estudo["mode"] == "grid"
    assert estudo["n_configs"] == len(estudo["trials"])
    assert estudo["n_configs"] > 0
    assert estudo["ms_per_config"] > 0
    assert estudo["pareto_front"], "a fronteira de Pareto nao foi calculada"
    assert all("params" in t and "objectives" in t for t in estudo["pareto_front"])
    assert "holdout" in estudo, "o estudo tem de registrar a particao de validacao"
