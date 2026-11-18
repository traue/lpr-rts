"""Busca de hiperparametros multiobjetivo.

Tres backends, escolhidos por ``backend``:

* ``optuna`` - se instalado, usa o amostrador NSGA-II do Optuna, com todo o
  ferramental de estudo persistente e visualizacao.
* ``nsga2`` - implementacao propria (padrao). Sem dependencia externa, guiada por
  nao-dominancia e distancia de aglomeracao.
* ``random`` - busca aleatoria. Nao e um placebo: e a **linha de base honesta**
  contra a qual qualquer otimizador precisa se provar, e em espacos pequenos
  costuma ser competitiva. Reportar o ganho sobre ela e boa pratica.

Todos devolvem a mesma estrutura, entao trocar de backend nao muda a analise.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .pareto import (
    Direction,
    Trial,
    default_reference,
    hypervolume,
    nsga2_survivors,
    pareto_front,
)
from .space import SearchSpace

log = get_logger(__name__)

ObjectiveFn = Callable[[dict[str, Any]], dict[str, float]]


@dataclass(slots=True)
class StudyResult:
    """Resultado de uma busca."""

    trials: list[Trial] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    directions: list[Direction] = field(default_factory=list)
    backend: str = ""
    elapsed_s: float = 0.0
    n_failed: int = 0

    @property
    def front(self) -> list[Trial]:
        return pareto_front(self.trials, self.objectives, self.directions)

    def best(self, objective: str | None = None) -> Trial | None:
        """Melhor trial num unico objetivo (o extremo da fronteira)."""
        alvo = objective or (self.objectives[0] if self.objectives else "")
        if not self.trials or not alvo:
            return None
        idx = self.objectives.index(alvo) if alvo in self.objectives else 0
        maior_melhor = self.directions[idx] == "max"
        return (max if maior_melhor else min)(
            self.trials, key=lambda t: t.objectives.get(alvo, 0.0)
        )

    def hypervolume(self, reference: list[float] | None = None) -> float:
        return hypervolume(self.trials, self.objectives, self.directions, reference)

    def to_dict(self) -> dict:
        referencia = default_reference(self.objectives, self.directions)
        return {
            "backend": self.backend,
            "elapsed_s": self.elapsed_s,
            "objectives": self.objectives,
            "directions": self.directions,
            "n_trials": len(self.trials),
            "n_failed": self.n_failed,
            "hypervolume": self.hypervolume(),
            # O ponto de referencia e parte do resultado: hipervolumes so se
            # comparam com o mesmo ponto.
            "hypervolume_reference": referencia if referencia is not None
            else "derivado da fronteira (nao comparavel entre estudos)",
            "front": [t.as_dict() for t in self.front],
            "trials": [t.as_dict() for t in self.trials],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path

    def to_text(self, max_rows: int = 15) -> str:
        linhas = [
            "=" * 78,
            "FRONTEIRA DE PARETO",
            "=" * 78,
            f"Backend ......: {self.backend}",
            f"Tentativas ...: {len(self.trials)} ({self.n_failed} com falha)",
            f"Tempo ........: {self.elapsed_s:.1f} s",
            f"Hipervolume ..: {self.hypervolume():.4f}",
            "",
            "Nenhuma configuracao abaixo e melhor que outra em TODOS os objetivos:",
            "escolha o ponto conforme a aplicacao.",
            "",
        ]
        # Configuracoes distintas podem produzir objetivos identicos (por
        # exemplo, min_agreement irrelevante quando min_reads=1). Mostrar as
        # dezenas de equivalentes nao acrescenta informacao: agrupamos por
        # valor de objetivo e indicamos quantas configuracoes empatam.
        agrupadas: dict[tuple, list[Trial]] = {}
        for t in self.front:
            chave = tuple(round(t.objectives.get(n, 0.0), 6) for n in self.objectives)
            agrupadas.setdefault(chave, []).append(t)

        if agrupadas:
            cab = "  " + "  ".join(f"{n:>11s}" for n in self.objectives) + "   n  parametros"
            linhas.append(cab)
            linhas.append("  " + "-" * 76)
            for chave, grupo in list(agrupadas.items())[:max_rows]:
                vals = "  ".join(f"{v:>11.4f}" for v in chave)
                # Representante: a configuracao mais simples do grupo empatado.
                rep = min(grupo, key=lambda t: sum(
                    float(v) for v in t.params.values() if isinstance(v, (int, float))
                ))
                params = ", ".join(
                    f"{k.split('.')[-1][:12]}={v}" for k, v in sorted(rep.params.items())
                )
                if len(params) > 44:
                    params = params[:41] + "..."
                linhas.append(f"  {vals} {len(grupo):>3d}  {params}")
            if len(agrupadas) > max_rows:
                linhas.append(f"  ... e mais {len(agrupadas) - max_rows} pontos na fronteira")
        linhas.append("=" * 78)
        linhas.append("A coluna 'n' indica quantas configuracoes distintas empatam nesse ponto.")
        return "\n".join(linhas)


class Tuner:
    """Otimizador multiobjetivo de hiperparametros."""

    def __init__(
        self,
        space: SearchSpace,
        objectives: list[str],
        directions: list[Direction],
        seed: int = 42,
    ) -> None:
        if len(objectives) != len(directions):
            raise ValueError("objectives e directions devem ter o mesmo tamanho")
        self.space = space
        self.objectives = objectives
        self.directions = directions
        self.seed = seed
        self._rng = random.Random(seed)

    # -- backends ----------------------------------------------------------

    def optimize(
        self,
        objective_fn: ObjectiveFn,
        n_trials: int = 50,
        backend: str = "auto",
        population: int = 16,
        on_trial: Callable[[Trial], None] | None = None,
    ) -> StudyResult:
        """Executa a busca e devolve todos os trials avaliados."""
        escolhido = backend
        if escolhido == "auto":
            escolhido = "optuna" if _optuna_disponivel() else "nsga2"

        iniciado = time.perf_counter()
        if escolhido == "optuna":
            trials, falhas = self._run_optuna(objective_fn, n_trials, on_trial)
        elif escolhido == "random":
            trials, falhas = self._run_random(objective_fn, n_trials, on_trial)
        elif escolhido == "nsga2":
            trials, falhas = self._run_nsga2(objective_fn, n_trials, population, on_trial)
        else:
            raise ValueError(f"Backend desconhecido: {backend!r}")

        return StudyResult(
            trials=trials,
            objectives=self.objectives,
            directions=self.directions,
            backend=escolhido,
            elapsed_s=time.perf_counter() - iniciado,
            n_failed=falhas,
        )

    def _avaliar(
        self, params: dict[str, Any], numero: int, objective_fn: ObjectiveFn
    ) -> Trial | None:
        try:
            valores = objective_fn(params)
        except Exception as exc:  # noqa: BLE001 - uma config ruim nao aborta a busca
            log.warning("Tentativa %d falhou (%s): %s", numero, params, exc)
            return None
        return Trial(params=dict(params), objectives=dict(valores), number=numero)

    def _run_random(
        self, objective_fn: ObjectiveFn, n_trials: int, on_trial
    ) -> tuple[list[Trial], int]:
        trials: list[Trial] = []
        falhas = 0
        for i in range(n_trials):
            t = self._avaliar(self.space.sample(self._rng), i, objective_fn)
            if t is None:
                falhas += 1
                continue
            trials.append(t)
            if on_trial:
                on_trial(t)
        return trials, falhas

    def _run_nsga2(
        self, objective_fn: ObjectiveFn, n_trials: int, population: int, on_trial
    ) -> tuple[list[Trial], int]:
        """NSGA-II: populacao inicial aleatoria, depois selecao + cruzamento + mutacao."""
        trials: list[Trial] = []
        falhas = 0
        numero = 0

        populacao: list[Trial] = []
        inicial = min(population, n_trials)
        for _ in range(inicial):
            t = self._avaliar(self.space.sample(self._rng), numero, objective_fn)
            numero += 1
            if t is None:
                falhas += 1
                continue
            populacao.append(t)
            trials.append(t)
            if on_trial:
                on_trial(t)

        while numero < n_trials and populacao:
            # Selecao por torneio binario, criterio (rank, aglomeracao) do NSGA-II.
            pais = nsga2_survivors(
                list(populacao), self.objectives, self.directions, max(2, len(populacao) // 2)
            )
            a = self._rng.choice(pais)
            b = self._rng.choice(pais)
            filho = self.space.crossover(a.params, b.params, self._rng)
            filho = self.space.mutate(filho, self._rng, forca=0.15)

            t = self._avaliar(filho, numero, objective_fn)
            numero += 1
            if t is None:
                falhas += 1
                continue
            trials.append(t)
            populacao.append(t)
            if on_trial:
                on_trial(t)
            if len(populacao) > population:
                populacao = nsga2_survivors(
                    populacao, self.objectives, self.directions, population
                )
        return trials, falhas

    def _run_optuna(
        self, objective_fn: ObjectiveFn, n_trials: int, on_trial
    ) -> tuple[list[Trial], int]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            directions=["maximize" if d == "max" else "minimize" for d in self.directions],
            sampler=optuna.samplers.NSGAIISampler(seed=self.seed),
        )
        coletados: list[Trial] = []
        falhas = 0

        def _objetivo(trial: optuna.Trial):
            params: dict[str, Any] = {}
            for p in self.space.params:
                if p.kind == "categorical":
                    params[p.name] = trial.suggest_categorical(p.name, list(p.choices))
                elif p.kind == "int":
                    params[p.name] = trial.suggest_int(
                        p.name, int(p.low), int(p.high), step=int(p.step) or 1
                    )
                else:
                    params[p.name] = trial.suggest_float(
                        p.name, p.low, p.high, step=p.step or None
                    )
            valores = objective_fn(params)
            registro = Trial(
                params=dict(params), objectives=dict(valores), number=trial.number
            )
            coletados.append(registro)
            if on_trial:
                on_trial(registro)
            return tuple(valores.get(n, 0.0) for n in self.objectives)

        study.optimize(_objetivo, n_trials=n_trials, catch=(Exception,))
        falhas = n_trials - len(coletados)
        return coletados, max(0, falhas)


def _optuna_disponivel() -> bool:
    import importlib.util

    return importlib.util.find_spec("optuna") is not None


def available_backends() -> list[str]:
    backends = ["nsga2", "random"]
    if _optuna_disponivel():
        backends.insert(0, "optuna")
    return backends
