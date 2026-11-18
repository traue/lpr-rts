"""Politica de publicacao: QUANDO a fusao de uma trilha vira uma decisao gravada.

A fusao (``voting``) diz qual e a melhor estimativa da placa a cada momento; a
politica decide quando essa estimativa e publicada como *evento* - o que um
consumidor em tempo real (cancela, alerta, contagem) recebe.

Politicas (``fusion.publish_policy``):

* ``first_stable`` (padrao): publica na primeira fusao estavel e nunca revisa.
  Decisao rapida; uma decisao errada cedo fica errada.
* ``revise``: como ``first_stable`` e, depois, sempre que uma fusao estavel
  posterior tiver OUTRO texto, publica uma revisao que substitui a anterior.
  Mesma latencia da primeira decisao; corrige decisoes erradas cedo (tipicamente
  a trilha que comecou na placa de outro veiculo) ao custo de mais de um evento
  por trilha.
* ``on_exit``: decide uma unica vez, quando a trilha termina, com todas as
  leituras. Mais informacao, latencia de decisao = permanencia em cena.

Em qualquer politica, ``fusion.exit_min_reads > 0`` permite que uma trilha que
termina SEM decisao publique com menos leituras do que ``min_reads`` (os demais
criterios valem). E o caso do veiculo rapido, que fica poucos quadros legivel:
sem isso ele nunca e publicado, mesmo com a fusao certa. A decisao e marcada
(``exit_relaxed``) para ser analisada a parte.

Cada decisao e gravada com ``decision`` (``stable`` | ``revision`` |
``exit_relaxed``) e ``revision`` (0 na primeira decisao da trilha, 1, 2... nas
seguintes); a decisao **vigente** de uma trilha e a de maior ``revision``.

O pipeline e a replica offline (``optimize.refusion.simulate_online``) usam esta
mesma classe: a regra existe num lugar so.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import TrackState
from .voting import FusedPlate, PlateVoter

POLITICAS = ("first_stable", "revise", "on_exit")


@dataclass(slots=True)
class Decision:
    """Uma decisao a publicar."""

    fused: FusedPlate
    kind: str            # "stable" | "revision" | "exit_relaxed"
    revision: int        # 0 = primeira decisao da trilha

    @property
    def text(self) -> str:
        return self.fused.text


class PublicationPolicy:
    """Decide e aplica as publicacoes de uma trilha."""

    def __init__(self, voter: PlateVoter, policy: str = "first_stable",
                 exit_min_reads: int = 0) -> None:
        if policy not in POLITICAS:
            from ..config import ConfigError

            raise ConfigError(
                f"fusion.publish_policy invalida: {policy!r}. Use uma de: {', '.join(POLITICAS)}."
            )
        if exit_min_reads < 0:
            from ..config import ConfigError

            raise ConfigError("fusion.exit_min_reads deve ser >= 0 (0 = mesmo criterio).")
        self.voter = voter
        self.policy = policy
        self.exit_min_reads = int(exit_min_reads)

    # -- decisao -------------------------------------------------------------

    def on_update(self, track: TrackState, fused: FusedPlate) -> Decision | None:
        """Depois de cada nova fusao, com a trilha ainda em cena."""
        if self.policy == "on_exit":
            return None
        if track.published_text is None:
            return Decision(fused, "stable", 0) if fused.stable else None
        if self._revisa(track, fused):
            return Decision(fused, "revision", track.n_revisions + 1)
        return None

    def on_exit(self, track: TrackState, fused: FusedPlate) -> Decision | None:
        """Quando a trilha termina, com a fusao final sobre todas as leituras."""
        if track.published_text is None:
            if fused.stable:
                return Decision(fused, "stable", 0)
            if self.exit_min_reads and self._relaxado(fused):
                return Decision(fused, "exit_relaxed", 0)
            return None
        if self._revisa(track, fused):
            return Decision(fused, "revision", track.n_revisions + 1)
        return None

    def _revisa(self, track: TrackState, fused: FusedPlate) -> bool:
        return (self.policy == "revise" and fused.stable and bool(fused.text)
                and fused.text != track.published_text)

    def _relaxado(self, fused: FusedPlate) -> bool:
        """O criterio de estabilidade com ``exit_min_reads`` no lugar de ``min_reads``."""
        if not fused.text or fused.n_reads < self.exit_min_reads:
            return False
        v = self.voter
        if fused.agreement < v.min_agreement or fused.confidence < v.min_confidence:
            return False
        return not (v.require_valid_syntax and not fused.valid_syntax)

    # -- aplicacao -----------------------------------------------------------

    @staticmethod
    def apply(track: TrackState, decision: Decision, frame_index: int,
              time_s: float | None) -> None:
        """Atualiza o estado de publicacao da trilha (texto vigente e historico)."""
        if decision.revision == 0:
            track.first_published_text = decision.text
            track.published_frame = frame_index
            track.published_time_s = time_s
        else:
            track.n_revisions = decision.revision
        track.published_text = decision.text
        track.published_decision = decision.kind


def build_policy(voter: PlateVoter, fusion_cfg: dict | object) -> PublicationPolicy:
    """Politica a partir da secao ``fusion`` (dataclass ou dicionario gravado na sessao)."""
    def pegar(nome: str, padrao):
        if isinstance(fusion_cfg, dict):
            return fusion_cfg.get(nome, padrao)
        return getattr(fusion_cfg, nome, padrao)

    return PublicationPolicy(voter, str(pegar("publish_policy", "first_stable")),
                             int(pegar("exit_min_reads", 0) or 0))
