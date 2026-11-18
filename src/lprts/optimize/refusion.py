"""Reotimizacao da fusao a partir das leituras ja gravadas.

Esta e a otimizacao de melhor custo-beneficio do projeto, e ela existe por causa
de uma decisao de esquema: a tabela ``reads`` guarda **cada leitura de OCR,
quadro a quadro**, com confianca (inclusive por caractere), nitidez, validade
sintatica e a caixa da placa.

Consequencia pratica: varrer os parametros de fusao nao exige reprocessar video.
Uma tentativa custa milissegundos - carregar linhas do SQLite e revotar.

Tres cuidados metodologicos, todos implementados aqui:

1. **Verdade por posicao.** Cada trilha recebe a placa do veiculo anotado com que
   ela coincide no espaco e no tempo (IoU das caixas no mesmo quadro). Uma versao
   anterior atribuia a placa anotada mais *parecida com a propria leitura* - num
   corpus com milhares de placas, leituras com 3-4 erros caiam na placa errada
   em 11% a 89% dos casos, e o otimizador era recompensado por isso.
2. **Decisao online replicada.** As leituras sao reapresentadas em ordem de
   quadro e a politica de publicacao do pipeline (``fusion.publication``: a
   mesma classe) decide - primeira fusao estavel, revisoes, decisao na saida -,
   nao uma unica fusao sobre todas as leituras no fim.
3. **Revocacao sobre veiculos anotados.** O denominador e o numero de veiculos do
   ground truth (veiculos perdidos contam), nao o numero de trilhas.

E um limite que precisa ser declarado: a refusao so ve as leituras que o
sistema fez. Se a sessao foi gravada com ``ocr.max_reads_published > 0``, trilhas
publicadas pararam de ser lidas e configuracoes que exigiriam mais leituras sao
avaliadas sobre dados censurados (``load_reads`` avisa).

Uso tipico::

    lprts tune --db runs/avaliacao.db --gt anotacoes.csv --holdout 0.3
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.types import PlateLayout, PlateRead, TrackState
from ..fusion.publication import PublicationPolicy
from ..fusion.voting import PlateVoter
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TrackReads:
    """Leituras brutas de uma trilha, carregadas do banco."""

    track_id: int
    session_id: str
    reads: list[PlateRead] = field(default_factory=list)
    truth: str = ""              # placa do veiculo associado ('' = nenhum veiculo anotado)
    vehicle: str = ""            # identidade do veiculo anotado associado
    excluded_difficult: bool = False


@dataclass(slots=True)
class LoadInfo:
    """Metadados da carga: o que pode limitar a validade da refusao."""

    censored_sessions: list[str] = field(default_factory=list)
    truncated_sessions: list[str] = field(default_factory=list)


def load_reads(
    db_path: str | Path,
    session_id: str | None = None,
    min_reads: int = 1,
    info: LoadInfo | None = None,
) -> list[TrackReads]:
    """Carrega as leituras com texto, agrupadas por trilha e em ordem de quadro."""
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    if info is not None:
        sql_s = "SELECT session_id, config_json FROM sessions"
        params_s: tuple = ()
        if session_id:
            sql_s += " WHERE session_id = ?"
            params_s = (session_id,)
        for row in con.execute(sql_s, params_s):
            cfg = json.loads(row["config_json"] or "{}")
            if cfg.get("ocr", {}).get("max_reads_published", 0):
                info.censored_sessions.append(row["session_id"])
            if cfg.get("fusion", {}).get("max_reads_per_track", 0):
                info.truncated_sessions.append(row["session_id"])

    sql = "SELECT * FROM reads WHERE text <> ''"
    params: tuple = ()
    if session_id:
        sql += " AND session_id = ?"
        params = (session_id,)
    sql += " ORDER BY session_id, track_id, frame_index, id"

    por_trilha: dict[tuple[str, int], TrackReads] = {}
    for row in con.execute(sql, params):
        chave = (row["session_id"], row["track_id"])
        alvo = por_trilha.setdefault(
            chave, TrackReads(track_id=row["track_id"], session_id=row["session_id"])
        )
        try:
            confs = json.loads(row["char_confidences"] or "[]")
        except (json.JSONDecodeError, IndexError, KeyError):
            confs = []
        caixa = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
        alvo.reads.append(
            PlateRead(
                text=row["text"] or "",
                confidence=float(row["confidence"] or 0.0),
                char_confidences=[float(c) for c in confs],
                layout=PlateLayout(row["layout"] or "unknown"),
                valid_syntax=bool(row["valid_syntax"]),
                engine=row["engine"] or "",
                sharpness=float(row["sharpness"] or 0.0),
                frame_index=int(row["frame_index"] or 0),
                raw_text=row["raw_text"] or "",
                bbox=caixa if (caixa[2] or 0) > (caixa[0] or 0) else None,
            )
        )
    con.close()
    return [t for t in por_trilha.values() if len(t.reads) >= min_reads]


def attach_ground_truth(
    tracks: list[TrackReads], gt: Any, iou_threshold: float = 0.3
) -> list[TrackReads]:
    """Associa cada trilha ao veiculo anotado pela POSICAO das suas leituras.

    ``gt`` e um ``GroundTruth``. Cada leitura vota no veiculo anotado de maior
    IoU no mesmo quadro; a trilha fica com o mais votado. Trilhas que nao
    coincidem com nenhuma placa anotada ficam sem verdade (``truth = ''``) e
    qualquer publicacao delas conta como erro.
    """
    from ..utils.geometry import iou

    if isinstance(gt, (set, list, tuple)):
        raise TypeError(
            "attach_ground_truth agora exige o GroundTruth completo (com caixas): "
            "a atribuicao pelo texto mais parecido foi abandonada por ser enviesada."
        )
    veiculos = gt.vehicles()
    por_quadro: dict[int, list[tuple[str, tuple]]] = defaultdict(list)
    for chave, v in veiculos.items():
        for quadro, caixa in v.boxes.items():
            por_quadro[quadro].append((chave, caixa))

    for t in tracks:
        votos: dict[str, int] = defaultdict(int)
        for r in t.reads:
            if r.bbox is None:
                continue
            melhor, chave = 0.0, None
            for k, caixa in por_quadro.get(r.frame_index, []):
                v = iou(r.bbox, caixa)
                if v > melhor:
                    melhor, chave = v, k
            if chave is not None and melhor >= iou_threshold:
                votos[chave] += 1
        if votos:
            chave = max(votos, key=lambda k: votos[k])
            t.vehicle = chave
            t.truth = veiculos[chave].plate_text
            t.excluded_difficult = veiculos[chave].all_difficult
        else:
            t.vehicle = ""
            t.truth = ""
            t.excluded_difficult = False
    return tracks


@dataclass(slots=True)
class Split:
    """Particao ajuste/validacao: trilhas e veiculos anotados de cada lado."""

    tune_tracks: list[TrackReads]
    tune_vehicles: set[str]
    holdout_tracks: list[TrackReads]
    holdout_vehicles: set[str]


def split_by_vehicle(
    tracks: list[TrackReads], vehicles: set[str], holdout: float, seed: int = 0
) -> Split:
    """Separa ajuste e validacao POR VEICULO ANOTADO (nunca por leitura).

    Quadros e leituras do mesmo veiculo sao quase identicos; separar por leitura
    faria o mesmo veiculo cair dos dois lados e a validacao seria ficcao. O
    sorteio e sobre TODOS os veiculos anotados - inclusive os que o sistema
    nunca leu, que contam no denominador da revocacao de cada lado. Trilhas sem
    veiculo anotado (falsos positivos) sao sorteadas a parte.
    """
    rng = random.Random(seed)
    chaves = sorted(vehicles)
    rng.shuffle(chaves)
    n_val = int(round(len(chaves) * holdout))
    val_v = set(chaves[:n_val])
    tune_v = set(chaves[n_val:])
    tune_t: list[TrackReads] = []
    val_t: list[TrackReads] = []
    for t in sorted(tracks, key=lambda x: (x.session_id, x.track_id)):
        if t.excluded_difficult:
            continue
        if t.vehicle:
            (val_t if t.vehicle in val_v else tune_t).append(t)
        else:
            (val_t if rng.random() < holdout else tune_t).append(t)
    return Split(tune_t, tune_v, val_t, val_v)


@dataclass(slots=True)
class RefusionMetrics:
    """Resultado de uma configuracao de fusao aplicada ao conjunto todo."""

    n_tracks: int = 0
    n_vehicles: int = 0          # veiculos anotados (denominador da revocacao)
    n_published: int = 0         # trilhas que publicaram
    n_correct: int = 0           # publicacoes corretas
    n_wrong: int = 0
    n_vehicles_correct: int = 0  # veiculos com ao menos uma publicacao correta
    n_revised: int = 0           # trilhas cuja decisao foi revisada
    n_first_wrong_fixed: int = 0 # ...das quais a 1a decisao estava errada e a vigente certa
    cer: float = 0.0

    @property
    def precision(self) -> float:
        """Das placas publicadas, quantas estavam certas."""
        return self.n_correct / self.n_published if self.n_published else 0.0

    @property
    def recall(self) -> float:
        """Dos veiculos anotados, quantos tiveram a placa correta publicada."""
        if self.n_vehicles:
            return self.n_vehicles_correct / self.n_vehicles
        return self.n_correct / self.n_tracks if self.n_tracks else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def publication_rate(self) -> float:
        return self.n_published / self.n_tracks if self.n_tracks else 0.0

    def as_objectives(self) -> dict[str, float]:
        return {
            "precisao": self.precision,
            "revocacao": self.recall,
            "f1": self.f1,
            "cer": self.cer,
            "taxa_publicacao": self.publication_rate,
        }


@dataclass(slots=True)
class Replay:
    """Decisoes que o pipeline tomaria para uma trilha."""

    first_text: str | None = None   # primeira decisao publicada
    final_text: str | None = None   # decisao vigente ao fim da trilha
    n_reads_first: int = 0          # leituras ate a primeira decisao
    n_revisions: int = 0
    decision: str = ""              # tipo da decisao vigente


def replay_decisions(
    reads: list[PlateRead], voter: PlateVoter, policy: str = "first_stable",
    exit_min_reads: int = 0,
) -> Replay:
    """Reapresenta as leituras em ordem de quadro e aplica a politica de publicacao.

    Reproduz o pipeline: a cada leitura a fusao e refeita e a politica decide;
    quando as leituras acabam, a trilha "termina" (``on_exit``).
    """
    pol = PublicationPolicy(voter, policy, exit_min_reads)
    trilha = TrackState(track_id=0, bbox=(0.0, 0.0, 0.0, 0.0), score=0.0)
    out = Replay()
    ordenadas = sorted(reads, key=lambda x: x.frame_index)
    for i, r in enumerate(ordenadas, start=1):
        voter.add_read(trilha, r)
        d = pol.on_update(trilha, voter.vote(trilha.reads))
        if d is not None:
            pol.apply(trilha, d, r.frame_index, None)
            if d.revision == 0:
                out.n_reads_first = i
        if policy == "first_stable" and trilha.published_text is not None:
            break                       # sem revisao: nada mais muda a decisao
    if trilha.reads and (trilha.published_text is None or policy == "revise"):
        d = pol.on_exit(trilha, voter.vote(trilha.reads))
        if d is not None:
            pol.apply(trilha, d, ordenadas[-1].frame_index, None)
            if d.revision == 0:
                out.n_reads_first = len(ordenadas)
    out.first_text = trilha.first_published_text
    out.final_text = trilha.published_text
    out.n_revisions = trilha.n_revisions
    out.decision = trilha.published_decision
    return out


def simulate_online(
    reads: list[PlateRead], voter: PlateVoter, policy: str = "first_stable",
    exit_min_reads: int = 0,
) -> tuple[str | None, int]:
    """(decisao vigente, leituras ate a primeira decisao); ``None`` se nunca publicar."""
    r = replay_decisions(reads, voter, policy, exit_min_reads)
    return r.final_text, (r.n_reads_first if r.final_text is not None else len(reads))


def evaluate_fusion(
    tracks: list[TrackReads],
    mode: str = "char",
    min_reads: int = 3,
    min_agreement: float = 0.6,
    min_confidence: float = 0.45,
    require_valid_syntax: bool = True,
    max_reads_per_track: int = 0,
    max_corrections: int = 2,
    n_vehicles: int | None = None,
    publish_policy: str = "first_stable",
    exit_min_reads: int = 0,
) -> RefusionMetrics:
    """Aplica uma configuracao de fusao a todas as trilhas e mede o resultado.

    Usa o mesmo ``PlateVoter`` e a mesma ``PublicationPolicy`` do pipeline; as
    metricas sao sobre a decisao VIGENTE de cada trilha. ``n_vehicles`` e o
    total de veiculos anotados no conjunto avaliado; sem ele, a revocacao cai
    para a razao por trilha.
    """
    from ..eval.metrics import character_error_rate

    voter = PlateVoter(
        min_reads=min_reads, min_agreement=min_agreement, min_confidence=min_confidence,
        require_valid_syntax=require_valid_syntax, mode=mode,
        max_reads_per_track=max_reads_per_track, max_corrections=max_corrections,
    )
    eligible_tracks = [t for t in tracks if not t.excluded_difficult]
    m = RefusionMetrics(n_tracks=len(eligible_tracks), n_vehicles=int(n_vehicles or 0))
    erros: list[float] = []
    veiculos_ok: set[str] = set()

    for t in eligible_tracks:
        if not t.reads:
            continue
        r = replay_decisions(t.reads, voter, publish_policy, exit_min_reads)
        texto = r.final_text
        if texto is None:
            continue
        m.n_published += 1
        if r.n_revisions:
            m.n_revised += 1
            if t.truth and texto == t.truth and r.first_text != t.truth:
                m.n_first_wrong_fixed += 1
        if t.truth and texto == t.truth:
            m.n_correct += 1
            veiculos_ok.add(t.vehicle or f"_t{t.track_id}")
        else:
            m.n_wrong += 1
        if t.truth:
            erros.append(character_error_rate(t.truth, texto))
    m.n_vehicles_correct = len(veiculos_ok)
    m.cer = sum(erros) / len(erros) if erros else 0.0
    return m


def make_objective(
    tracks: list[TrackReads], objectives: list[str], n_vehicles: int | None = None
):
    """Cria a funcao-objetivo para o ``Tuner``, sobre as leituras carregadas."""

    def objetivo(params: dict[str, Any]) -> dict[str, float]:
        m = evaluate_fusion(
            tracks,
            mode=str(params.get("fusion.mode", "char")),
            min_reads=int(params.get("fusion.min_reads", 3)),
            min_agreement=float(params.get("fusion.min_agreement", 0.6)),
            min_confidence=float(params.get("fusion.min_confidence", 0.45)),
            require_valid_syntax=bool(params.get("fusion.require_valid_syntax", True)),
            n_vehicles=n_vehicles,
            publish_policy=str(params.get("fusion.publish_policy", "first_stable")),
            exit_min_reads=int(params.get("fusion.exit_min_reads", 0)),
        )
        todos = m.as_objectives()
        return {k: todos.get(k, 0.0) for k in objectives}

    return objetivo


def grid_search(
    tracks: list[TrackReads],
    min_reads_range: range = range(1, 9),
    agreement_values: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    modes: tuple[str, ...] = ("char", "string"),
    n_vehicles: int | None = None,
) -> list[tuple[dict, RefusionMetrics]]:
    """Varredura exaustiva do espaco pequeno de fusao.

    Com poucos parametros discretos, a grade completa e mais informativa que uma
    busca estocastica: cobre tudo, e reproduzivel e permite tracar superficies de
    resposta. Custa segundos, porque nada de video e reprocessado.
    """
    resultados = []
    for mode in modes:
        for min_reads in min_reads_range:
            for acordo in agreement_values:
                params = {
                    "fusion.mode": mode,
                    "fusion.min_reads": min_reads,
                    "fusion.min_agreement": acordo,
                }
                m = evaluate_fusion(
                    tracks, mode=mode, min_reads=min_reads, min_agreement=acordo,
                    n_vehicles=n_vehicles,
                )
                resultados.append((params, m))
    return resultados


def read_statistics(tracks: list[TrackReads]) -> dict[str, Any]:
    """Estatisticas descritivas das leituras - contexto antes de otimizar."""
    if not tracks:
        return {}
    contagens = [len(t.reads) for t in tracks]
    confs = [r.confidence for t in tracks for r in t.reads]
    validas = sum(1 for t in tracks for r in t.reads if r.valid_syntax)
    total = sum(contagens)
    por_motor: dict[str, int] = defaultdict(int)
    for t in tracks:
        for r in t.reads:
            por_motor[r.engine or "?"] += 1
    ordenado = sorted(contagens)
    return {
        "n_trilhas": len(tracks),
        "n_leituras": total,
        "leituras_por_trilha_min": ordenado[0],
        "leituras_por_trilha_mediana": ordenado[len(ordenado) // 2],
        "leituras_por_trilha_max": ordenado[-1],
        "confianca_media": sum(confs) / len(confs) if confs else 0.0,
        "taxa_sintaxe_valida": validas / total if total else 0.0,
        "por_motor": dict(por_motor),
        "com_ground_truth": sum(1 for t in tracks if t.truth),
        "veiculos_associados": len({t.vehicle for t in tracks if t.vehicle}),
    }
