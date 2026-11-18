"""Rastreador multi-objeto no estilo ByteTrack.

A ideia central do ByteTrack (Zhang et al., ECCV 2022) e associar em **duas
passadas**: primeiro as deteccoes de alta confianca, depois as de baixa
confianca contra as trilhas que sobraram. Deteccoes fracas costumam ser objetos
reais parcialmente ocluidos - descarta-las por limiar cria trocas de ID. Manter
a identidade e o que permite a votacao multi-frame da placa (``fusion.voting``).

Semantica dos limiares - atencao ao comparar com o artigo
---------------------------------------------------------
Aqui os limiares de associacao sao expressos como **IoU minimo**
(``match_iou``, ``second_match_iou``). No codigo de referencia do ByteTrack o
parametro ``match_thresh=0.8`` e um **custo maximo** (1 - IoU), ou seja, IoU
minimo de 0,2; a segunda passada usa custo 0,5 (IoU >= 0,5). Os padroes deste
modulo reproduzem esses valores efetivos. Uma versao anterior usava IoU >= 0,8
na primeira passada: um carro de 200 px andando 25 px por quadro (~60 km/h em
escala tipica) nunca chegava a ser confirmado.

Associacao
----------
Em cada passada, o emparelhamento maximiza a soma de ``IoU - limiar`` sobre os
pares acima do limiar - o mesmo criterio do ``lap.lapjv(..., cost_limit=...)``
do codigo de referencia. Maximizar o NUMERO de pares (o criterio do MOTA, em
``eval.session``) trocaria um par forte por dois fracos: duas trilhas vizinhas
trocariam de veiculo. Uma trilha so e associavel enquanto
``time_since_update <= max_age``; o codigo de referencia ainda aceita um quadro
a mais, porque expira as trilhas depois de associar.

Tempo e quadros
---------------
``update`` recebe o indice do quadro da fonte. Quando quadros sao pulados
(descarte em tempo real, ``detect_every_n``), o filtro de Kalman preve o numero
real de quadros decorridos, e ``max_age`` e contado em quadros da fonte - a
tolerancia a oclusao fica a mesma em segundos, qualquer que seja a carga.

Ciclo de vida
-------------
Trilhas que expiram nao desaparecem em silencio: ``pop_removed`` as devolve
ao pipeline, que grava o estado final de cada uma. Sem isso, so os objetos ainda
em cena no ultimo quadro chegariam ao banco.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.taxonomy import ObjectClass, from_label
from ..core.types import Detection, TrackState
from ..utils.geometry import iou_matrix
from .assignment import max_weight_assignment
from .kalman import KalmanBoxFilter


class _Track:
    """Trilha interna: filtro de Kalman + estado publico exposto ao pipeline."""

    def __init__(self, track_id: int, det: Detection, frame_index: int, now: float) -> None:
        self.kf = KalmanBoxFilter(det.bbox)
        self.state = TrackState(
            track_id=track_id,
            bbox=det.bbox,
            score=det.score,
            object_class=from_label(det.label),
            hits=1,
            age=1,
            first_seen=now,
            last_seen=now,
            first_frame=frame_index,
            last_frame=frame_index,
        )
        self.state.add_point(frame_index, now)
        det.track_id = track_id

    def predict(self, steps: int) -> tuple[float, float, float, float]:
        predicted = self.kf.predict(steps)
        self.state.age += steps
        self.state.time_since_update += steps
        self.state.bbox = predicted
        return predicted

    def update(self, det: Detection, frame_index: int, now: float) -> None:
        self.state.bbox = self.kf.update(det.bbox)
        self.state.score = det.score
        self.state.hits += 1
        self.state.time_since_update = 0
        self.state.last_seen = now
        self.state.last_frame = frame_index
        cls = from_label(det.label)
        if cls is not ObjectClass.DESCONHECIDO:
            self.state.object_class = cls
        self.state.add_point(frame_index, now)
        det.track_id = self.state.track_id


class ByteTracker:
    """Rastreador por IoU com associacao em duas etapas e filtro de Kalman."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_iou: float = 0.2,
        second_match_iou: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.second_match_iou = second_match_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_Track] = []
        self._removed: list[TrackState] = []
        self._frame_index: int | None = None
        self._last_detection_frame: int | None = None
        # Contador por instancia: o mesmo video processado duas vezes no mesmo
        # processo (lote sequencial, interface reiniciada, notebook) precisa
        # produzir os mesmos identificadores.
        self._next_id = 1

    # -- API ---------------------------------------------------------------

    def _advance(self, frame_index: int | None) -> int:
        """Avanca o relogio do rastreador e devolve quantos quadros se passaram."""
        if frame_index is None:
            frame_index = (self._frame_index if self._frame_index is not None else -1) + 1
        if self._frame_index is not None and frame_index < self._frame_index:
            self.reset()  # uma descontinuidade nao pode transferir identidades antigas
        steps = 1 if self._frame_index is None else max(1, frame_index - self._frame_index)
        self._frame_index = frame_index
        for track in self._tracks:
            track.predict(steps)
        return steps

    def update(
        self, detections: list[Detection], frame_index: int | None = None, now: float | None = None
    ) -> list[TrackState]:
        """Processa um quadro com deteccoes e devolve as trilhas confirmadas associadas."""
        steps = self._advance(frame_index)
        self._prune()  # expirar ANTES de associar: update zeraria a idade da trilha
        frame = self._frame_index
        assert frame is not None
        now = now if now is not None else time.monotonic()
        self._last_detection_frame = frame

        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        # --- 1a passada: deteccoes fortes contra todas as trilhas ---
        unmatched_tracks = list(range(len(self._tracks)))
        matches, unmatched_tracks, unmatched_high = self._associate(
            high, unmatched_tracks, self.match_iou
        )
        for t_idx, d_idx in matches:
            self._tracks[t_idx].update(high[d_idx], frame, now)

        # --- 2a passada: deteccoes fracas contra as trilhas ainda livres ---
        # So trilhas vistas recentemente participam: casar um objeto fraco com
        # uma trilha antiga costuma ser um falso positivo. "Recente" e medido em
        # quadros da fonte, proporcional ao passo atual.
        recent = [
            i for i in unmatched_tracks
            if self._tracks[i].state.time_since_update <= 2 * steps
        ]
        matches2, _, _ = self._associate(low, recent, self.second_match_iou)
        for t_idx, d_idx in matches2:
            self._tracks[t_idx].update(low[d_idx], frame, now)

        # --- novas trilhas apenas para deteccoes fortes nao associadas ---
        for d_idx in unmatched_high:
            self._tracks.append(_Track(self._new_id(), high[d_idx], frame, now))

        self._prune()
        return self._report(lambda t: t.state.time_since_update == 0)

    def predict_only(self, frame_index: int, now: float | None = None) -> list[TrackState]:
        """Quadro sem deteccao (``detect_every_n``): so a predicao do Kalman.

        Nenhuma trilha recebe medida - reenviar as deteccoes do quadro anterior,
        como uma versao antiga fazia, puxava o filtro de volta a posicoes velhas
        e zerava a velocidade estimada. Devolve as trilhas confirmadas que
        foram associadas na ultima deteccao, com a caixa prevista.
        """
        self._advance(frame_index)
        ultima = self._last_detection_frame
        self._prune()
        if ultima is None:
            return []
        decorridos = (self._frame_index or 0) - ultima
        return self._report(lambda t: t.state.time_since_update <= decorridos, mark=False)

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _prune(self) -> None:
        vivas: list[_Track] = []
        for t in self._tracks:
            if t.state.time_since_update <= self.max_age:
                vivas.append(t)
            else:
                t.state.end_reason = t.state.end_reason or "lost"
                self._removed.append(t.state)
        self._tracks = vivas

    def _report(self, criterio, mark: bool = True) -> list[TrackState]:
        saida: list[TrackState] = []
        for t in self._tracks:
            if t.state.hits >= self.min_hits:
                t.state.confirmed = True
            if t.state.confirmed and criterio(t):
                if mark:
                    t.state.reported = True
                saida.append(t.state)
        return saida

    def _associate(
        self, dets: list[Detection], track_indices: list[int], min_iou: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associacao otima por IoU. Retorna (pares, trilhas livres, deteccoes livres).

        Peso maximo com limiar, como o ``lapjv`` do ByteTrack de referencia: um
        par forte nao e trocado por dois fracos (ver ``max_weight_assignment``).
        """
        if not dets or not track_indices:
            return [], list(track_indices), list(range(len(dets)))

        track_boxes = np.array([self._tracks[i].state.bbox for i in track_indices], dtype=np.float32)
        det_boxes = np.array([d.bbox for d in dets], dtype=np.float32)
        ious = iou_matrix(track_boxes, det_boxes)
        rows, cols = max_weight_assignment(ious, min_iou)
        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for r, c in zip(rows, cols, strict=True):
            matches.append((track_indices[r], int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

        free_tracks = [track_indices[i] for i in range(len(track_indices)) if i not in matched_tracks]
        free_dets = [i for i in range(len(dets)) if i not in matched_dets]
        return matches, free_tracks, free_dets

    # -- consultas ---------------------------------------------------------

    @property
    def tracks(self) -> list[TrackState]:
        """Todas as trilhas vivas, inclusive as temporariamente perdidas."""
        return [t.state for t in self._tracks]

    def get(self, track_id: int) -> TrackState | None:
        for t in self._tracks:
            if t.state.track_id == track_id:
                return t.state
        return None

    def pop_removed(self) -> list[TrackState]:
        """Trilhas encerradas desde a ultima chamada (o chamador grava o estado final)."""
        removidas, self._removed = self._removed, []
        return removidas

    def close_all(self, reason: str = "end_of_stream") -> list[TrackState]:
        """Encerra todas as trilhas vivas (fim do video) e as devolve."""
        for t in self._tracks:
            t.state.end_reason = t.state.end_reason or reason
            self._removed.append(t.state)
        self._tracks = []
        return self.pop_removed()

    def reset(self) -> None:
        """Descarta as trilhas atuais (troca de perfil) sem perder a numeracao.

        As trilhas descartadas vao para ``pop_removed``: o pipeline ainda grava o
        que elas acumularam. A numeracao continua - IDs unicos na sessao.
        """
        for t in self._tracks:
            t.state.end_reason = "reset"
            self._removed.append(t.state)
        self._tracks.clear()
        self._frame_index = None
        self._last_detection_frame = None
