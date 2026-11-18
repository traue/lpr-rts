"""Montagem e apresentacao do relatorio de avaliacao.

O relatorio e organizado pela unidade que interessa em LPR - o **veiculo** -, e
cada proporcao vem com intervalo de confianca de 95%. Alem do JSON, as tabelas
rotuladas (uma linha por veiculo, por leitura e por evento) sao gravadas em CSV:
sao a entrada direta de qualquer modelagem estatistica posterior.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import DetectionMetrics, OCRMetrics


def _fmt(v: Any, casas: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{casas}f}"
    return str(v)


def _ic(d: dict, chave: str) -> str:
    """Formata 'estimativa [lo, hi]' a partir de d[chave], d[chave_ci]."""
    est = d.get(chave)
    ci = d.get(f"{chave}_ci") or [None, None]
    if est is None:
        return "-"
    return f"{_fmt(est)}  [{_fmt(ci[0])}, {_fmt(ci[1])}]"


@dataclass(slots=True)
class EvaluationReport:
    """Resultado completo de uma avaliacao contra ground truth."""

    video: str = ""
    session_id: str = ""
    n_frames: int = 0
    detection: DetectionMetrics | None = None          # placas, por quadro
    object_detection: DetectionMetrics | None = None   # objetos (se anotados)
    map_scores: dict = field(default_factory=dict)
    ocr_frame_level: OCRMetrics | None = None          # cada tentativa, pareada por posicao
    frame_level: dict = field(default_factory=dict)    # taxas e ICs do nivel de leitura
    vehicle_level: dict = field(default_factory=dict)  # a metrica principal
    fusion_comparison: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    components: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    # Tabelas rotuladas (gravadas em CSV, nao no texto)
    vehicles: list[dict] = field(default_factory=list)
    reads: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # Compatibilidade com o relatorio antigo: o "nivel de trilha" foi substituido
    # pelo nivel de veiculo, que conta veiculos perdidos e duplicatas.
    @property
    def ocr_track_level(self) -> None:
        return None

    def as_dict(self, include_tables: bool = False) -> dict:
        dados = {
            "video": self.video,
            "session_id": self.session_id,
            "n_frames": self.n_frames,
            "created_at": self.created_at,
            "vehicle_level": self.vehicle_level,
            "fusion_comparison": self.fusion_comparison,
            "frame_level": self.frame_level,
            "ocr_frame_level": self.ocr_frame_level.as_dict() if self.ocr_frame_level else None,
            "detection": self.detection.as_dict() if self.detection else None,
            "object_detection": (
                self.object_detection.as_dict() if self.object_detection else None
            ),
            "map": self.map_scores,
            "tracking": self.tracking,
            "performance": self.performance,
            "coverage": self.coverage,
            "warnings": self.warnings,
            "components": self.components,
            "config": self.config,
            "environment": self.environment,
        }
        if include_tables:
            dados["vehicles"] = self.vehicles
            dados["events"] = self.events
        return dados

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(include_tables=True), indent=2, ensure_ascii=False,
                       default=str),
            encoding="utf-8",
        )
        return path

    def save_tables(self, json_path: str | Path) -> list[Path]:
        """Grava ``<base>_veiculos.csv``, ``<base>_leituras.csv`` e ``<base>_eventos.csv``."""
        json_path = Path(json_path)
        base = json_path.with_suffix("")
        escritos: list[Path] = []
        for sufixo, linhas in (
            ("veiculos", self.vehicles), ("leituras", self.reads), ("eventos", self.events)
        ):
            caminho = base.parent / f"{base.name}_{sufixo}.csv"
            caminho.parent.mkdir(parents=True, exist_ok=True)
            with caminho.open("w", newline="", encoding="utf-8") as fh:
                if linhas:
                    colunas: list[str] = []
                    for linha in linhas:
                        for k in linha:
                            if k not in colunas:
                                colunas.append(k)
                    w = csv.DictWriter(fh, fieldnames=colunas)
                    w.writeheader()
                    w.writerows(linhas)
            escritos.append(caminho)
        return escritos

    def to_text(self) -> str:
        """Relatorio em texto puro, pronto para colar num caderno de laboratorio."""
        lines: list[str] = []
        add = lines.append
        add("=" * 76)
        add("RELATORIO DE AVALIACAO - LPR-RTS")
        add("=" * 76)
        add(f"Video......: {self.video}")
        add(f"Sessao.....: {self.session_id}")
        add(f"Quadros....: {self.n_frames}")
        add(f"Gerado em..: {self.created_at}")
        add("Intervalos: IC de 95% por bootstrap (veiculos ou eventos como unidade).")

        v = self.vehicle_level
        if v:
            add("")
            add("-" * 76)
            add("POR VEICULO (ponta a ponta: deteccao + rastreio + OCR + fusao)")
            add("-" * 76)
            add(f"  Veiculos anotados ..........: {v.get('n_vehicles')}"
                f"  (excluidos por difficult: {v.get('n_excluded_difficult', 0)})")
            add(f"  Revocacao (placa correta) ..: {_ic(v, 'recall')}")
            if v.get("recall_readable") is not None:
                add(f"  Revocacao, so legiveis .....: {_ic(v, 'recall_readable')}"
                    f"  (n={v.get('n_readable')}, placa >= {_fmt(v.get('readable_min_width_px'), 0)} px)")
            add(f"  Precisao dos eventos .......: {_ic(v, 'precision')}  (n={v.get('n_events')})")
            if v.get("precision_emitted") is not None:
                # Com revisoes ou decisoes relaxadas na saida: as linhas acima usam a
                # decisao VIGENTE; estas, o que um consumidor recebeu primeiro.
                add(f"  Revocacao, 1a decisao ......: {_ic(v, 'first_decision_recall')}")
                add(f"  Precisao, tudo o que saiu ..: {_ic(v, 'precision_emitted')}"
                    f"  (revisoes: {v.get('n_revisions', 0)},"
                    f" na saida relaxada: {v.get('n_exit_relaxed', 0)})")
            add(f"  F1 .........................: {_fmt(v.get('f1'))}")
            add(f"  Eventos sem veiculo anotado : {v.get('n_events_unmatched')}")
            add(f"  Duplicatas (mesmo veiculo) .: {v.get('n_duplicate_events')}")
            add(f"  Trilhas por veiculo detect. : {_fmt(v.get('tracks_per_detected_vehicle'), 2)}")
            add(f"  Latencia de decisao (s) ....: mediana {_fmt(v.get('decision_delay_median_s'), 2)}"
                f"  p95 {_fmt(v.get('decision_delay_p95_s'), 2)}"
                "  (video, do 1o quadro anotado)")
            # Offline, o relogio do host mede a velocidade do processamento, nao uma
            # latencia: a linha so aparece em tempo real (o JSON traz sempre).
            if v.get("decision_elapsed_n") and self.config.get("source", {}).get("realtime"):
                add(f"  Ate decidir, no relogio (s) : mediana "
                    f"{_fmt(v.get('decision_elapsed_median_s'), 2)}"
                    f"  p95 {_fmt(v.get('decision_elapsed_p95_s'), 2)}"
                    "  (da 1a imagem da trilha)")

        f = self.fusion_comparison
        if f:
            add("")
            add("-" * 76)
            add("EFEITO DA FUSAO (mesmos veiculos: os que receberam ao menos uma leitura)")
            add("-" * 76)
            add(f"  Veiculos comparados ........: {f.get('n')}")
            add(f"  Uma leitura ao acaso .......: {_fmt(f.get('single_read'))}")
            add(f"  Leitura mais confiante .....: {_fmt(f.get('best_read'))}")
            add(f"  Fusao de todas as leituras .: {_fmt(f.get('fused'))}")
            add(f"  Publicado (decisao online) .: {_fmt(f.get('published'))}")
            mc = f.get("fused_vs_best") or {}
            if mc:
                add(f"  Fusao - melhor leitura .....: {_fmt(mc.get('diff_b_minus_a'))}"
                    f"  [{_fmt(mc.get('diff_ci_low'))}, {_fmt(mc.get('diff_ci_high'))}]"
                    f"  McNemar p={_fmt(mc.get('mcnemar_p'))}"
                    f"  (discordantes: {mc.get('only_a_correct')}/{mc.get('only_b_correct')})")

        r = self.frame_level
        m = self.ocr_frame_level
        if m is not None:
            add("")
            add("-" * 76)
            add("OCR - POR TENTATIVA (pareada por posicao com a placa anotada)")
            add("-" * 76)
            add(f"  Tentativas pareadas ........: {m.n}"
                f"  (sem placa anotada: {r.get('n_unmatched_reads', 0)})")
            add(f"  Exatidao (placa inteira) ...: {_ic(r, 'accuracy')}")
            add(f"  Exatidao, so com texto .....: {_fmt(r.get('accuracy_nonempty'))}")
            add(f"  Rejeicao (sem texto) .......: {_fmt(r.get('rejection_rate'))}")
            add(f"  CER agregado ...............: {m.cer:.4f}")
            top = m.top_confusions(8)
            if top:
                add("  Principais confusoes (referencia -> predito):")
                for ref, hyp, n in top:
                    add(f"      {ref} -> {hyp}  ({n}x)")
            if m.per_position_errors:
                worst = sorted(m.per_position_errors.items(), key=lambda kv: kv[1], reverse=True)[:5]
                add("  Erros por posicao: " + ", ".join(f"pos{p}={n}" for p, n in worst))
            if r.get("censored"):
                add("  ATENCAO: ocr.max_reads_published > 0 - a amostra de leituras depende do")
                add("  acerto do motor (trilhas publicadas param de ser lidas).")

        for titulo, d in (
            ("DETECCAO DE PLACAS (por quadro)", self.detection),
            ("DETECCAO DE OBJETOS (por quadro, sem classe)", self.object_detection),
        ):
            if d is None:
                continue
            add("")
            add("-" * 76)
            add(f"{titulo}  IoU >= {d.iou_threshold:.2f}")
            add("-" * 76)
            add(f"  TP / FP / FN ...............: {d.tp} / {d.fp} / {d.fn}")
            add(f"  Precisao / Revocacao / F1 ..: {d.precision:.4f} / {d.recall:.4f} / {d.f1:.4f}")
            add(f"  AP .........................: {d.ap:.4f}")
            add(f"  IoU medio (TPs) ............: {d.mean_iou:.4f}")
            if d is self.detection and self.map_scores:
                add(f"  mAP@[.5:.95] ...............: {self.map_scores.get('mAP@[.5:.95]', 0):.4f}")

        t = self.tracking
        if t:
            add("")
            add("-" * 76)
            add(f"RASTREAMENTO (caixas de placa, IoU >= {_fmt(t.get('iou_threshold'), 2)})")
            add("-" * 76)
            add(f"  MOTA .......................: {_fmt(t.get('mota'))}")
            add(f"  IDF1 .......................: {_fmt(t.get('idf1'))}")
            add(f"  Trocas de identidade .......: {t.get('id_switches')}")
            add(f"  Fragmentacao (trilhas/veic.): {_fmt(t.get('tracks_per_vehicle'), 2)}")

        p = self.performance
        if p:
            add("")
            add("-" * 76)
            add("DESEMPENHO")
            add("-" * 76)
            add(f"  Vazao ......................: {_fmt(p.get('throughput_fps'), 1)} quadros/s")
            add(f"  Latencia ponta a ponta (ms) : p50 {_fmt(p.get('latency_p50_ms'), 1)}"
                f"  p95 {_fmt(p.get('latency_p95_ms'), 1)}  p99 {_fmt(p.get('latency_p99_ms'), 1)}"
                f"  max {_fmt(p.get('latency_max_ms'), 1)}")
            add(f"  Tempo de servico (ms) ......: p50 {_fmt(p.get('service_p50_ms'), 1)}"
                f"  p95 {_fmt(p.get('service_p95_ms'), 1)}")
            add(f"  Descartes ..................: {p.get('dropped_frames')}"
                f"  (taxa {_fmt(p.get('drop_rate'))})")
            if p.get("deadline_ms"):
                add(f"  Prazo perdido ..............: {_fmt(p.get('deadline_miss_rate'))}"
                    f"  (prazo {_fmt(p.get('deadline_ms'), 1)} ms)")

        if self.coverage:
            c = self.coverage
            add("")
            add(f"  Quadros anotados processados: {c.get('gt_frames_processed')}"
                f" de {c.get('gt_frames')}")

        if self.warnings:
            add("")
            add("-" * 76)
            add("AVISOS")
            add("-" * 76)
            for w in self.warnings:
                add(f"  ! {w}")
        add("=" * 76)
        return "\n".join(lines)

    def print(self) -> None:
        print(self.to_text())
