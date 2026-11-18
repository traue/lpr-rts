"""Exportacao dos resultados para formatos de analise (CSV, Parquet, JSON).

CSV sai sempre - usa apenas a biblioteca padrao, abre em qualquer ferramenta e e
o formato que o R le sem atrito. Parquet exige ``pandas``+``pyarrow`` e vale a
pena quando as tabelas passam de alguns milhoes de linhas (``reads`` cresce
rapido: uma leitura por placa por quadro).
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

log = get_logger(__name__)

TABLES = (
    "sessions", "tracks", "events", "reads", "frames", "crossings", "detections", "track_points",
)
VIEWS = (
    "v_decisions", "v_unique_plates", "v_session_performance", "v_object_counts", "v_category_share",
    "v_vulnerable_exposure", "v_line_composition",
)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _existe(con: sqlite3.Connection, nome: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? AND type IN ('table', 'view')", (nome,)
    ).fetchone() is not None


def _decisoes(con: sqlite3.Connection) -> str:
    """Decisoes vigentes; num banco da 0.2 ainda nao migrado, todo evento e vigente."""
    return "v_decisions" if _existe(con, "v_decisions") else "events"


def export_csv(
    db_path: str | Path,
    out_dir: str | Path,
    session_id: str | None = None,
    tables: tuple[str, ...] = TABLES + VIEWS,
) -> list[Path]:
    """Exporta cada tabela/visao para um CSV em ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with _connect(db_path) as con:
        for table in tables:
            if not _existe(con, table):
                log.info("%s nao existe neste banco (versao anterior); pulado.", table)
                continue
            has_session = any(
                r["name"] == "session_id" for r in con.execute(f"PRAGMA table_info({table})")
            )
            if session_id and not has_session:
                # Sem como filtrar: exportar sairia com as linhas de TODAS as sessoes.
                log.warning("%s nao tem session_id; pulado na exportacao de uma sessao.", table)
                continue
            if session_id:
                rows = list(con.execute(f"SELECT * FROM {table} WHERE session_id=?", (session_id,)))
            else:
                rows = list(con.execute(f"SELECT * FROM {table}"))
            path = out_dir / f"{table}.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                if rows:
                    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows([dict(r) for r in rows])
                else:
                    # Cabecalho mesmo sem dados: mantem o esquema estavel para scripts.
                    cols = [r["name"] for r in con.execute(f"PRAGMA table_info({table})")]
                    if cols:
                        csv.writer(fh).writerow(cols)
            written.append(path)
            log.info("Exportado %s (%d linhas) -> %s", table, len(rows), path)
    return written


def export_parquet(
    db_path: str | Path, out_dir: str | Path, session_id: str | None = None
) -> list[Path]:
    """Exporta para Parquet (colunar, comprimido). Requer ``pip install '.[analysis]'``."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Exportacao Parquet requer pandas+pyarrow. Rode: pip install '.[analysis]'"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with _connect(db_path) as con:
        for table in TABLES + VIEWS:
            if not _existe(con, table):
                continue
            query = f"SELECT * FROM {table}"
            params: tuple = ()
            if session_id:
                cols = [r["name"] for r in con.execute(f"PRAGMA table_info({table})")]
                if "session_id" not in cols:
                    log.warning("%s nao tem session_id; pulado na exportacao de uma sessao.", table)
                    continue
                query += " WHERE session_id=?"
                params = (session_id,)
            frame = pd.read_sql_query(query, con, params=params)
            path = out_dir / f"{table}.parquet"
            frame.to_parquet(path, index=False, compression="snappy")
            written.append(path)
            log.info("Exportado %s (%d linhas) -> %s", table, len(frame), path)
    return written


def export_session_json(db_path: str | Path, out_path: str | Path, session_id: str) -> Path:
    """Resumo compacto de uma sessao em JSON (bom para anexar a um artigo)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as con:
        session = con.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if session is None:
            raise ValueError(f"Sessao nao encontrada: {session_id}")
        plates = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM v_unique_plates WHERE session_id=? ORDER BY n_events DESC",
                (session_id,),
            )
        ]
        perf = con.execute(
            "SELECT * FROM v_session_performance WHERE session_id=?", (session_id,)
        ).fetchone()
        crossings = [
            dict(r)
            for r in con.execute(
                """SELECT line_name, direction, COUNT(*) AS n, AVG(speed_kmh) AS avg_speed_kmh
                   FROM crossings WHERE session_id=? GROUP BY line_name, direction""",
                (session_id,),
            )
        ]

    payload: dict[str, Any] = {
        "session": dict(session),
        "performance": dict(perf) if perf else {},
        "unique_plates": plates,
        "crossings": crossings,
    }
    for key in ("config_json", "environment_json"):
        raw = payload["session"].get(key)
        if raw:
            try:
                payload["session"][key.replace("_json", "")] = json.loads(raw)
                payload["session"].pop(key)
            except json.JSONDecodeError:
                pass

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Resumo da sessao -> %s", out_path)
    return out_path


def list_sessions(db_path: str | Path) -> list[dict]:
    """Lista as sessoes gravadas no banco, da mais recente para a mais antiga."""
    with _connect(db_path) as con:
        dec = _decisoes(con)
        return [
            dict(r)
            for r in con.execute(
                f"""SELECT s.session_id, s.started_at, s.finished_at, s.source_uri,
                          s.ocr_engine, s.device, s.code_version,
                          (SELECT COUNT(*) FROM {dec} e WHERE e.session_id=s.session_id) AS n_events,
                          (SELECT COUNT(DISTINCT plate_text) FROM {dec} e
                            WHERE e.session_id=s.session_id AND plate_text<>'') AS n_plates
                   FROM sessions s ORDER BY s.started_at DESC"""
            )
        ]


def latest_session_id(db_path: str | Path) -> str:
    """Sessao mais recente do banco (``ValueError`` se nao houver nenhuma)."""
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise ValueError(f"Nenhuma sessao gravada em {db_path}")
    return str(row["session_id"])


def session_source(db_path: str | Path, session_id: str) -> str:
    """Fonte (video, camera ou stream) de uma sessao."""
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT source_uri FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"Sessao nao encontrada: {session_id}")
    return str(row["source_uri"] or "")


def _rotulo_da_fonte(uri: str) -> str:
    """Nome curto e seguro para pasta: o nome do video, ``camera0`` ou ``stream``.

    De uma URL de stream so o tipo entra no nome - ela pode trazer usuario e senha.
    """
    import re

    from ..capture.devices import classify_source

    tipo = classify_source(uri)
    if tipo == "camera":
        rotulo = f"camera{uri.strip()}"
    elif tipo == "stream":
        rotulo = "stream"
    else:
        rotulo = Path(uri).stem or "fonte"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", rotulo).strip("._") or "fonte"


def export_analyzed_video(
    db_path: str | Path,
    out_base: str | Path,
    session_id: str | None = None,
    fmt: str = "csv",
) -> tuple[str, Path, list[Path]]:
    """Exporta SO o video analisado: uma sessao, nao o banco inteiro.

    O banco acumula todas as execucoes (cada video, cada repeticao). Misturar
    sessoes numa exportacao junta videos diferentes nas mesmas tabelas - e o
    ``track_id``, que recomeca a cada sessao, passa a repetir. Aqui cada tabela
    e visao sai filtrada pela sessao, numa pasta propria
    ``<out_base>/<video>_<sessao>/``, junto com o resumo ``sessao.json``
    (configuracao, ambiente, desempenho e placas).

    Sem ``session_id``, exporta a sessao mais recente. Devolve
    ``(sessao, pasta, arquivos)``.
    """
    sid = session_id or latest_session_id(db_path)
    pasta = Path(out_base) / f"{_rotulo_da_fonte(session_source(db_path, sid))}_{sid}"
    if fmt == "csv":
        arquivos = export_csv(db_path, pasta, sid)
    elif fmt == "parquet":
        arquivos = export_parquet(db_path, pasta, sid)
    elif fmt == "json":
        arquivos = []
    else:
        raise ValueError(f"Formato desconhecido: {fmt} (use csv, parquet ou json)")
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos.append(export_session_json(db_path, pasta / "sessao.json", sid))
    return sid, pasta, arquivos
