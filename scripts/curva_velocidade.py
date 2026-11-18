#!/usr/bin/env python3
"""Curvas de acuracia x velocidade (e x tempo de obturador) em video sintetico.

Para cada combinacao de velocidade e obturador, gera um video com os MESMOS N
veiculos (desenho pareado; ver ``make_synthetic_dataset.py``), avalia com
``lprts eval`` em cada modo (offline e/ou tempo real) e consolida:

* ``<out>/resumo.csv`` - uma linha por condicao x modo x repeticao: revocacao
  por veiculo (IC 95%), precisao, acuracia da fusao e da decisao publicada,
  latencia e descartes.
* ``<out>/veiculos.csv`` - uma linha por veiculo x condicao x modo x repeticao,
  com as covariaveis da geracao (velocidade aparente da placa, extensao do
  borrao, quadros legiveis). E o formato para regressao logistica ou modelos
  mistos com o veiculo como efeito aleatorio (o mesmo ``vehicle`` em todas as
  condicoes).
* ``<out>/experimento.json`` - comando, parametros, commit e ambiente.
* ``<out>/curva.png`` - revocacao por veiculo x velocidade (se houver matplotlib).

Uso::

    python scripts/curva_velocidade.py --out runs/curva_velocidade \\
        --speeds 1 1.5 2 3 4 --shutters 0 4 16.7 --total-vehicles 60 \\
        --modes offline realtime --repeat 3 \\
        --config configs/tempo_real_rapido.yaml \\
        -- --no-vehicle-detector --plate-detector classic --ocr fastplate

Tudo depois de ``--`` vai para o ``lprts eval``. Videos ja gerados com os
mesmos parametros sao reaproveitados.

Variantes de configuracao entram como mais um eixo da grade (coluna
``variant``), sobre os mesmos videos::

    --variant padrao= \\
    --variant revisao=fusion.publish_policy:revise \\
    --variant saida1=fusion.exit_min_reads:1 Os recortes de placa nao sao salvos
(``--save-crops`` para salvar). Sintetico serve para isolar o efeito da
velocidade; o resultado do artigo continua precisando de video real.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

GERADOR = RAIZ / "scripts" / "make_synthetic_dataset.py"
LPRTS = [sys.executable, "-c", "import sys; sys.argv[0] = 'lprts'; from lprts.cli import main; main()"]

# Parametros do gerador que definem o video (os demais sao fixos aqui).
CHAVES_VIDEO = ("speed", "shutter_ms", "total_vehicles", "vehicles", "lanes", "pedestrians",
                "fps", "width", "height", "seed", "min_plate_width")


def _rotulo(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def gerar_video(pasta: Path, args, speed: float, shutter: float) -> tuple[Path, Path, dict]:
    """Gera (ou reaproveita) o video de uma condicao; devolve video, GT e metadados."""
    video = pasta / f"v{_rotulo(speed)}x_e{_rotulo(shutter)}ms.mp4"
    meta_path = video.with_name(video.stem + "_meta.json")
    gt = video.with_name(video.stem + "_gt.csv")
    desejado = {"speed": speed, "shutter_ms": shutter, "total_vehicles": args.total_vehicles,
                "vehicles": args.vehicles, "lanes": 3, "pedestrians": args.pedestrians,
                "fps": args.fps, "width": 1280, "height": 720, "seed": args.seed,
                "min_plate_width": args.min_plate_width}
    if video.exists() and gt.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if {k: meta["args"].get(k) for k in CHAVES_VIDEO} == desejado:
            return video, gt, meta
    cmd = [sys.executable, str(GERADOR), "--out", str(video)]
    for chave, valor in desejado.items():
        cmd += [f"--{chave.replace('_', '-')}", str(valor)]
    print(f"  gerando {video.name} ...", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return video, gt, json.loads(meta_path.read_text(encoding="utf-8"))


def variante(texto: str, pasta: Path) -> tuple[str, Path | None]:
    """``nome=secao.chave:valor,...`` -> (nome, YAML com as sobreposicoes)."""
    import yaml

    nome, _, corpo = texto.partition("=")
    if not nome:
        raise SystemExit(f"--variant sem nome: {texto!r}")
    if not corpo.strip():
        return nome, None
    dados: dict = {}
    for item in corpo.split(","):
        chave, _, valor = item.partition(":")
        secao, _, campo = chave.strip().partition(".")
        if not campo:
            raise SystemExit(f"--variant {nome}: use secao.chave:valor, veio {item!r}")
        dados.setdefault(secao, {})[campo] = yaml.safe_load(valor)
    caminho = pasta / f"_variante_{nome}.yaml"
    caminho.write_text(yaml.safe_dump(dados, allow_unicode=True), encoding="utf-8")
    return nome, caminho


def avaliar(video: Path, gt: Path, saida: Path, modo: str, repeat: int, configs: list[str],
            extra: list[str], reuse: bool = False) -> list[Path]:
    """Roda ``lprts eval`` e devolve os relatorios por repeticao."""
    saida.parent.mkdir(parents=True, exist_ok=True)
    n = repeat if modo == "realtime" else 1      # offline e deterministico
    relatorios = [saida] if n == 1 else [
        saida.with_name(f"{saida.stem}_rep{i + 1}{saida.suffix}") for i in range(n)]
    if reuse and all(r.exists() and r.with_name(r.stem + "_veiculos.csv").exists()
                     for r in relatorios):
        print(f"  reaproveitando {saida.name}", flush=True)
        return relatorios
    db = saida.with_suffix(".db")
    cmd = [*LPRTS, "eval", "--source", str(video), "--gt", str(gt), "--out", str(saida),
           "--db", str(db)]
    for c in configs:
        cmd += ["--config", c]
    cmd += ["--realtime" if modo == "realtime" else "--offline", "--repeat", str(n), *extra]
    print(f"  avaliando {video.name} [{modo}, {n}x] ...", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return relatorios


def _mediana(valores: list) -> float | None:
    v = sorted(x for x in valores if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def linha_resumo(cond: dict, rel: dict, meta: dict) -> dict:
    vl, fc, perf = rel["vehicle_level"], rel.get("fusion_comparison") or {}, rel["performance"]
    veics = meta["vehicles"]
    return {
        **cond,
        "n_frames": meta["n_frames"],
        "n_vehicles": vl["n_vehicles"],
        "frames_legible_median": meta["frames_legible_median"],
        "plate_px_per_frame_median": _mediana([v["plate_px_per_frame_median"] for v in veics]),
        "motion_blur_px_median": _mediana([v["motion_blur_px_median"] for v in veics]),
        "recall": vl["recall"], "recall_ci_low": vl["recall_ci"][0], "recall_ci_high": vl["recall_ci"][1],
        "precision": vl["precision"], "f1": vl["f1"],
        "n_duplicate_events": vl["n_duplicate_events"],
        "decision_delay_median_s": vl["decision_delay_median_s"],
        "first_decision_recall": vl.get("first_decision_recall", vl["recall"]),
        "precision_emitted": vl.get("precision_emitted", vl.get("precision")),
        "n_revisions": vl.get("n_revisions", 0), "n_exit_relaxed": vl.get("n_exit_relaxed", 0),
        "fusion_n": fc.get("n"), "single_read_acc": fc.get("single_read"),
        "fused_acc": fc.get("fused"), "published_acc": fc.get("published"),
        "latency_p50_ms": perf["latency_p50_ms"], "latency_p99_ms": perf["latency_p99_ms"],
        "drop_rate": perf["drop_rate"], "deadline_miss_rate": perf["deadline_miss_rate"],
        "session_id": rel["session_id"],
    }


def linhas_veiculos(cond: dict, relatorio: Path, meta: dict) -> list[dict]:
    """O CSV por veiculo do relatorio, com as condicoes e as covariaveis da geracao."""
    csv_veic = relatorio.with_name(relatorio.stem + "_veiculos.csv")
    covar = {f"id:{v['track_id']}": v for v in meta["vehicles"]}
    linhas = []
    with csv_veic.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = covar.get(r["vehicle"], {})
            linhas.append({
                **cond, **r,
                "speed_scene_per_frame": g.get("speed_scene_per_frame"),
                "plate_px_per_frame": g.get("plate_px_per_frame_median"),
                "motion_blur_px": g.get("motion_blur_px_median"),
                "frames_legible": g.get("frames_legible"),
            })
    return linhas


def _gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    if not linhas:
        return
    campos: list[str] = []
    for linha in linhas:
        campos += [k for k in linha if k not in campos]
    with caminho.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def grafico(caminho: Path, resumo: list[dict]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series: dict[tuple, list[dict]] = {}
    for r in resumo:
        series.setdefault((r["variant"], r["mode"], r["shutter_ms"]), []).append(r)
    varias = len({r["variant"] for r in resumo}) > 1
    for (var, modo, obt), linhas in sorted(series.items()):
        por_vel: dict[float, list[dict]] = {}
        for r in linhas:
            por_vel.setdefault(r["speed"], []).append(r)
        xs = sorted(por_vel)
        # Repeticoes: media da revocacao e o IC mais largo entre elas.
        ys = [sum(r["recall"] for r in por_vel[x]) / len(por_vel[x]) for x in xs]
        lo = [y - min(r["recall_ci_low"] for r in por_vel[x]) for x, y in zip(xs, ys, strict=True)]
        hi = [max(r["recall_ci_high"] for r in por_vel[x]) - y for x, y in zip(xs, ys, strict=True)]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                    linestyle="-" if modo == "offline" else "--",
                    label=(f"{var}, " if varias else "") + f"{modo}, obturador {obt:g} ms")
    ax.set_xlabel("Fator de velocidade dos veiculos")
    ax.set_ylabel("Revocacao por veiculo (IC 95%)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/curva_velocidade")
    ap.add_argument("--speeds", type=float, nargs="+", default=[1, 1.5, 2, 3, 4])
    ap.add_argument("--shutters", type=float, nargs="+", default=[0, 4],
                    help="tempos de obturador em ms (0 = sem borrao)")
    ap.add_argument("--total-vehicles", type=int, default=60)
    ap.add_argument("--vehicles", type=int, default=5, help="veiculos simultaneos")
    ap.add_argument("--pedestrians", type=int, default=2)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min-plate-width", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", nargs="+", choices=["offline", "realtime"], default=["offline"])
    ap.add_argument("--repeat", type=int, default=3, help="repeticoes em tempo real")
    ap.add_argument("--config", action="append", default=[], help="YAML(s) do lprts eval")
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="reaproveitar relatorios ja gravados em <out> (mesmos parametros!)")
    ap.add_argument("--variant", action="append", default=[],
                    help="nome=secao.chave:valor,... (repetivel; 'nome=' = sem sobreposicao)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    configs = list(args.config)
    if not args.save_crops:
        sem_recortes = out / "_sem_recortes.yaml"
        sem_recortes.write_text("analytics:\n  save_crops: false\n", encoding="utf-8")
        configs.append(str(sem_recortes))

    variantes = [variante(v, out) for v in args.variant] or [("padrao", None)]
    resumo: list[dict] = []
    veiculos: list[dict] = []
    for shutter in args.shutters:
        for speed in args.speeds:
            video, gt, meta = gerar_video(out / "videos", args, speed, shutter)
            for (nome_var, yaml_var), modo in itertools.product(variantes, args.modes):
                saida = out / "avaliacoes" / f"{video.stem}_{nome_var}_{modo}.json"
                cfgs = configs + ([str(yaml_var)] if yaml_var else [])
                for rep, relatorio in enumerate(
                    avaliar(video, gt, saida, modo, args.repeat, cfgs, extra, args.reuse),
                    start=1,
                ):
                    rel = json.loads(relatorio.read_text(encoding="utf-8"))
                    cond = {"variant": nome_var, "speed": speed, "shutter_ms": shutter,
                            "exposure_fraction": meta["exposure_fraction_of_frame"],
                            "mode": modo, "rep": rep}
                    resumo.append(linha_resumo(cond, rel, meta))
                    veiculos.extend(linhas_veiculos(cond, relatorio, meta))
                    r = resumo[-1]
                    print(f"    [{nome_var}] rep {rep}: revocacao {r['recall']:.3f} "
                          f"[{r['recall_ci_low']:.3f}, {r['recall_ci_high']:.3f}]  "
                          f"fusao {r['fused_acc']}  publicada {r['published_acc']}", flush=True)

    _gravar_csv(out / "resumo.csv", resumo)
    _gravar_csv(out / "veiculos.csv", veiculos)
    from lprts.detect.backends import describe_environment

    (out / "experimento.json").write_text(json.dumps({
        "command": [sys.executable, *sys.argv],
        "args": vars(args), "lprts_eval_extra_args": extra, "configs": configs,
        "environment": describe_environment(include_freeze=False),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tem_grafico = grafico(out / "curva.png", resumo)
    print(f"\nResumo ....: {out / 'resumo.csv'} ({len(resumo)} linhas)")
    print(f"Veiculos ..: {out / 'veiculos.csv'} ({len(veiculos)} linhas)")
    if tem_grafico:
        print(f"Grafico ...: {out / 'curva.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
