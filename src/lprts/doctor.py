"""Diagnostico do ambiente: verifica se a maquina esta pronta para executar.

Responde a pergunta pratica "posso rodar?" sem que o usuario precise interpretar
mensagens de erro de import. Cada verificacao reporta um dos tres estados:

* **ok** - funciona;
* **aviso** - funciona, mas em modo degradado (e diz o que se perde);
* **erro** - impede a execucao (e diz o comando que resolve).

A distincao importa: a maior parte das dependencias deste projeto e opcional. Sem
PySide6 nao ha interface, mas a linha de comando roda; sem pesos YOLO nao ha
deteccao multi-classe, mas o baseline classico roda. Um verificador que tratasse
tudo como erro assustaria sem motivo.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

OK, AVISO, ERRO = "ok", "aviso", "erro"


@dataclass(slots=True)
class Check:
    """Resultado de uma verificacao."""

    nome: str
    estado: str
    detalhe: str = ""
    solucao: str = ""


def _tem(modulo: str) -> bool:
    try:
        return importlib.util.find_spec(modulo) is not None
    except (ImportError, ValueError):
        return False


def _versao(modulo: str) -> str:
    try:
        m = importlib.import_module(modulo)
        return str(getattr(m, "__version__", "?"))
    except Exception:  # noqa: BLE001
        return "?"


# ---------------------------------------------------------------------------
# Verificacoes
# ---------------------------------------------------------------------------

def check_python() -> Check:
    v = sys.version_info
    versao = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 10):
        return Check("Python", ERRO, versao, "Instale Python 3.11: brew install python@3.11")
    if v >= (3, 13):
        return Check(
            "Python", AVISO, f"{versao} (o projeto foi testado ate 3.12)",
            "Se algo falhar, use Python 3.11 ou 3.12",
        )
    return Check("Python", OK, versao)


def check_nucleo() -> list[Check]:
    """Dependencias sem as quais nada roda."""
    checks = []
    for modulo, rotulo in (("numpy", "numpy"), ("cv2", "OpenCV"), ("yaml", "PyYAML"),
                           ("typer", "typer"), ("rich", "rich")):
        if _tem(modulo):
            checks.append(Check(rotulo, OK, _versao(modulo)))
        else:
            checks.append(
                Check(rotulo, ERRO, "ausente", "pip install -e .")
            )
    return checks


def check_opencv_duplo() -> Check:
    """``opencv-python`` e ``opencv-python-headless`` juntos sobrescrevem o mesmo ``cv2``.

    Acontece porque o ultralytics depende de um e o easyocr do outro. O que foi
    instalado por ultimo vence; se for o headless, a janela de calibracao (que
    usa a interface do OpenCV) deixa de funcionar.
    """
    from .provenance import opencv_conflict, package_version

    if not opencv_conflict():
        return Check("Pacotes do OpenCV", OK, "um unico pacote cv2")
    return Check(
        "Pacotes do OpenCV", AVISO,
        f"opencv-python {package_version('opencv-python')} e opencv-python-headless "
        f"{package_version('opencv-python-headless')} instalados juntos",
        "pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python",
    )


def check_bancos_antigos() -> Check:
    """Bancos criados por versoes anteriores (esquema 1) sao recusados pela atual."""
    from .analytics.store import legacy_schema_version

    antigos = [
        str(db) for db in sorted(Path("runs").glob("**/*.db"))
        if "legado" not in db.parts and legacy_schema_version(db) is not None
    ]
    if not antigos:
        return Check("Bancos de resultados", OK, "nenhum banco de versao anterior")
    return Check(
        "Bancos de resultados", AVISO,
        f"{len(antigos)} banco(s) de versao anterior: {', '.join(antigos[:3])}",
        "Os dados antigos tem erros conhecidos e nao podem ser misturados com os novos. "
        "O ./start.sh os arquiva em runs/legado/; ou: "
        "mkdir -p runs/legado && mv runs/lprts.db* runs/legado/",
    )


def check_interface() -> Check:
    if not _tem("PySide6"):
        return Check(
            "Interface grafica (PySide6)", AVISO, "ausente",
            'pip install -e ".[ui]"   (sem ela, use --headless)',
        )
    return Check("Interface grafica (PySide6)", OK, _versao("PySide6"))


def check_deep_learning() -> list[Check]:
    checks = []
    if _tem("torch"):
        checks.append(Check("PyTorch", OK, _versao("torch")))
    else:
        checks.append(
            Check("PyTorch", AVISO, "ausente", 'pip install -e ".[dl]"   (necessario para YOLO)')
        )
    if _tem("ultralytics"):
        checks.append(Check("ultralytics (YOLO)", OK, _versao("ultralytics")))
    else:
        checks.append(
            Check(
                "ultralytics (YOLO)", AVISO, "ausente",
                'pip install -e ".[dl]"   (sem ela: so o detector classico)',
            )
        )
    if _tem("onnxruntime"):
        checks.append(Check("ONNX Runtime", OK, _versao("onnxruntime")))
    else:
        checks.append(Check("ONNX Runtime", AVISO, "ausente", 'pip install -e ".[dl]"'))
    return checks


def check_ocr() -> Check:
    from .ocr.engines import available_engines

    motores = available_engines()
    profundos = [m for m in motores if m != "template"]
    if profundos:
        return Check("Motores de OCR", OK, ", ".join(motores))
    return Check(
        "Motores de OCR", AVISO, "so o baseline classico (template)",
        'pip install -e ".[ocr]"   (fast-plate-ocr e bem mais preciso)',
    )


def check_dispositivo() -> Check:
    from .detect.backends import is_apple_silicon, resolve_device

    d = resolve_device("auto")
    apple = is_apple_silicon()

    if d.kind == "coreml":
        return Check(
            "Aceleracao de inferencia", OK, f"{d.kind} - {d.description}",
            "O modelo de OCR roda em CPU por padrao mesmo aqui: o CoreML o fatia "
            "em ~56 particoes e o custo das trocas supera o ganho. Se houver "
            "instabilidade tambem na deteccao, use --device cpu.",
        )
    if d.kind in ("mps", "cuda"):
        return Check("Aceleracao de inferencia", OK, f"{d.kind} - {d.description}")
    if apple:
        return Check(
            "Aceleracao de inferencia", AVISO,
            "CPU (Apple Silicon detectado, mas sem CoreML/MPS disponivel)",
            'pip install -e ".[dl]"  e depois  python scripts/download_models.py --coreml',
        )
    return Check(
        "Aceleracao de inferencia", AVISO, f"{d.kind} - {d.description}",
        "Sem GPU/ANE o desempenho cai; e esperado em maquina Intel ou Linux sem CUDA",
    )


def check_modelos() -> list[Check]:
    checks = []
    veiculos = Path("data/models/yolo11n.pt")
    placas = Path("data/models/plate_yolo.pt")

    if veiculos.exists():
        mb = veiculos.stat().st_size / 1e6
        checks.append(Check("Pesos de deteccao de objetos", OK, f"{veiculos} ({mb:.1f} MB)"))
    else:
        checks.append(
            Check(
                "Pesos de deteccao de objetos", AVISO, "ausentes",
                "python scripts/download_models.py   "
                "(sem eles nao ha deteccao de pessoas/bicicletas)",
            )
        )
    if placas.exists():
        checks.append(Check("Pesos de deteccao de placas", OK, str(placas)))
    else:
        checks.append(
            Check(
                "Pesos de deteccao de placas", AVISO, "ausentes (usando o baseline classico)",
                "python scripts/download_models.py --plate morsetech-n   (para testar; "
                "para a tese, treine com scripts/train_plate_detector.py)",
            )
        )
    return checks


def check_combinacao_detectores() -> Check:
    """Avisa sobre a combinacao que mais confunde na primeira execucao.

    Com pesos de objetos presentes mas sem pesos de placa, o sistema detecta
    veiculos com YOLO e placas com o baseline classico. Funciona bem em video
    real; em video sintetico ou estilizado, o YOLO nao reconhece os veiculos e
    - antes do fallback automatico - nada era lido, sem explicacao.
    """
    tem_objetos = Path("data/models/yolo11n.pt").exists()
    tem_placas = Path("data/models/plate_yolo.pt").exists()
    if tem_objetos and not tem_placas:
        return Check(
            "Combinacao de detectores", AVISO,
            "YOLO para objetos + baseline classico para placas",
            "Em video sintetico o YOLO nao reconhece os veiculos; o sistema cai "
            "sozinho para o modo so-placa e avisa. Para fixar: --no-vehicle-detector",
        )
    if not tem_objetos:
        return Check(
            "Combinacao de detectores", OK,
            "modo so-placa (sem pesos de objetos)",
        )
    return Check("Combinacao de detectores", OK, "YOLO para objetos e para placas")


def check_escrita() -> Check:
    alvo = Path("runs")
    try:
        alvo.mkdir(parents=True, exist_ok=True)
        teste = alvo / ".escrita_teste"
        teste.write_text("ok", encoding="utf-8")
        teste.unlink()
    except OSError as exc:
        return Check("Permissao de escrita em runs/", ERRO, str(exc),
                     "Rode a partir da raiz do projeto, ou ajuste analytics.db_path")
    livre = shutil.disk_usage(alvo).free / 1e9
    if livre < 1.0:
        return Check("Permissao de escrita em runs/", AVISO, f"apenas {livre:.1f} GB livres",
                     "Libere espaco: recortes e bancos crescem rapido")
    return Check("Permissao de escrita em runs/", OK, f"{livre:.0f} GB livres")


def check_camera(testar: bool = False) -> Check:
    """Verifica a webcam. So abre o dispositivo quando ``testar`` e True."""
    if not testar:
        return Check("Webcam", AVISO, "nao testada",
                     "Use --camera para testar (no macOS pede permissao do sistema)")
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        aberta = cap.isOpened()
        if aberta:
            ok, quadro = cap.read()
            cap.release()
            if ok and quadro is not None:
                return Check("Webcam", OK, f"{quadro.shape[1]}x{quadro.shape[0]}")
            return Check("Webcam", AVISO, "abriu mas nao entregou quadro")
        cap.release()
    except Exception as exc:  # noqa: BLE001
        return Check("Webcam", AVISO, str(exc))
    return Check(
        "Webcam", AVISO, "nao acessivel",
        "No macOS: Ajustes > Privacidade e Seguranca > Camera > habilite o Terminal",
    )


def check_pipeline() -> Check:
    """Teste de fumaca: monta o pipeline e processa um quadro sintetico."""
    try:
        import numpy as np

        from .config import AppConfig
        from .core.pipeline import LPRPipeline
        from .core.types import Frame
        from .detect.classic_plate import ClassicPlateDetector
        from .ocr.engines import build_ocr_engine

        cfg = AppConfig()
        cfg.detector.detect_vehicles = False
        cfg.analytics.save_crops = False
        pipe = LPRPipeline(
            cfg, None, ClassicPlateDetector(), build_ocr_engine("template"), None
        ).start()
        quadro = Frame(0, np.full((360, 640, 3), 100, dtype=np.uint8), 0.0)
        resultado = pipe.process_frame(quadro)
        pipe.stop()
        return Check(
            "Pipeline (teste de fumaca)", OK,
            f"quadro processado em {resultado.timing.total_ms:.1f} ms",
        )
    except Exception as exc:  # noqa: BLE001
        return Check("Pipeline (teste de fumaca)", ERRO, f"{type(exc).__name__}: {exc}",
                     "Reinstale: pip install -e '.[dev]' e rode pytest tests/")


def check_analise() -> Check:
    presentes = [m for m in ("pandas", "pyarrow", "duckdb", "matplotlib") if _tem(m)]
    if len(presentes) >= 3:
        return Check("Ferramentas de analise", OK, ", ".join(presentes))
    return Check(
        "Ferramentas de analise", AVISO,
        ", ".join(presentes) if presentes else "ausentes",
        'pip install -e ".[analysis]"   (para exportar Parquet e usar DuckDB)',
    )


def check_paralelismo() -> Check:
    import os

    nucleos = os.cpu_count() or 1
    mpi = "com mpi4py" if _tem("mpi4py") else "sem mpi4py (so local)"
    if nucleos >= 4:
        return Check("Paralelismo", OK, f"{nucleos} nucleos, {mpi}")
    return Check(
        "Paralelismo", AVISO, f"apenas {nucleos} nucleo(s), {mpi}",
        "Com poucos nucleos, use --parallel 1 (o padrao serial)",
    )


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------

def run_all(testar_camera: bool = False, rapido: bool = False) -> list[Check]:
    """Roda todas as verificacoes na ordem em que importam."""
    checks: list[Check] = [check_python()]
    checks.extend(check_nucleo())
    checks.append(check_opencv_duplo())
    checks.append(check_interface())
    checks.extend(check_deep_learning())
    checks.append(check_ocr())
    checks.append(check_dispositivo())
    checks.extend(check_modelos())
    checks.append(check_combinacao_detectores())
    checks.append(check_analise())
    checks.append(check_paralelismo())
    checks.append(check_escrita())
    checks.append(check_bancos_antigos())
    checks.append(check_camera(testar_camera))
    if not rapido:
        checks.append(check_pipeline())
    return checks


def resumo(checks: list[Check]) -> tuple[int, int, int]:
    """Contagem de (ok, avisos, erros)."""
    return (
        sum(1 for c in checks if c.estado == OK),
        sum(1 for c in checks if c.estado == AVISO),
        sum(1 for c in checks if c.estado == ERRO),
    )


def proximo_passo(checks: list[Check]) -> str:
    """Sugere o que fazer em seguida, conforme o que foi encontrado."""
    erros = [c for c in checks if c.estado == ERRO]
    if erros:
        return (
            "Resolva os erros acima antes de executar. O primeiro:\n"
            f"  {erros[0].solucao or 'consulte o README'}"
        )

    tem_gui = any(c.nome.startswith("Interface") and c.estado == OK for c in checks)
    tem_modelos = any(c.nome.startswith("Pesos de deteccao de objetos") and c.estado == OK
                      for c in checks)

    linhas = ["Tudo pronto para executar. Sugestao de primeiros comandos:", ""]
    linhas.append("  lprts demo                    # gera video sintetico, processa e avalia")
    if tem_gui:
        linhas.append("  lprts run --source data/videos/sintetico.mp4    # interface grafica")
        linhas.append("  lprts run --source 0 --realtime                 # webcam ao vivo")
    else:
        linhas.append("  lprts run --source video.mp4 --headless          # sem interface")
        linhas.append('  pip install -e ".[ui]"                           # para ter a interface')
    if not tem_modelos:
        linhas.append("")
        linhas.append("  python scripts/download_models.py   # habilita deteccao multi-classe")
    return "\n".join(linhas)


def ambiente() -> dict[str, str]:
    from .provenance import code_version, hardware

    hw = hardware()
    return {
        "Sistema": platform.platform(),
        # platform.processor() devolve so "arm" no Apple Silicon.
        "Processador": hw.get("cpu_model") or platform.processor() or platform.machine(),
        "Python": sys.executable,
        "Codigo": code_version(),
        "Diretorio": str(Path.cwd()),
    }
