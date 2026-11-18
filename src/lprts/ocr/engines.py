"""Adaptadores para motores de OCR profundo, todos executados localmente.

Motores suportados (instale so o que for usar):

===============  ==========================  ==========================================
nome             pacote                      observacao
===============  ==========================  ==========================================
``fastplate``    ``fast-plate-ocr``          Especifico para placas, ONNX, o mais rapido.
``paddle``       ``paddleocr`` (2.x)         OCR generico muito preciso; modelo maior.
``easyocr``      ``easyocr``                 Generico, PyTorch; usa CUDA ou MPS se houver.
``tesseract``    ``pytesseract`` + binario   Classico; util como terceira referencia.
``template``     (nenhum)                    Baseline classico interno, sempre disponivel.
===============  ==========================  ==========================================

Todo adaptador devolve o mesmo ``PlateRead``, ja normalizado e validado pela
gramatica brasileira, para que possam ser trocados e comparados sem alterar o
restante do pipeline.

Pedir um motor que nao esta instalado e um **erro**, nao uma troca silenciosa:
numa ablacao, ``--ocr paddle`` que roda o baseline por baixo produziria um
relatorio rotulado "paddle" com numeros de outro motor. So ``auto`` escolhe o
melhor disponivel - e a escolha fica registrada na sessao.
"""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

from ..core.types import PlateRead
from ..utils.imgproc import deskew, enhance_plate
from ..utils.logging import get_logger
from .base import OCREngine, build_read, empty_read
from .template_ocr import TemplateOCR

log = get_logger(__name__)


class EngineUnavailableError(RuntimeError):
    """O motor pedido explicitamente nao pode ser carregado nesta maquina."""


def _garantir_3_canais(imagem: np.ndarray) -> np.ndarray:
    """Converte para BGR de 3 canais se vier em escala de cinza.

    Motores treinados sobre recortes coloridos rejeitam entrada 2D; converter e
    barato e evita uma falha por formato que se manifesta como "nada reconhecido".
    """
    import cv2

    if imagem.ndim == 2:
        return cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    if imagem.ndim == 3 and imagem.shape[2] == 4:
        return cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
    return imagem


def _versao(pacote: str) -> str:
    try:
        from importlib.metadata import version

        return version(pacote)
    except Exception:  # noqa: BLE001
        return "?"


class FastPlateOCR(OCREngine):
    """``fast-plate-ocr``: modelos ONNX treinados especificamente em placas.

    E a escolha recomendada para tempo real: entrada pequena (64x128),
    inferencia em poucos milissegundos na CPU, e devolve confianca por
    caractere - exatamente o que a fusao multi-frame usa para ponderar os votos.
    """

    name = "fastplate"

    def __init__(
        self,
        model_name: str = "cct-s-v1-global-model",
        device: str = "auto",
        force_coreml: bool = False,
        threads: int = 4,
    ) -> None:
        """
        ``device`` e aceito por simetria com os demais motores, mas por padrao
        este roda em **CPU**, mesmo em Apple Silicon. A razao e medida: o grafo
        deste modelo tem 384 nos, dos quais o CoreML aceita 239 - distribuidos em
        **56 particoes**. Cada particao e uma troca de contexto entre CPU e
        Neural Engine, e o custo dessas trocas supera o ganho num modelo tao
        pequeno. Use ``force_coreml=True`` para medir por conta propria.

        ``threads`` fixa as threads do onnxruntime desta sessao (``ocr.threads``):
        com o padrao da biblioteca, o resultado dependia dos nucleos da maquina.
        A espera ativa fica desligada - as threads dormem entre leituras.
        """
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "fast-plate-ocr nao instalado. Rode: pip install '.[ocr]'"
            ) from exc

        providers: list | None = ["CPUExecutionProvider"]
        if force_coreml:
            from ..detect.backends import resolve_device

            providers = list(resolve_device(device).onnx_providers)
            log.info("fast-plate-ocr: CoreML forcado pelo usuario (providers=%s)", providers)

        import onnxruntime as ort

        opcoes = ort.SessionOptions()
        if threads > 0:
            opcoes.intra_op_num_threads = threads
        opcoes.inter_op_num_threads = 1
        opcoes.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.threads = threads
        self.model_name = model_name
        # O nome vem do YAML (str); a biblioteca o tipa como os modelos do hub e o
        # valida ao carregar.
        self._model = LicensePlateRecognizer(cast(Any, model_name), providers=providers,
                                             sess_options=opcoes)
        config = getattr(self._model, "config", None)
        self._pad = str(getattr(config, "pad_char", "_") or "_")
        self._modo_cor = str(getattr(config, "image_color_mode", "rgb") or "rgb").lower()

    def _preparar(self, recorte: np.ndarray) -> np.ndarray:
        """Converte o recorte BGR (OpenCV) no modo de cor que o modelo espera.

        A biblioteca assume que arrays em memoria ja estao no modo do modelo
        (RGB para o modelo global); entregar BGR inverte os canais azul e
        vermelho - a tarja azul da placa Mercosul chegaria vermelha.
        """
        import cv2

        imagem = _garantir_3_canais(deskew(recorte))
        if self._modo_cor == "grayscale":
            return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    def read(self, plate_crop: np.ndarray) -> PlateRead:
        t0 = time.perf_counter()
        if plate_crop is None or plate_crop.size == 0:
            return empty_read(self.name)
        entrada = self._preparar(plate_crop)
        try:
            # remove_pad_char=False: precisamos das posicoes de preenchimento para
            # alinhar as probabilidades aos caracteres reais. O modelo global tem 9
            # posicoes; uma placa brasileira ocupa 7 e as 2 restantes sao padding,
            # cuja probabilidade (quase sempre ~1) inflava a confianca media.
            try:
                predicoes = self._model.run(
                    entrada, return_confidence=True, remove_pad_char=False
                )
            except TypeError:
                # Versoes sem o parametro removem o padding final; as posicoes
                # restantes continuam alinhadas ao inicio do vetor.
                predicoes = self._model.run(entrada, return_confidence=True)
        except Exception as exc:  # noqa: BLE001
            self._registrar_falha(exc)
            return empty_read(self.name)

        if not predicoes:
            return empty_read(self.name)

        pred = predicoes[0]
        slots = str(getattr(pred, "plate", "") or "")
        probs = getattr(pred, "char_probs", None)
        probs_l = np.asarray(probs, dtype=float).ravel().tolist() if probs is not None else []
        manter = [i for i, ch in enumerate(slots) if ch != self._pad]
        texto = "".join(slots[i] for i in manter)
        confs = [probs_l[i] for i in manter if i < len(probs_l)]
        if len(confs) != len(texto):
            confs = []
        confianca = float(np.mean(confs)) if confs else 0.0
        self._sucesso()
        return build_read(texto, confianca, self.name, t0, confs, self.max_corrections)

    def describe(self) -> dict[str, Any]:
        sessao = getattr(self._model, "model", None)
        try:
            providers = list(sessao.get_providers()) if sessao is not None else []
        except Exception:  # noqa: BLE001
            providers = []
        return {
            "name": self.name,
            "runtime": "onnxruntime",
            "device": "coreml" if "CoreMLExecutionProvider" in providers else "cpu",
            "providers": providers,
            "model": self.model_name,
            "package_version": _versao("fast-plate-ocr"),
            "color_mode": self._modo_cor,
            "threads": self.threads,
        }


class PaddleOCREngine(OCREngine):
    """PaddleOCR 2.x - reconhecimento generico, muito preciso.

    A API 3.x mudou (``use_gpu``, ``show_log``, ``det``/``cls`` deixaram de
    existir); o pyproject fixa ``paddleocr<3`` por isso.
    """

    name = "paddle"

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover
            raise ImportError("paddleocr nao instalado. Rode: pip install '.[paddle]'") from exc
        self._use_gpu = use_gpu
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    def read(self, plate_crop: np.ndarray) -> PlateRead:
        t0 = time.perf_counter()
        if plate_crop is None or plate_crop.size == 0:
            return empty_read(self.name)
        try:
            # det=False: a placa ja foi recortada pelo detector; rodar a deteccao
            # de texto de novo so acrescenta latencia e chances de erro.
            result = self._ocr.ocr(plate_crop, det=False, cls=True)
        except Exception as exc:  # noqa: BLE001
            self._registrar_falha(exc)
            return empty_read(self.name)
        pieces: list[tuple[str, float]] = []
        for line in result or []:
            for item in line or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pieces.append((str(item[0]), float(item[1])))
        self._sucesso()
        if not pieces:
            return empty_read(self.name)
        text = "".join(p[0] for p in pieces)
        confidence = float(np.mean([p[1] for p in pieces]))
        return build_read(text, confidence, self.name, t0, None, self.max_corrections)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "runtime": "paddle",
            "device": "gpu" if self._use_gpu else "cpu",
            "package_version": _versao("paddleocr"),
        }


class EasyOCREngine(OCREngine):
    """EasyOCR - PyTorch; usa CUDA ou, no Apple Silicon, MPS."""

    name = "easyocr"

    def __init__(self, device: str = "auto") -> None:
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover
            raise ImportError("easyocr nao instalado. Rode: pip install '.[ocr]'") from exc
        # gpu=True deixa o proprio EasyOCR escolher CUDA, depois MPS, depois CPU.
        # A versao anterior so ligava a GPU quando o dispositivo resolvido era
        # "cuda" ou "mps" - e no Mac o "auto" resolve para "coreml", entao o
        # EasyOCR rodava em CPU enquanto a documentacao dizia MPS.
        quer_gpu = (device or "auto").lower() != "cpu"
        self._reader = easyocr.Reader(["en"], gpu=quer_gpu, verbose=False)
        # Restringir o alfabeto ao das placas ja remove metade dos erros possiveis.
        self._allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def read(self, plate_crop: np.ndarray) -> PlateRead:
        t0 = time.perf_counter()
        if plate_crop is None or plate_crop.size == 0:
            return empty_read(self.name)
        gray = enhance_plate(deskew(plate_crop), target_height=64)
        try:
            results = self._reader.readtext(gray, allowlist=self._allowlist, detail=1)
        except Exception as exc:  # noqa: BLE001
            self._registrar_falha(exc)
            return empty_read(self.name)
        self._sucesso()
        if not results:
            return empty_read(self.name)
        results.sort(key=lambda r: r[0][0][0])  # ordem de leitura (esquerda->direita)
        text = "".join(str(r[1]) for r in results)
        confidence = float(np.mean([float(r[2]) for r in results]))
        return build_read(text, confidence, self.name, t0, None, self.max_corrections)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "runtime": "pytorch",
            "device": str(getattr(self._reader, "device", "?")),
            "package_version": _versao("easyocr"),
        }


class TesseractEngine(OCREngine):
    """Tesseract via ``pytesseract`` - baseline classico amplamente citado."""

    name = "tesseract"

    def __init__(self, psm: int = 7) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pytesseract nao instalado. Rode: pip install pytesseract && brew install tesseract"
            ) from exc
        self._pt = pytesseract
        # psm 7 = "uma unica linha de texto", que descreve exatamente uma placa.
        self._config = (
            f"--psm {psm} --oem 3 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    def read(self, plate_crop: np.ndarray) -> PlateRead:
        t0 = time.perf_counter()
        if plate_crop is None or plate_crop.size == 0:
            return empty_read(self.name)
        gray = enhance_plate(deskew(plate_crop), target_height=64)
        try:
            data = self._pt.image_to_data(
                gray, config=self._config, output_type=self._pt.Output.DICT
            )
        except Exception as exc:  # noqa: BLE001
            self._registrar_falha(exc)
            return empty_read(self.name)
        self._sucesso()
        words, confs = [], []
        for txt, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            if txt and txt.strip() and float(conf) >= 0:
                words.append(txt.strip())
                confs.append(float(conf) / 100.0)
        if not words:
            return empty_read(self.name)
        return build_read("".join(words), float(np.mean(confs)), self.name, t0, None,
                          self.max_corrections)

    def describe(self) -> dict[str, Any]:
        try:
            versao = str(self._pt.get_tesseract_version())
        except Exception:  # noqa: BLE001
            versao = "?"
        return {"name": self.name, "runtime": "tesseract", "device": "cpu",
                "package_version": versao}


_ENGINES: dict[str, type[OCREngine]] = {
    "fastplate": FastPlateOCR,
    "paddle": PaddleOCREngine,
    "easyocr": EasyOCREngine,
    "tesseract": TesseractEngine,
    "template": TemplateOCR,
}

# Ordem tentada no modo "auto": do mais adequado a placas ao mais generico.
_AUTO_ORDER = ("fastplate", "paddle", "easyocr", "template")

# Configuracao de um adaptador nao e argumento valido para os outros. Em
# particular, model_name/force_coreml so pertencem ao FastPlate.
_ENGINE_OPTIONS = {
    "fastplate": {"model_name", "device", "force_coreml", "threads"},
    "paddle": {"lang", "use_gpu"},
    "easyocr": {"device"},
    "tesseract": {"psm"},
    "template": {"target_height", "min_confidence"},
}

_MODULOS = {
    "fastplate": "fast_plate_ocr",
    "paddle": "paddleocr",
    "easyocr": "easyocr",
    "tesseract": "pytesseract",
}


def available_engines() -> list[str]:
    """Nomes dos motores efetivamente instalaveis nesta maquina."""
    import importlib.util

    found = [n for n, mod in _MODULOS.items() if importlib.util.find_spec(mod) is not None]
    return [*found, "template"]  # o baseline interno esta sempre disponivel


def build_ocr_engine(name: str = "auto", **kwargs) -> OCREngine:
    """Instancia um motor pelo nome.

    ``auto`` escolhe o melhor disponivel e cai para o baseline ``template`` se
    nada mais carregar. Um nome explicito e um pedido firme: se o motor nao
    carregar, levanta ``EngineUnavailableError`` com a instrucao de instalacao.
    """
    key = (name or "auto").lower().strip()

    if key == "auto":
        for candidate in _AUTO_ORDER:
            try:
                options = {k: v for k, v in kwargs.items() if k in _ENGINE_OPTIONS[candidate]}
                engine = _ENGINES[candidate](**options)
                log.info("Motor de OCR selecionado automaticamente: %s", candidate)
                return engine
            except Exception as exc:  # noqa: BLE001
                log.debug("Motor '%s' indisponivel: %s", candidate, exc)
        return TemplateOCR()

    if key not in _ENGINES:
        raise ValueError(
            f"Motor de OCR desconhecido: {name!r}. Opcoes: {sorted(_ENGINES)} ou 'auto'."
        )
    try:
        return _ENGINES[key](**kwargs)
    except TypeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EngineUnavailableError(
            f"O motor de OCR '{key}' foi pedido explicitamente mas nao pode ser "
            f"carregado: {exc}\nInstale-o, ou use --ocr auto (escolhe o melhor "
            "disponivel e registra qual) ou --ocr template (baseline interno)."
        ) from exc
