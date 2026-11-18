"""Detector YOLO (ultralytics) para veiculos e para placas.

Tres runtimes intercambiaveis, escolhidos pela extensao dos pesos:

* ``coreml`` (``.mlpackage``) - o modelo roda no Neural Engine pelo coremltools,
  com pre/pos-processamento feito aqui. E o caminho de tempo real no Apple
  Silicon: o ultralytics gastava 3-4x a propria inferencia em preparo.
* ``ultralytics`` - aceita ``.pt`` (PyTorch: MPS/CUDA/CPU) e ``.mlpackage``; e o
  caminho do treino e a referencia (``detector.runtime: ultralytics``).
* ``onnxruntime`` - carrega ``.onnx`` diretamente, com o mesmo pre/pos-processamento
  do caminho CoreML.

Ambos rodam **localmente**, na CPU/GPU/ANE da maquina.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.types import Detection
from ..utils.imgproc import letterbox, unletterbox_boxes
from ..utils.logging import get_logger
from .backends import DeviceInfo, resolve_device
from .base import Detector, file_sha256, nms

log = get_logger(__name__)

# Classes de veiculo no COCO, usadas pelos pesos pre-treinados yolo11*.pt.
# Mantido por compatibilidade; prefira os presets de ``core.taxonomy``.
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@lru_cache(maxsize=1)
def _precision_param() -> str:
    """Nome do parametro de meia precisao aceito pelo ultralytics instalado.

    As versoes recentes renomearam ``half`` para ``quantize`` e emitem um aviso de
    depreciacao a CADA chamada de ``predict`` - centenas de linhas por minuto num
    video, afogando os logs uteis. Descobrimos o nome correto uma unica vez, por
    introspecao da assinatura, em vez de fixar um dos dois e quebrar na outra
    versao.
    """
    try:
        import inspect

        from ..runtime import import_ultralytics

        import_ultralytics()
        from ultralytics.engine.model import Model

        if "quantize" in inspect.signature(Model.predict).parameters:
            return "quantize"
    except Exception:  # noqa: BLE001 - introspecao e best-effort
        pass
    return "half"


def coreml_export_info(weights: str) -> dict | None:
    """Parametros fixados na exportacao de um modelo CoreML do ultralytics.

    Um ``.mlpackage`` tem entrada de tamanho fixo e, se exportado com
    ``nms=True``, NMS embutido com os limiares da exportacao: o ultralytics
    passa a ignorar ``imgsz`` e ``iou`` da predicao **sem avisar** (medido: as
    mesmas 10 deteccoes com iou 0,1, 0,45 ou 0,9). Le os metadados gravados
    pelo ultralytics; ``None`` se nao for possivel (sem coremltools, outro
    exportador).
    """
    import ast

    try:
        import coremltools as ct

        meta = dict(ct.models.MLModel(weights, skip_model_load=True).user_defined_metadata)
        args = ast.literal_eval(meta.get("args", "{}"))
        imgsz = ast.literal_eval(meta["imgsz"])
    except Exception:  # noqa: BLE001 - metadados sao best-effort
        return None
    return {
        "imgsz": [int(v) for v in imgsz],
        "nms_embedded": bool(args.get("nms", False)),
        "iou_embedded": float(meta["IoU threshold"]) if "IoU threshold" in meta else None,
        "conf_embedded": (
            float(meta["Confidence threshold"]) if "Confidence threshold" in meta else None
        ),
        "precision": "fp16" if args.get("quantize") == 16 or args.get("half") else "fp32",
    }


class YOLODetector(Detector):
    """Envolve um modelo YOLO, escolhendo o runtime pela extensao do arquivo."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | DeviceInfo = "auto",
        classes: dict[int, str] | None = None,
        runtime: str = "auto",
        half: bool = True,
        name: str = "yolo",
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.half = half
        self.name = name
        self.device = device if isinstance(device, DeviceInfo) else resolve_device(str(device))

        suffix = self.weights.suffix.lower()
        if runtime == "auto":
            runtime = {".onnx": "onnx", ".mlpackage": "coreml"}.get(suffix, "ultralytics")
        if runtime == "coreml" and not self._is_coreml:
            runtime = "ultralytics"       # .pt pedido como coreml: so o ultralytics le
        if runtime not in ("coreml", "ultralytics", "onnx"):
            raise ValueError(f"Runtime de deteccao desconhecido: {runtime!r} "
                             "(use auto, coreml, ultralytics ou onnx).")
        self.runtime = runtime

        # APIs opcionais de terceiros: os objetos concretos dependem do runtime.
        self._model: Any = None
        self._session: Any = None
        self._ml: Any = None
        self._input_name = ""
        self._onnx_size = (self.imgsz, self.imgsz)
        self._onnx_dtype: np.dtype = np.dtype(np.float32)
        self._coreml_info: dict | None = None
        self._load()

    # -- carregamento ------------------------------------------------------

    def _load(self) -> None:
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Pesos nao encontrados: {self.weights}\n"
                "Execute `python scripts/download_models.py` para obter os modelos."
            )
        if self.runtime == "onnx":
            self._load_onnx()
        elif self.runtime == "coreml":
            self._load_coreml()
        else:
            self._load_ultralytics()
        log.info(
            "Detector '%s' carregado: %s (runtime=%s, device=%s)",
            self.name, self.weights.name, self.runtime, self.device,
        )

    def _load_ultralytics(self) -> None:
        try:
            from ..runtime import import_ultralytics

            import_ultralytics()      # sem isto o OpenCV do processo fica numa thread
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics nao instalado. Rode: pip install '.[dl]'"
            ) from exc
        self._model = YOLO(str(self.weights))
        if self._is_coreml:
            self._check_coreml_export()

    def _check_coreml_export(self) -> None:
        """Alinha os parametros registrados aos que o modelo CoreML de fato usa."""
        info = coreml_export_info(str(self.weights))
        self._coreml_info = info
        if info is None:
            return
        h, w = info["imgsz"]
        if h == w and h != self.imgsz:
            log.warning(
                "%s tem entrada fixa de %d px; imgsz=%d e ignorado (registrado o efetivo).",
                self.weights.name, h, self.imgsz,
            )
            self.imgsz = h
        if info["nms_embedded"]:
            log.warning(
                "%s foi exportado com NMS embutido: iou=%.2f e ignorado (vale %s, fixado na "
                "exportacao) e conf abaixo de %s nao tem efeito. Reexporte sem NMS: "
                "python scripts/download_models.py --coreml",
                self.weights.name, self.iou, info["iou_embedded"], info["conf_embedded"],
            )

    def _load_coreml(self) -> None:
        """CoreML sem o ultralytics: letterbox, inferencia e decodificacao aqui.

        Medido no M4 Pro com o mesmo ``.mlpackage``: pelo ultralytics, 6,6 ms por
        quadro de 1280 px (YOLO11n, 640) e 2,1 ms por recorte de veiculo no
        detector de placas (320); a inferencia no Neural Engine e 1,6 e 0,5 ms.
        O resto era preparo (uint8 -> float -> uint8 -> PIL) e pos-processamento
        em PyTorch. Um modelo com NMS embutido ou entrada que nao e imagem volta
        para o ultralytics.
        """
        try:
            import coremltools as ct
        except ImportError:
            log.info("coremltools ausente: %s roda pelo ultralytics.", self.weights.name)
            self.runtime = "ultralytics"
            self._load_ultralytics()
            return
        self._check_coreml_export()
        if self._coreml_info is not None and self._coreml_info["nms_embedded"]:
            self.runtime = "ultralytics"
            self._load_ultralytics()
            return
        # CPU_AND_NE, como o ultralytics: ALL/CPU_AND_GPU derrubam o processo por
        # um defeito do compilador MPSGraph (coremltools 9.x).
        try:
            self._ml = ct.models.MLModel(str(self.weights), compute_units=ct.ComputeUnit.CPU_AND_NE)
        except Exception:  # noqa: BLE001 - macOS < 13 sem CPU_AND_NE
            self._ml = ct.models.MLModel(str(self.weights), compute_units=ct.ComputeUnit.CPU_ONLY)
        entrada = self._ml.get_spec().description.input[0]
        if not entrada.type.HasField("imageType"):
            self._ml = None
            self.runtime = "ultralytics"
            self._load_ultralytics()
            return
        self._input_name = entrada.name
        lado = int(entrada.type.imageType.width)
        if lado and lado != self.imgsz:
            self.imgsz = lado            # a entrada do .mlpackage e fixa

    def _load_onnx(self) -> None:
        from ..runtime import configure_onnxruntime

        configure_onnxruntime()
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError("onnxruntime nao instalado. Rode: pip install '.[dl]'") from exc
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Determinismo de latencia importa mais que throughput bruto em RT.
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(self.weights), sess_options=opts, providers=list(self.device.onnx_providers)
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("ONNX requer uma unica entrada de imagem NCHW.")
        entrada = inputs[0]
        shape = entrada.shape
        if len(shape) != 4 or (
            isinstance(shape[0], int) and shape[0] != 1
        ) or (isinstance(shape[1], int) and shape[1] != 3):
            raise ValueError(f"Entrada ONNX nao suportada: {shape}; esperado [1, 3, H, W].")
        types: dict[str, np.dtype] = {
            "tensor(float)": np.dtype(np.float32), "tensor(float16)": np.dtype(np.float16),
        }
        if entrada.type not in types:
            raise ValueError(f"Tipo de entrada ONNX nao suportado: {entrada.type}.")
        self._onnx_dtype = types[entrada.type]
        h, w = (d if isinstance(d, int) else self.imgsz for d in shape[2:])
        if h <= 0 or w <= 0:
            raise ValueError(f"Dimensoes de entrada ONNX invalidas: {shape}.")
        self._onnx_size = (w, h)
        if h == w and self.imgsz != h:
            log.warning("%s tem entrada ONNX fixa de %d px; imgsz=%d ignorado.",
                        self.weights.name, h, self.imgsz)
            self.imgsz = h
        self._input_name = entrada.name

    # -- inferencia --------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        if self.runtime == "coreml":
            return self._detect_coreml(image)
        if self.runtime == "onnx":
            return self._detect_onnx(image)
        return self._detect_ultralytics(image)

    def _precision_kwargs(self) -> dict:
        """Argumentos de precisao, omitidos quando nao ha nada a ativar.

        Passar ``half=False`` nao desliga nada e ainda assim dispara o aviso de
        depreciacao; entao simplesmente nao passamos o parametro nesse caso.
        """
        if not (self.half and self.device.kind == "cuda"):
            return {}
        return {_precision_param(): True}

    @property
    def _is_coreml(self) -> bool:
        return self.weights.suffix.lower() in (".mlpackage", ".mlmodel")

    def _detect_ultralytics(self, image: np.ndarray) -> list[Detection]:
        assert self._model is not None
        # O CoreML escolhe as proprias unidades de computo (Neural Engine/GPU);
        # o "device" do ultralytics so governa o pos-processamento em PyTorch.
        # Passar "mps" ali fazia toda deteccao falhar (o NMS converte para
        # float64, que o MPS nao suporta) - o pos-processamento vai para a CPU.
        dispositivo = "cpu" if self._is_coreml else self.device.torch_device
        results = self._model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=dispositivo,
            classes=list(self.classes) if self.classes else None,
            verbose=False,
            **self._precision_kwargs(),
        )
        out: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            names = getattr(res, "names", {}) or {}
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss, strict=True):
                label = (self.classes or {}).get(int(cls_id)) or names.get(int(cls_id), str(cls_id))
                out.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        score=float(score), label=str(label), class_id=int(cls_id),
                    )
                )
        return out

    def _detect_coreml(self, image: np.ndarray) -> list[Detection]:
        assert self._ml is not None
        from PIL import Image

        blob, scale, pad = letterbox(image, (self.imgsz, self.imgsz))
        # cvtColor e nao blob[:, :, ::-1]: o fromarray de uma visao com passo
        # negativo copia elemento a elemento (3,3 ms a 640 px, contra 0,2 ms).
        saida = self._ml.predict({self._input_name: Image.fromarray(
            cv2.cvtColor(blob, cv2.COLOR_BGR2RGB))})
        raw = next((v for v in saida.values() if np.ndim(v) == 3), None)
        if raw is None:
            return []
        return self._decode(raw, scale, pad, image.shape[:2])

    def _detect_onnx(self, image: np.ndarray) -> list[Detection]:
        assert self._session is not None
        blob, scale, pad = letterbox(image, self._onnx_size)
        tensor = blob[:, :, ::-1].transpose(2, 0, 1)[None].astype(self._onnx_dtype) / 255.0
        raw = self._session.run(None, {self._input_name: np.ascontiguousarray(tensor)})[0]
        return self._decode(raw, scale, pad, image.shape[:2])

    def _decode(
        self, raw: np.ndarray, scale: float, pad: tuple[int, int], hw: tuple[int, int]
    ) -> list[Detection]:
        """Saida crua do YOLOv8/v11 ``(1, 4+nc, N)`` -> deteccoes na imagem de entrada.

        Mesma semantica do ultralytics: classe = a de maior score entre TODAS,
        limiar de confianca, filtro de classes depois da escolha e NMS por classe.
        """
        pred = np.asarray(raw, dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] > pred.shape[1]:     # (N, 4+nc) -> (4+nc, N)
            pred = pred.T
        if pred.shape[0] < 5:
            return []
        cls = pred[4:]
        # Maximo primeiro, argmax so nas colunas que passam: com 80 classes e
        # 8400 ancoras e a parte cara da decodificacao.
        scores = cls.max(axis=0)
        keep = np.flatnonzero(scores >= self.conf)
        if keep.size == 0:
            return []
        class_ids = cls[:, keep].argmax(axis=0) if cls.shape[0] > 1 else np.zeros(keep.size, int)
        scores = scores[keep]

        cx, cy, bw, bh = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        boxes = unletterbox_boxes(boxes, scale, pad)
        h, w = hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        dets = [
            Detection(
                bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(sc),
                label=(self.classes or {}).get(int(c), str(int(c))),
                class_id=int(c),
            )
            for b, sc, c in zip(boxes, scores, class_ids, strict=True)
            if self.classes is None or int(c) in self.classes
        ]
        return nms(dets, self.iou)

    def describe(self) -> dict:
        """Runtime e dispositivo efetivos.

        O dispositivo "resolvido" nao basta: com pesos ``.pt`` o ultralytics roda
        em PyTorch (MPS/CUDA/CPU) mesmo quando a preferencia resolvida se chama
        "coreml"; so um ``.mlpackage`` roda de fato no CoreML/Neural Engine.
        """
        suffix = self.weights.suffix.lower()
        if self.runtime == "onnx":
            providers: list[str]
            try:
                providers = list(self._session.get_providers()) if self._session else []
            except Exception:  # noqa: BLE001
                providers = []
            runtime, device = "onnxruntime", ",".join(providers) or "?"
        elif self.runtime == "coreml":
            runtime, device = "coremltools", "coreml"
        elif suffix == ".mlpackage" or suffix == ".mlmodel":
            runtime, device = "ultralytics/coreml", "coreml"
        else:
            runtime, device = "ultralytics/pytorch", self.device.torch_device
        info = {
            "name": self.name, "runtime": runtime, "device": device,
            "weights": str(self.weights), "imgsz": self.imgsz, "conf": self.conf,
            "iou": self.iou, "classes": sorted(self.classes) if self.classes else None,
        }
        if self.runtime == "onnx":
            info["input_shape"] = [1, 3, self._onnx_size[1], self._onnx_size[0]]
            info["input_dtype"] = str(self._onnx_dtype)
        if self._coreml_info is not None:
            # Com NMS embutido, o iou efetivo e o da exportacao, nao o configurado.
            info["coreml_export"] = dict(self._coreml_info)
            if self._coreml_info["nms_embedded"]:
                info["iou"] = self._coreml_info["iou_embedded"]
        try:
            info["weights_sha256"] = file_sha256(str(self.weights))
        except OSError:
            info["weights_sha256"] = "?"
        return info

    def close(self) -> None:
        self._model = None
        self._session = None
        self._ml = None


def build_vehicle_detector(
    weights: str | Path, device: str | DeviceInfo = "auto", **kwargs
) -> YOLODetector:
    """Detector de veiculos com as classes COCO relevantes ja filtradas."""
    kwargs.setdefault("classes", COCO_VEHICLES)
    kwargs.setdefault("name", "vehicle")
    return YOLODetector(weights, device=device, **kwargs)


def build_object_detector(
    weights: str | Path,
    device: str | DeviceInfo = "auto",
    preset: str = "trafego",
    **kwargs,
) -> YOLODetector:
    """Detector geral de objetos, restrito ao preset de classes escolhido.

    Filtrar as classes na propria inferencia (em vez de descartar depois) evita
    NMS e pos-processamento sobre deteccoes que serao jogadas fora - a diferenca
    e mensuravel em cenas urbanas cheias.
    """
    from ..core.taxonomy import preset as preset_classes

    kwargs.setdefault("classes", preset_classes(preset))
    kwargs.setdefault("name", f"objetos:{preset}")
    return YOLODetector(weights, device=device, **kwargs)


def build_plate_detector(
    weights: str | Path, device: str | DeviceInfo = "auto", **kwargs
) -> YOLODetector:
    """Detector de placas (modelo de classe unica, treinado ou baixado)."""
    kwargs.setdefault("classes", {0: "plate"})
    kwargs.setdefault("name", "plate")
    kwargs.setdefault("imgsz", 640)
    return YOLODetector(weights, device=device, **kwargs)
