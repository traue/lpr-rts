"""Modelos e distribuicao: falhas exercitadas sem baixar nem carregar modelos neurais."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lprts.config import AppConfig
from lprts.detect.backends import DeviceInfo
from lprts.detect.base import file_sha256
from lprts.detect.yolo_detector import YOLODetector, coreml_export_info
from lprts.factory import build_ocr
from lprts.ocr import engines
from lprts.ocr.template_ocr import TemplateOCR


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("limit", [0, 2])
@pytest.mark.parametrize("text", ["1231231", "0000000"])
def test_gabaritos_nao_inventam_placa_alem_do_limite_de_correcoes(plate_image, limit, text):
    engine = TemplateOCR()
    engine.max_corrections = limit
    read = engine.read(plate_image(text))
    assert not read.valid_syntax
    assert read.n_corrections <= limit


@pytest.mark.parametrize("fallback", ["paddle", "easyocr"])
def test_auto_passa_so_as_opcoes_do_motor_escolhido(monkeypatch, fallback):
    def unavailable(**kwargs):
        raise ImportError("not installed")

    class Paddle:
        name = "paddle"

        def __init__(self, lang="en", use_gpu=False):
            pass

    class Easy:
        name = "easyocr"

        def __init__(self, device="auto"):
            pass

    monkeypatch.setitem(engines._ENGINES, "fastplate", unavailable)
    monkeypatch.setitem(engines._ENGINES, "paddle", Paddle if fallback == "paddle" else unavailable)
    monkeypatch.setitem(engines._ENGINES, "easyocr", Easy)
    assert build_ocr(AppConfig()).name == fallback


@pytest.mark.parametrize("package", [False, True])
def test_hash_dos_pesos_muda_quando_o_arquivo_e_trocado(tmp_path, package):
    path = tmp_path / "weights"
    if package:
        path.mkdir()
        content = path / "model.bin"
    else:
        content = path
    content.write_bytes(b"v1")
    first = file_sha256(str(path))
    content.write_bytes(b"v2")
    actual = hashlib.sha256((b"model.bin" if package else b"") + b"v2").hexdigest()
    assert file_sha256(str(path)) == actual
    assert actual != first


def test_metadados_coreml_sao_relidos_apos_trocar_o_modelo(monkeypatch, tmp_path):
    model = tmp_path / "model.mlpackage"
    model.mkdir()
    size = model / "size"
    size.write_text("320")

    class FakeModel:
        def __init__(self, weights, **kwargs):
            value = int((Path(weights) / "size").read_text())
            self.user_defined_metadata = {"imgsz": str([value, value])}

    monkeypatch.setitem(sys.modules, "coremltools", SimpleNamespace(models=SimpleNamespace(MLModel=FakeModel)))
    assert coreml_export_info(str(model))["imgsz"] == [320, 320]
    size.write_text("640")
    assert coreml_export_info(str(model))["imgsz"] == [640, 640]


def _onnx_runtime(monkeypatch, shape, dtype):
    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=shape, type=dtype)]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, outputs, inputs):
            calls.append(inputs["images"])
            return [np.zeros((1, 5, 20), dtype=np.float32)]

    ort = SimpleNamespace(
        SessionOptions=SimpleNamespace,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0), InferenceSession=Session,
        disable_telemetry_events=lambda: None, set_default_logger_severity=lambda n: None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return calls


@pytest.mark.parametrize("shape,dtype,expected_shape,expected_dtype", [
    ([1, 3, 640, 640], "tensor(float)", (1, 3, 640, 640), np.float32),
    ([1, 3, 320, 640], "tensor(float16)", (1, 3, 320, 640), np.float16),
    (["batch", 3, "height", "width"], "tensor(float)", (1, 3, 128, 128), np.float32),
])
def test_preprocessamento_onnx_respeita_a_entrada_declarada(
    monkeypatch, tmp_path, shape, dtype, expected_shape, expected_dtype,
):
    calls = _onnx_runtime(monkeypatch, shape, dtype)
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")
    cpu = DeviceInfo("cpu", "cpu", ("CPUExecutionProvider",), "test")
    detector = YOLODetector(weights, imgsz=128, device=cpu)
    assert detector.detect(np.full((80, 160, 3), 255, dtype=np.uint8)) == []
    assert calls[0].shape == expected_shape
    assert calls[0].dtype == expected_dtype
    assert calls[0].max() == 1


@pytest.mark.parametrize("shape,dtype", [
    ([2, 3, 640, 640], "tensor(float)"),
    ([1, 640, 640, 3], "tensor(float)"),
    ([1, 3, 640, 640], "tensor(uint8)"),
])
def test_entrada_onnx_nao_suportada_falha_na_partida(monkeypatch, tmp_path, shape, dtype):
    _onnx_runtime(monkeypatch, shape, dtype)
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")
    cpu = DeviceInfo("cpu", "cpu", ("CPUExecutionProvider",), "test")
    with pytest.raises(ValueError, match="ONNX"):
        YOLODetector(weights, device=cpu)


def test_download_que_falha_retorna_erro(monkeypatch):
    script = _script("download_models")
    monkeypatch.setattr(sys, "argv", ["download_models", "--plate", "morsetech-n", "--onnx"])
    monkeypatch.setattr(script, "download_vehicle_model", lambda *a, **kw: None)
    monkeypatch.setattr(script, "download_plate_model", lambda *a, **kw: None)
    assert script.main() != 0


def test_preparo_so_tem_sucesso_com_os_artefatos(monkeypatch, tmp_path):
    script = _script("download_models")
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    exported = weights.with_suffix(".onnx")
    exported.write_bytes(b"export")
    monkeypatch.setattr(sys, "argv", ["download_models", "--onnx"])
    monkeypatch.setattr(script, "PLATE_TARGET", tmp_path / "absent.pt")
    monkeypatch.setattr(script, "download_vehicle_model", lambda *a: weights)
    monkeypatch.setattr(script, "export_onnx", lambda *a: exported)
    assert script.main() == 0


@pytest.mark.parametrize("result", [None, "missing.onnx"])
def test_download_confere_o_artefato_exportado_pedido(monkeypatch, tmp_path, result):
    script = _script("download_models")
    weights = tmp_path / "vehicle.pt"
    weights.write_bytes(b"valid weights")
    monkeypatch.setattr(sys, "argv", ["download_models", "--onnx"])
    monkeypatch.setattr(script, "PLATE_TARGET", tmp_path / "missing-plate.pt")
    monkeypatch.setattr(script, "download_vehicle_model", lambda *a: weights)
    monkeypatch.setattr(script, "export_onnx", lambda *a: None if result is None else tmp_path / result)
    assert script.main() != 0


@pytest.mark.parametrize("format", ["onnx", "coreml"])
def test_exportacao_que_falha_preserva_o_artefato_existente(monkeypatch, tmp_path, format):
    script = _script("download_models")
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"current weights")
    suffix = ".onnx" if format == "onnx" else ".mlpackage"
    target = weights.with_suffix(suffix)
    if format == "coreml":
        target.mkdir()
        content = target / "model.bin"
    else:
        content = target
    content.write_bytes(b"previous export")

    class FakeYOLO:
        def __init__(self, path):
            self.weights = Path(path)

        def export(self, **kwargs):
            artifact = self.weights.with_suffix(suffix)
            if format == "coreml":
                artifact.mkdir(exist_ok=True)
                artifact /= "model.bin"
            artifact.write_bytes(b"partial broken export")
            raise RuntimeError("conversion failed")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(script, "_is_apple_silicon", lambda: True)
    assert getattr(script, f"export_{format}")(weights) is None
    assert content.read_bytes() == b"previous export"


@pytest.mark.parametrize("failure", ["missing_best", "empty_best", "export_error", "missing_export"])
def test_treino_que_falha_retorna_erro_e_preserva_os_modelos(monkeypatch, tmp_path, failure):
    script = _script("train_plate_detector")
    data = tmp_path / "dataset.yaml"
    data.write_text("names: {0: plate}")
    out = tmp_path / "model.pt"
    out.write_bytes(b"previous model")
    exported = out.with_suffix(".mlpackage")
    exported.mkdir()
    (exported / "model.bin").write_bytes(b"previous export")
    if failure != "missing_best":
        best = tmp_path / "weights" / "best.pt"
        best.parent.mkdir()
        best.write_bytes(b"" if failure == "empty_best" else b"new model")

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, **kwargs):
            return SimpleNamespace(save_dir=tmp_path)

        def export(self, **kwargs):
            if failure == "export_error":
                raise RuntimeError("export failed")
            return tmp_path / "not-created.mlpackage"

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(sys, "argv", ["train_plate_detector", "--data", str(data),
                                     "--out", str(out), "--device", "cpu", "--export-coreml"])
    assert script.main() != 0
    assert out.read_bytes() == b"previous model"
    assert (exported / "model.bin").read_bytes() == b"previous export"


@pytest.mark.parametrize("coreml", [False, True])
def test_treino_publica_so_artefatos_completos(monkeypatch, tmp_path, coreml):
    script = _script("train_plate_detector")
    data = tmp_path / "data.yaml"
    data.write_text("names: {0: plate}")
    best = tmp_path / "weights" / "best.pt"
    best.parent.mkdir()
    best.write_bytes(b"trained weights")
    out = tmp_path / "model.pt"
    out.write_bytes(b"previous weights")
    out.with_suffix(".onnx").write_bytes(b"obsolete export")
    exported = out.with_suffix(".mlpackage")
    exported.mkdir()
    (exported / "model.bin").write_bytes(b"obsolete export")

    class FakeYOLO:
        def __init__(self, path):
            self.weights = Path(path)

        def train(self, **kwargs):
            return SimpleNamespace(save_dir=tmp_path)

        def export(self, **kwargs):
            target = self.weights.with_suffix(".mlpackage")
            target.mkdir()
            (target / "model.bin").write_bytes(b"new export")
            return target

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    args = ["train_plate_detector", "--data", str(data), "--out", str(out), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", args + (["--export-coreml"] if coreml else []))
    assert script.main() == 0
    assert out.read_bytes() == b"trained weights"
    assert not out.with_suffix(".onnx").exists()
    if coreml:
        assert (exported / "model.bin").read_bytes() == b"new export"
    else:
        assert not exported.exists()
