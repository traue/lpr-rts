"""Descoberta de fontes de video."""

from __future__ import annotations

import pytest

from lprts.capture.devices import (
    VideoDevice,
    classify_source,
    describe_sources,
    is_stream_uri,
    list_cameras,
)


class TestClassificacao:
    @pytest.mark.parametrize(
        "uri,esperado",
        [
            ("0", "camera"),
            ("1", "camera"),
            ("rtsp://cam/stream", "stream"),
            ("http://host/video.mjpg", "stream"),
            ("udp://239.0.0.1:1234", "stream"),
            ("video.mp4", "arquivo"),
            ("/caminho/absoluto/x.mov", "arquivo"),
            ("data/videos/sintetico.mp4", "arquivo"),
        ],
    )
    def test_classifica(self, uri, esperado):
        assert classify_source(uri) == esperado

    def test_reconhece_streams(self):
        assert is_stream_uri("RTSP://CAM/STREAM")      # sem depender de caixa
        assert not is_stream_uri("arquivo.mp4")


class TestVideoDevice:
    def test_uri_de_camera_e_o_indice(self):
        assert VideoDevice(index=2, name="X").uri == "2"

    def test_rotulo_inclui_indice_e_nome(self):
        rotulo = VideoDevice(index=0, name="FaceTime HD").label
        assert "Camera 0" in rotulo and "FaceTime HD" in rotulo

    def test_detalhe_entra_no_rotulo(self):
        assert "1920x1080" in VideoDevice(index=0, name="X", detail="1920x1080").label

    def test_arquivo_usa_o_caminho(self):
        d = VideoDevice(index=-1, name="Arquivo...", kind="arquivo", path="/x.mp4")
        assert d.uri == "/x.mp4"


class TestEnumeracao:
    def test_listar_nao_abre_dispositivos(self):
        """Abrir a camera so para listar dispararia o pedido de permissao no macOS."""
        dispositivos = list_cameras(probe=False)
        assert isinstance(dispositivos, list)
        assert all(isinstance(d, VideoDevice) for d in dispositivos)

    def test_fontes_incluem_arquivo_e_stream(self):
        tipos = {d.kind for d in describe_sources()}
        assert "arquivo" in tipos and "stream" in tipos

    def test_cameras_vem_antes_das_entradas_genericas(self):
        fontes = describe_sources()
        assert fontes[-1].kind == "stream"
        assert fontes[-2].kind == "arquivo"
