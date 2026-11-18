"""Widgets da interface: tabela de placas e painel de telemetria.

Rodam com QT_QPA_PLATFORM=offscreen; nao exigem display.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="interface grafica nao instalada")

from PySide6.QtWidgets import QApplication  # noqa: E402

from lprts.analytics.metrics import MetricsCollector  # noqa: E402
from lprts.ui.widgets import MetricsPanel, PlateTable, StatusStrip  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestPlateTable:
    def test_adiciona_placas(self, qapp):
        t = PlateTable()
        t.add_plate("ABC1D23", 0.90, 1, "carro")
        t.add_plate("XYZ9K88", 0.85, 2, "motocicleta")
        assert t.table.rowCount() == 2

    def test_mais_recente_no_topo(self, qapp):
        t = PlateTable()
        t.add_plate("ABC1D23", 0.9, 1, "carro")
        t.add_plate("XYZ9K88", 0.9, 2, "carro")
        assert t.table.item(0, 1).text() == "XYZ9K88"

    def test_agrupa_a_mesma_placa_de_trilhas_diferentes(self, qapp):
        """Regressao: troca de identidade duplicava a linha e parecia defeito."""
        t = PlateTable()
        t.add_plate("ABC1D23", 0.80, 1, "carro")
        t.add_plate("ABC1D23", 0.95, 7, "carro")     # mesma placa, outra trilha
        assert t.table.rowCount() == 1
        assert t.table.item(0, 3).text() == "2"       # contador de vezes

    def test_agrupamento_mantem_a_maior_confianca(self, qapp):
        t = PlateTable()
        t.add_plate("ABC1D23", 0.95, 1, "carro")
        t.add_plate("ABC1D23", 0.40, 2, "carro")
        assert t.table.item(0, 2).text() == "0.95"

    def test_placas_distintas_nao_sao_agrupadas(self, qapp):
        t = PlateTable()
        for i, p in enumerate(["ABC1D23", "XYZ9K88", "QWE5R66"]):
            t.add_plate(p, 0.9, i, "carro")
        assert t.table.rowCount() == 3

    def test_indice_acompanha_a_insercao_no_topo(self, qapp):
        """A linha guardada precisa seguir o deslocamento causado por novas linhas."""
        t = PlateTable()
        t.add_plate("AAA1A11", 0.9, 1, "carro")
        for p in ("BBB2B22", "CCC3C33", "DDD4D44"):
            t.add_plate(p, 0.9, 9, "carro")
        t.add_plate("AAA1A11", 0.9, 2, "carro")      # volta a primeira
        assert t.table.rowCount() == 4
        linha = next(
            r for r in range(4) if t.table.item(r, 1).text() == "AAA1A11"
        )
        assert t.table.item(linha, 3).text() == "2"

    def test_fora_da_janela_cria_linha_nova(self, qapp):
        """Passagens realmente distintas devem aparecer separadas."""
        t = PlateTable(merge_window_s=0.0)
        t.add_plate("ABC1D23", 0.9, 1, "carro")
        t.add_plate("ABC1D23", 0.9, 2, "carro")
        assert t.table.rowCount() == 2

    def test_limite_de_linhas(self, qapp):
        t = PlateTable(max_rows=5, merge_window_s=0.0)
        for i in range(20):
            t.add_plate(f"ABC{i % 10}D23", 0.9, i, "carro")
        assert t.table.rowCount() <= 5

    def test_limpar(self, qapp):
        t = PlateTable()
        t.add_plate("ABC1D23", 0.9, 1, "carro")
        t.clear()
        assert t.table.rowCount() == 0
        t.add_plate("ABC1D23", 0.9, 2, "carro")      # nao deve agrupar com o antigo
        assert t.table.rowCount() == 1


class TestMetricsPanel:
    def test_atualiza_sem_erro(self, qapp):
        m = MetricsCollector()
        for _ in range(10):
            m.on_frame(total_ms=15.0, detect_ms=12.0, ocr_ms=3.0, n_active_tracks=2)
        m.on_read(valid_syntax=True)
        m.on_event("ABC1D23")
        painel = MetricsPanel()
        painel.update_metrics(m.snapshot())
        assert painel._labels["plates"].text() == "1"


class TestStatusStrip:
    def test_estado_e_detalhe(self, qapp):
        s = StatusStrip()
        s.set_state("Executando", "#4ade80")
        s.set_detail("fonte=video.mp4")
        assert s.state.text() == "Executando"
        assert "video.mp4" in s.detail.text()


class TestSeletoresDeDisponibilidade:
    """Seletores da janela principal refletem o que existe na maquina.

    Regressao: o seletor "Placas" oferecia 'yolo' mesmo sem os pesos, e escolher
    essa opcao derrubava o pipeline no arranque com FileNotFoundError - a
    interface ficava com a tabela vazia e o usuario sem saber por que.
    """

    def test_opcao_indisponivel_fica_bloqueada(self, qapp):
        from PySide6.QtWidgets import QComboBox

        from lprts.ui.app import _preencher_combo, _valor_do_combo

        combo = QComboBox()
        _preencher_combo(
            combo, [("auto", True, ""), ("yolo", False, "sem pesos"), ("classic", True, "")],
            "auto",
        )
        assert combo.count() == 3
        assert "sem pesos" in combo.itemText(1)
        assert not combo.model().item(1).isEnabled()
        assert combo.model().item(0).isEnabled()
        assert _valor_do_combo(combo, "?") == "auto"

    def test_valor_lido_e_o_interno_nao_o_rotulo(self, qapp):
        """O rotulo ganha o motivo entre parenteses; o valor precisa ficar limpo."""
        from PySide6.QtWidgets import QComboBox

        from lprts.ui.app import _preencher_combo, _valor_do_combo

        combo = QComboBox()
        _preencher_combo(combo, [("yolo", False, "sem pesos"), ("classic", True, "")], "classic")
        combo.setCurrentIndex(0)
        assert _valor_do_combo(combo, "?") == "yolo"      # e nao "yolo (sem pesos)"

    def test_config_indisponivel_cai_para_opcao_valida(self, qapp):
        """Um YAML pedindo 'yolo' sem os pesos nao deve deixar a interface travada."""
        from PySide6.QtWidgets import QComboBox

        from lprts.ui.app import _preencher_combo, _valor_do_combo

        combo = QComboBox()
        _preencher_combo(
            combo, [("auto", True, ""), ("yolo", False, "sem pesos")], "yolo"
        )
        assert _valor_do_combo(combo, "?") == "auto"


class TestValidacaoAntesDeIniciar:
    def test_nao_inicia_com_yolo_sem_pesos(self, qapp, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        from lprts.config import AppConfig
        from lprts.ui.app import MainWindow

        monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
        cfg = AppConfig()
        cfg.detector.plate_weights = str(tmp_path / "nao_existe.pt")
        janela = MainWindow(cfg)
        janela.config.detector.plate_detector = "yolo"
        assert janela._validar_selecao() != ""

    def test_configuracao_valida_passa(self, qapp):
        from lprts.config import AppConfig
        from lprts.ui.app import MainWindow

        cfg = AppConfig()
        cfg.detector.plate_detector = "classic"
        assert MainWindow(cfg)._validar_selecao() == ""


class TestSourceSelector:
    """Seletor de fonte: camera detectada, arquivo ou stream."""

    def test_monta_com_arquivo_e_stream(self, qapp):
        from lprts.ui.widgets import SourceSelector

        s = SourceSelector("video.mp4")
        tipos = {
            s.combo.itemData(i).kind
            for i in range(s.combo.count())
            if s.combo.itemData(i) is not None
        }
        assert {"arquivo", "stream"} <= tipos

    def test_preserva_a_uri_inicial(self, qapp):
        from lprts.ui.widgets import SourceSelector

        assert SourceSelector("data/videos/x.mp4").uri() == "data/videos/x.mp4"

    def test_set_uri_atualiza_o_campo(self, qapp):
        from lprts.ui.widgets import SourceSelector

        s = SourceSelector("0")
        s.set_uri("rtsp://cam/stream")
        assert s.uri() == "rtsp://cam/stream"

    def test_emite_mudanca_de_fonte(self, qapp):
        from lprts.ui.widgets import SourceSelector

        s = SourceSelector("0")
        recebidas = []
        s.source_changed.connect(recebidas.append)
        s.campo.setText("video.mp4")
        assert "video.mp4" in recebidas

    def test_travar_desabilita_todos_os_controles(self, qapp):
        """Durante a execucao a fonte nao pode mudar sob o pipeline."""
        from lprts.ui.widgets import SourceSelector

        s = SourceSelector("0")
        s.set_enabled(False)
        assert not s.combo.isEnabled()
        assert not s.campo.isEnabled()
        assert not s.botao_abrir.isEnabled()
        s.set_enabled(True)
        assert s.combo.isEnabled()

    def test_rescan_preserva_a_fonte_atual(self, qapp):
        from lprts.ui.widgets import SourceSelector

        s = SourceSelector("data/videos/x.mp4")
        s.rescan(silencioso=True)
        assert s.uri() == "data/videos/x.mp4"

    def test_sem_camera_mostra_item_desabilitado(self, qapp):
        """Sem cameras, o usuario precisa saber disso - nao ver um menu vazio."""
        from lprts.ui.widgets import SourceSelector

        s = SourceSelector("video.mp4")
        n_cameras = sum(
            1 for i in range(s.combo.count())
            if (d := s.combo.itemData(i)) is not None and d.kind == "camera"
        )
        if n_cameras == 0:
            assert "Nenhuma camera" in s.combo.itemText(0)
            assert not s.combo.model().item(0).isEnabled()
