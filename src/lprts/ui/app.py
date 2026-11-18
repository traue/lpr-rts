"""Janela principal da interface grafica (PySide6/Qt6).

Layout em tres regioes, na ordem em que o operador olha:

    +--------------------------------------------------+
    |  barra de controles (fonte, motores, iniciar)     |
    +-------------------------------+------------------+
    |                               |  telemetria      |
    |   video anotado               +------------------+
    |                               |  placas lidas    |
    +-------------------------------+------------------+
    |  barra de estado                                  |
    +--------------------------------------------------+

O pipeline nunca roda na thread da interface (ver ``worker.PipelineWorker``):
uma inferencia de 30 ms no laco de eventos congelaria a janela.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..core.types import FrameResult
from ..ocr.engines import available_engines
from ..utils.logging import get_logger
from .overlay import draw_overlay
from .theme import stylesheet
from .widgets import MetricsPanel, PlateTable, SourceSelector, StatusStrip, VideoView
from .worker import PipelineWorker

log = get_logger(__name__)


def _preencher_combo(
    combo: QComboBox, opcoes: list[tuple[str, bool, str]], atual: str
) -> None:
    """Preenche um seletor marcando as opcoes indisponiveis.

    Uma opcao que so pode falhar nao deve ser oferecida como se fosse valida.
    Ela continua visivel - para que se saiba que existe e o que falta -, porem
    desabilitada e com o motivo no rotulo e na dica.

    ``opcoes`` sao triplas ``(valor, disponivel, motivo)``.
    """
    combo.clear()
    for i, (valor, disponivel, motivo) in enumerate(opcoes):
        rotulo = valor if disponivel else f"{valor} ({motivo})"
        combo.addItem(rotulo, userData=valor)
        if not disponivel:
            item = cast(QStandardItemModel, combo.model()).item(i)
            if item is not None:
                item.setEnabled(False)
            combo.setItemData(i, f"Indisponivel: {motivo}", Qt.ItemDataRole.ToolTipRole)

    if _selecionar(combo, atual):
        return
    # A opcao configurada esta indisponivel: cai na primeira que funciona.
    for i in range(combo.count()):
        item = cast(QStandardItemModel, combo.model()).item(i)
        if item is None or item.isEnabled():
            combo.setCurrentIndex(i)
            return


def _selecionar(combo: QComboBox, valor: str) -> bool:
    """Seleciona a opcao de valor ``valor``, se existir e estiver habilitada."""
    for i in range(combo.count()):
        if combo.itemData(i) == valor:
            item = cast(QStandardItemModel, combo.model()).item(i)
            if item is None or item.isEnabled():
                combo.setCurrentIndex(i)
                return True
            return False
    return False


def deteccao_de_veiculos_disponivel(config: AppConfig) -> tuple[bool, str]:
    """O detector de objetos (YOLO) pode rodar? Devolve ``(disponivel, motivo)``."""
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return False, "ultralytics nao instalado: pip install '.[dl]'"
    from ..factory import resolve_weights

    if not resolve_weights(config.detector.vehicle_weights).exists():
        return False, "sem pesos: python scripts/download_models.py --coreml"
    return True, ""


def padroes_para_fonte(uri: str) -> tuple[bool, str]:
    """``(detectar veiculos, detector de placas)`` adequados a uma fonte.

    Video sintetico: os veiculos sao formas desenhadas, que o YOLO nao
    reconhece - modo so-placa, com o detector classico (sem veiculos, o YOLO
    de placas roda no quadro inteiro a 320 px e perde as placas pequenas).
    Video real, camera ou stream: YOLO de veiculos e detector de placas
    automatico (o YOLO de placas, se instalado).
    """
    from ..capture.devices import is_synthetic_video

    if is_synthetic_video(uri):
        return False, "classic"
    return True, "auto"


def _valor_do_combo(combo: QComboBox, padrao: str) -> str:
    """Le o valor real (userData), nao o rotulo exibido."""
    dado = combo.currentData()
    return str(dado) if dado is not None else padrao


class MainWindow(QMainWindow):
    """Janela principal do LPR-RTS."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.worker: PipelineWorker | None = None
        self._latest: FrameResult | None = None
        # Sessao gravada pela ultima execucao nesta janela: e o que o Exportar
        # leva (com "Limpar anteriores" desmarcado, o banco acumula os videos).
        self._ultima_sessao: str | None = None
        self._detalhe_execucao = ""

        self.setWindowTitle(config.ui.window_title)
        self.resize(1440, 860)

        self._build_ui()
        self._build_menu()

        # A interface repinta em cadencia propria (~30 Hz), desacoplada do
        # pipeline: um pico de latencia na inferencia nao trava o desenho, e um
        # pipeline muito rapido nao inunda o laco de eventos do Qt.
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_latest)
        self._render_timer.setInterval(33)

    # -- construcao --------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        # Duas linhas: a fonte tem seletor, campo e botoes proprios e nao cabe
        # junto das opcoes sem espremer tudo.
        root.addWidget(self._build_source_bar())
        root.addLayout(self._build_controls())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.video = VideoView()
        splitter.addWidget(self.video)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_panel = MetricsPanel()
        self.plate_table = PlateTable()
        side_layout.addWidget(self.metrics_panel)
        side_layout.addWidget(self.plate_table, stretch=1)
        splitter.addWidget(side)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1000, 420])
        root.addWidget(splitter, stretch=1)

        self.setCentralWidget(central)

        self.status_strip = StatusStrip()
        self.statusBar().addPermanentWidget(self.status_strip, 1)
        self.status_strip.set_state("Ocioso")

    def _build_source_bar(self) -> QWidget:
        """Linha dedicada a fonte: cameras detectadas, arquivo ou stream."""
        from ..capture.devices import is_synthetic_video

        self.source_selector = SourceSelector(str(self.config.source.uri))
        self.source_selector.realtime_suggested.connect(self._on_realtime_sugerido)
        self._fonte_sintetica = is_synthetic_video(str(self.config.source.uri))
        self.source_selector.source_changed.connect(self._on_fonte_alterada)
        return self.source_selector

    def _on_fonte_alterada(self, uri: str) -> None:
        """Ajusta os detectores quando a fonte passa de sintetica a real, ou o contrario.

        Antes, o ``./start.sh`` abria o video sintetico com
        ``--no-vehicle-detector`` e essa escolha valia para a interface inteira:
        um video real aberto depois rodava sem o YOLO. A escolha agora acompanha
        a fonte, e so muda quando a natureza dela muda - para nao desfazer um
        ajuste manual a cada tecla digitada no campo.
        """
        if not hasattr(self, "vehicle_check"):
            return                      # ainda montando a janela
        from ..capture.devices import is_synthetic_video

        sintetica = is_synthetic_video(uri)
        if sintetica == self._fonte_sintetica:
            return
        self._fonte_sintetica = sintetica
        veiculos, placas = padroes_para_fonte(uri)
        if self.vehicle_check.isEnabled():
            self.vehicle_check.setChecked(veiculos)
        _selecionar(self.plate_combo, placas)
        self.status_strip.set_detail(
            "Video sintetico: modo so-placa com o detector classico "
            "(o YOLO nao reconhece veiculos desenhados)."
            if sintetica else
            "Fonte real: veiculos pelo YOLO e placas pelo detector automatico."
        )

    def _on_realtime_sugerido(self, ligar: bool) -> None:
        """Camera e stream pedem tempo real; arquivo pede processar tudo.

        E sugestao, nao imposicao: a caixa permanece editavel, porque avaliar um
        arquivo exige o modo offline e assistir a uma camera exige o de tempo
        real - so o usuario sabe qual dos dois quer em cada momento.
        """
        self.realtime_check.setChecked(ligar)
        self.status_strip.set_detail(
            "Modo tempo real ligado (fonte ao vivo): quadros atrasados sao descartados."
            if ligar
            else "Modo offline (arquivo): todos os quadros sao processados."
        )

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("OCR:"))
        self.ocr_combo = QComboBox()
        instalados = set(available_engines())
        _preencher_combo(
            self.ocr_combo,
            [("auto", True, "")]
            + [
                (m, m in instalados, "nao instalado")
                for m in ("fastplate", "paddle", "easyocr", "tesseract", "template")
            ],
            self.config.ocr.engine,
        )
        bar.addWidget(self.ocr_combo)

        self.vehicle_check = QCheckBox("Veiculos (YOLO)")
        disponivel, motivo = deteccao_de_veiculos_disponivel(self.config)
        self.vehicle_check.setEnabled(disponivel)
        self.vehicle_check.setChecked(disponivel and self.config.detector.detect_vehicles)
        self.vehicle_check.setToolTip(
            "Detecta os veiculos com o YOLO e procura a placa dentro de cada um.\n"
            "Desmarcado: rastreia as placas direto no quadro (modo so-placa).\n"
            "Ajustado ao trocar de fonte: desmarcado para o video sintetico,\n"
            "marcado para videos reais e cameras."
            if disponivel else f"Indisponivel: {motivo}"
        )
        bar.addWidget(self.vehicle_check)

        # Os tipos de objeto vem do mesmo YOLO: so fazem sentido com ele ligado.
        bar.addWidget(QLabel("Objetos:"))
        self.classes_combo = QComboBox()
        _preencher_combo(
            self.classes_combo,
            [("trafego", True, ""), ("completo", True, ""), ("veiculos", True, "")],
            self.config.detector.classes,
        )
        self.classes_combo.setToolTip(
            "Tipos de objeto detectados, rastreados e gravados (tabela tracks):\n"
            "trafego: pessoas, bicicletas, carros, motos, onibus e caminhoes\n"
            "completo: trafego + semaforos, placas de pare, bancos, animais...\n"
            "veiculos: so os que portam placa (o mais leve)"
        )
        self.classes_combo.setEnabled(self.vehicle_check.isChecked())
        self.vehicle_check.toggled.connect(self.classes_combo.setEnabled)
        bar.addWidget(self.classes_combo)

        bar.addWidget(QLabel("Placas:"))
        self.plate_combo = QComboBox()
        self.plate_combo.setToolTip(
            "yolo exige um modelo em data/models/plate_yolo.pt.\n"
            "Para testar: python scripts/download_models.py --plate morsetech-n\n"
            "Nao existe um peso oficial universal - veja a secao 9 do README."
        )
        from ..factory import resolve_weights

        _preencher_combo(
            self.plate_combo,
            [
                ("auto", True, ""),
                ("yolo", resolve_weights(self.config.detector.plate_weights).exists(),
                 "sem pesos"),
                ("classic", True, ""),
            ],
            self.config.detector.plate_detector,
        )
        bar.addWidget(self.plate_combo)

        bar.addWidget(QLabel("Dispositivo:"))
        self.device_combo = QComboBox()
        from ..detect.backends import is_available

        # is_available e silencioso; resolve_device avisaria a cada opcao
        # indisponivel, poluindo o log so por montar um menu.
        disponiveis = {nome: is_available(nome) for nome in ("coreml", "mps", "cuda")}
        _preencher_combo(
            self.device_combo,
            [("auto", True, ""), ("cpu", True, "")]
            + [(n, ok, "nao disponivel") for n, ok in disponiveis.items()],
            self.config.detector.device,
        )
        bar.addWidget(self.device_combo)

        self.realtime_check = QCheckBox("Tempo real")
        self.realtime_check.setChecked(self.config.source.realtime)
        self.realtime_check.setToolTip(
            "Descarta quadros atrasados para manter a latencia baixa.\n"
            "Desmarque para avaliacao offline, em que nenhum quadro pode ser perdido."
        )
        bar.addWidget(self.realtime_check)

        self.clear_check = QCheckBox("Limpar anteriores")
        self.clear_check.setChecked(self.config.analytics.clear_previous)
        self.clear_check.setToolTip(
            f"Ao iniciar, apaga o banco {self.config.analytics.db_path} e os recortes\n"
            "das execucoes anteriores: os dados descrevem so o video desta execucao.\n"
            "Desmarque para acumular as execucoes no mesmo banco."
        )
        bar.addWidget(self.clear_check)

        self.trajectory_check = QCheckBox("Trajetorias")
        self.trajectory_check.setChecked(self.config.ui.show_trajectories)
        bar.addWidget(self.trajectory_check)

        bar.addStretch(1)

        self.start_button = QPushButton("Iniciar")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start)
        bar.addWidget(self.start_button)

        self.pause_button = QPushButton("Pausar")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        bar.addWidget(self.pause_button)

        self.stop_button = QPushButton("Parar")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        bar.addWidget(self.stop_button)
        return bar

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&Arquivo")
        for title, shortcut, slot in (
            ("Abrir video...", QKeySequence.StandardKey.Open, self._browse),
            ("Procurar cameras", "Ctrl+D", lambda: self.source_selector.rescan()),
            ("Exportar dados do video analisado...", "Ctrl+E", self._export),
            ("Sair", QKeySequence.StandardKey.Quit, self.close),
        ):
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)

        exec_menu = self.menuBar().addMenu("&Execucao")
        for title, shortcut, slot in (
            ("Iniciar", "Ctrl+R", self.start),
            ("Pausar/Retomar", "Space", self.toggle_pause),
            ("Parar", "Ctrl+.", self.stop),
        ):
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            exec_menu.addAction(action)

        help_menu = self.menuBar().addMenu("A&juda")
        about = QAction("Sobre", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    # -- acoes -------------------------------------------------------------

    def _browse(self) -> None:
        self.source_selector.escolher_arquivo()

    def _export(self) -> None:
        """Exporta SO o video analisado (uma sessao), nao o banco inteiro.

        Antes saiam todas as sessoes do banco - todos os videos ja processados,
        misturados nas mesmas tabelas e com ``track_id`` repetidos.
        """
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, "Exportar",
                "Pare a execucao antes de exportar: a sessao ainda esta sendo gravada.",
            )
            return
        from ..analytics.export import export_analyzed_video, latest_session_id, session_source

        db = self.config.analytics.db_path
        try:
            sessao = self._ultima_sessao or latest_session_id(db)
            video = Path(session_source(db, sessao)).name or session_source(db, sessao)
        except Exception as exc:  # noqa: BLE001 - banco ausente, vazio ou de outra versao
            QMessageBox.information(self, "Exportar", f"Nada para exportar ainda.\n\n{exc}")
            return
        destino = QFileDialog.getExistingDirectory(
            self, f"Exportar os dados de {video} - pasta de destino", "runs"
        )
        if not destino:
            return
        try:
            _, pasta, arquivos = export_analyzed_video(db, destino, sessao)
            QMessageBox.information(
                self, "Exportacao concluida",
                f"Dados de {video} (sessao {sessao}):\n"
                f"{len(arquivos)} arquivos em\n{pasta}",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Falha na exportacao", str(exc))

    def _about(self) -> None:
        QMessageBox.about(
            self, "Sobre o LPR-RTS",
            "<h3>LPR-RTS</h3>"
            "<p>Reconhecimento de placas veiculares em tempo real com visao "
            "computacional e aprendizado de maquina.</p>"
            "<p>Prova de conceito para pesquisa de doutorado. "
            "Toda a inferencia e executada localmente (CPU/GPU/ANE).</p>",
        )

    def _apply_controls(self) -> None:
        self.config.source.uri = self.source_selector.uri() or "0"
        self.config.source.realtime = self.realtime_check.isChecked()
        self.config.ocr.engine = _valor_do_combo(self.ocr_combo, "auto")
        self.config.detector.plate_detector = _valor_do_combo(self.plate_combo, "auto")
        self.config.detector.detect_vehicles = self.vehicle_check.isChecked()
        self.config.detector.classes = _valor_do_combo(self.classes_combo, "trafego")
        self.config.detector.device = _valor_do_combo(self.device_combo, "auto")
        self.config.analytics.clear_previous = self.clear_check.isChecked()
        self.config.ui.show_trajectories = self.trajectory_check.isChecked()

    def _validar_selecao(self) -> str:
        """Verifica a configuracao antes de iniciar; devolve o problema, se houver.

        Desabilitar o item no seletor impede o clique, mas nao uma configuracao
        vinda de arquivo nem uma selecao programatica. Esta e a rede de seguranca
        que garante que a interface nunca inicie um pipeline fadado a falhar.
        """
        from ..factory import resolve_weights

        if self.config.detector.plate_detector == "yolo" and not resolve_weights(
            self.config.detector.plate_weights
        ).exists():
            return (
                f"O detector de placas 'yolo' exige {self.config.detector.plate_weights}, "
                "que nao existe. Use 'classic' ou 'auto'."
            )
        return ""

    def start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self._apply_controls()

        problema = self._validar_selecao()
        if problema:
            self.status_strip.set_state("Configuracao invalida", "#f87171")
            self.status_strip.set_detail(problema)
            QMessageBox.warning(self, "Configuracao invalida", problema)
            return

        self.plate_table.clear()

        self.worker = PipelineWorker(self.config)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.plate_found.connect(self.plate_table.add_plate)
        self.worker.metrics_ready.connect(self.metrics_panel.update_metrics)
        self.worker.status.connect(lambda s: self.status_strip.set_state(s, "#4ade80"))
        self.worker.warning.connect(self._on_warning)
        self.worker.detail.connect(
            lambda texto: self.status_strip.set_detail(f"{texto} | {self._detalhe_execucao}")
        )
        self.worker.failed.connect(self._on_failed)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.start()

        self._render_timer.start()
        self.source_selector.set_enabled(False)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self._detalhe_execucao = (
            f"fonte={self.config.source.uri} | ocr={self.config.ocr.engine} | "
            f"veiculos={'yolo' if self.config.detector.detect_vehicles else 'nao'} | "
            f"objetos={self.config.detector.classes} | "
            f"placas={self.config.detector.plate_detector} | "
            f"dispositivo={self.config.detector.device} | "
            f"banco={self.config.analytics.db_path}"
        )
        self.status_strip.set_detail(self._detalhe_execucao)

    def toggle_pause(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        paused = self.worker.toggle_pause()
        self.pause_button.setText("Retomar" if paused else "Pausar")
        self.status_strip.set_state("Pausado" if paused else "Executando",
                                    "#fbbf24" if paused else "#4ade80")

    def stop(self) -> bool:
        if self.worker is not None:
            self.worker.stop()
            # O encerramento espera o OCR em voo (ate 15 s) e grava as trilhas
            # finais; com uma espera mais curta a janela fechava com a thread
            # ainda viva - perda de dados e, no macOS, queda do processo.
            if not self.worker.wait(30000):
                log.warning("A thread do pipeline nao terminou em 30 s.")
                return False
        return True

    # -- sinais ------------------------------------------------------------

    def _on_frame(self, result: FrameResult) -> None:
        # Apenas guarda a referencia: o desenho acontece no timer, para nao
        # gastar tempo de interface uma vez por quadro do pipeline.
        self._latest = result

    def _render_latest(self) -> None:
        result = self._latest
        if result is None:
            return
        self._latest = None
        metrics = None
        lines = None
        if self.worker is not None and self.worker.pipeline is not None:
            m = self.worker.pipeline.metrics.snapshot()
            metrics = {
                "fps": m.fps, "p95": m.latency_e2e.p95,
                "unique": m.n_unique_plates, "dropped": m.frames_dropped,
            }
            lines = self.worker.pipeline.traffic.lines
        image = draw_overlay(
            result.frame.image.copy(), result, self.config.ui, metrics, lines,
            line_scale=result.frame.scale,
        )
        self.video.show_frame(image)

    def _on_warning(self, message: str) -> None:
        """Aviso que muda a interpretacao do resultado: fica visivel e persiste."""
        self.status_strip.set_state("Executando (modo so-placa)", "#fbbf24")
        self.status_strip.set_detail(f"AVISO: {message}")

    def _on_failed(self, message: str) -> None:
        """Falha ao iniciar ou executar: precisa ficar visivel depois do dialogo.

        Um QMessageBox se fecha e some. Se o unico registro do erro for esse
        dialogo, o que resta na tela e uma tabela vazia - indistinguivel de um
        video sem placas. Por isso o motivo tambem fica na barra de estado.
        """
        self.status_strip.set_state("Erro - o pipeline nao executou", "#f87171")
        self.status_strip.set_detail(message.replace("\n", "  "))

        sugestao = ""
        if "outra versao do LPR-RTS" in message:
            sugestao = (
                "\n\nPara arquivar os bancos antigos e seguir (nada e apagado):\n"
                "  mkdir -p runs/legado && mv runs/lprts.db* runs/legado/\n\n"
                "O ./start.sh faz isso sozinho."
            )
        elif "plate_yolo" in message or "detector de placas" in message:
            sugestao = (
                "\n\nO detector de placas 'yolo' exige um modelo treinado em\n"
                "data/models/plate_yolo.pt, que nao acompanha o projeto - nao existe\n"
                "um peso oficial universal para placas.\n\n"
                "Escolha 'classic' ou 'auto' no seletor Placas, ou treine o seu:\n"
                "  python scripts/train_plate_detector.py --data data/placas.yaml"
            )
        elif "yolo11n" in message or "detector de objetos" in message:
            sugestao = (
                "\n\nBaixe os pesos com:\n  python scripts/download_models.py\n"
                "ou desmarque a deteccao de objetos."
            )
        QMessageBox.critical(self, "Falha no pipeline", message + sugestao)

    def _on_finished(self) -> None:
        if self.worker is not None and self.worker.store is not None:
            self._ultima_sessao = self.worker.store.session_id
        self._render_timer.stop()
        self.source_selector.set_enabled(True)
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pausar")
        self.stop_button.setEnabled(False)
        self.status_strip.set_state("Parado", "#94a3b8")

    def closeEvent(self, event) -> None:  # noqa: N802 - assinatura do Qt
        if self.stop():
            event.accept()
        else:
            event.ignore()


def launch_gui(config: AppConfig) -> int:
    """Cria a aplicacao Qt e abre a janela principal."""
    app = QApplication.instance() or QApplication(sys.argv)
    if not isinstance(app, QApplication):
        raise RuntimeError("A interface exige QApplication, nao QCoreApplication")
    app.setApplicationName("LPR-RTS")
    app.setStyleSheet(stylesheet(config.ui.theme))

    window = MainWindow(config)
    window.show()

    # Auto-inicia quando a fonte ja foi informada e existe: evita um clique
    # redundante em `lprts run --source video.mp4`.
    uri = str(config.source.uri)
    if uri and (uri.isdigit() or Path(uri).exists() or "://" in uri):
        QTimer.singleShot(300, window.start)

    return app.exec()
