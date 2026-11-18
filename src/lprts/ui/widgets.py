"""Widgets reutilizaveis da interface: video, telemetria e tabela de placas."""

from __future__ import annotations

import time
from datetime import datetime
from typing import cast

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class VideoView(QLabel):
    """Exibe o quadro anotado, preservando a proporcao."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #101214; border-radius: 6px;")
        self.setText("Aguardando video...")
        self._pixmap: QPixmap | None = None

    def show_frame(self, image_bgr: np.ndarray) -> None:
        """Recebe um quadro BGR do OpenCV e o exibe."""
        if image_bgr is None or image_bgr.size == 0:
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # QImage nao copia o buffer: sem .copy() a imagem apontaria para memoria
        # que a thread do pipeline reaproveita no quadro seguinte.
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - assinatura do Qt
        super().resizeEvent(event)
        self._render()


def resumo_de_objetos(contagens: dict[str, int], maximo: int = 6) -> str:
    """Objetos unicos por tipo, do mais frequente ao menos: ``carro 12 · pessoa 3``."""
    if not contagens:
        return "-"
    ordem = sorted(contagens.items(), key=lambda kv: (-kv[1], kv[0]))
    partes = [f"{classe} {n}" for classe, n in ordem[:maximo]]
    if len(ordem) > maximo:
        partes.append(f"+{sum(n for _, n in ordem[maximo:])}")
    return " · ".join(partes)


class MetricsPanel(QGroupBox):
    """Telemetria em tempo real: vazao, latencia por estagio e rendimento."""

    FIELDS = (
        ("fps", "FPS"),
        ("latency", "Latencia ponta a ponta p50 / p95"),
        ("service", "Tempo de servico p50"),
        ("detect", "Deteccao p50"),
        ("ocr", "OCR p50"),
        ("tracks", "Trilhas ativas"),
        ("reads", "Leituras de OCR"),
        ("syntax", "Sintaxe valida"),
        ("plates", "Placas unicas"),
        ("objects", "Objetos (trilhas)"),
        ("vulnerable", "Usuarios vulneraveis"),
        ("dropped", "Quadros descartados"),
    )

    def __init__(self) -> None:
        super().__init__("Telemetria")
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._labels: dict[str, QLabel] = {}
        for key, caption in self.FIELDS:
            value = QLabel("-")
            value.setStyleSheet("font-family: monospace; font-weight: 600;")
            value.setWordWrap(key == "objects")
            self._labels[key] = value
            layout.addRow(f"{caption}:", value)

    def update_metrics(self, m) -> None:
        self._labels["fps"].setText(f"{m.fps:.1f}")
        e2e = getattr(m, "latency_e2e", m.latency)
        self._labels["latency"].setText(f"{e2e.p50:.1f} / {e2e.p95:.1f} ms")
        self._labels["service"].setText(f"{m.latency.p50:.1f} ms")
        self._labels["detect"].setText(f"{m.detect.p50:.1f} ms")
        self._labels["ocr"].setText(f"{m.ocr.p50:.1f} ms" if m.ocr.n else "-")
        self._labels["tracks"].setText(str(m.n_active_tracks))
        self._labels["reads"].setText(str(m.n_reads))
        self._labels["syntax"].setText(f"{m.syntax_valid_rate:.0%}")
        self._labels["plates"].setText(str(m.n_unique_plates))
        self._labels["objects"].setText(resumo_de_objetos(m.counts_by_class))
        self._labels["vulnerable"].setText(str(m.n_vulnerable))
        self._labels["dropped"].setText(str(m.frames_dropped))

        # Cor do FPS como semaforo: verde >= 24 (fluido), ambar >= 12, vermelho abaixo.
        color = "#4ade80" if m.fps >= 24 else ("#fbbf24" if m.fps >= 12 else "#f87171")
        self._labels["fps"].setStyleSheet(
            f"font-family: monospace; font-weight: 700; color: {color};"
        )


class PlateTable(QGroupBox):
    """Tabela das placas reconhecidas, mais recente no topo."""

    COLUMNS = ("Horario", "Placa", "Conf.", "Vezes", "Classe")

    def __init__(self, max_rows: int = 500, merge_window_s: float = 20.0) -> None:
        super().__init__("Placas reconhecidas")
        self.max_rows = max_rows
        # Uma mesma placa pode chegar por trilhas diferentes quando o rastreador
        # perde e readquire o veiculo. Como painel de monitoramento, repetir a
        # linha parece defeito; agrupamos as reaparicoes proximas e mostramos a
        # contagem. O historico completo, com uma linha por evento, continua no
        # banco - a interface resume, nao substitui o registro.
        self.merge_window_s = merge_window_s
        self._ultimas: dict[str, tuple[float, int]] = {}   # placa -> (quando, linha)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        # A tabela vive num painel estreito. Com larguras fixas as colunas
        # transbordam e aparece barra de rolagem horizontal - o usuario perde de
        # vista justamente a confianca. Aqui a coluna da placa absorve a folga e
        # as demais se ajustam ao conteudo.
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col in range(2, len(self.COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setWordWrap(False)
        layout.addWidget(self.table)

    def add_plate(self, text: str, confidence: float, track_id: int, object_class: str) -> None:
        agora = time.monotonic()
        anterior = self._ultimas.get(text)
        if anterior is not None and agora - anterior[0] <= self.merge_window_s:
            self._incrementar(text, confidence, agora)
            return

        self.table.insertRow(0)
        self._deslocar_indices()
        valores = (
            datetime.now().strftime("%H:%M:%S"),
            text,
            f"{confidence:.2f}",
            "1",
            object_class,
        )
        for col, valor in enumerate(valores):
            item = QTableWidgetItem(valor)
            if col == 1:
                fonte = item.font()
                fonte.setBold(True)
                fonte.setPointSize(fonte.pointSize() + 1)
                item.setFont(fonte)
            self.table.setItem(0, col, item)
        self._ultimas[text] = (agora, 0)

        # Limite de linhas: uma sessao de horas encheria a memoria da interface.
        while self.table.rowCount() > self.max_rows:
            self.table.removeRow(self.table.rowCount() - 1)
        self._esquecer_fora_da_tabela()

    def _deslocar_indices(self) -> None:
        """Uma linha inserida no topo empurra as demais uma posicao para baixo."""
        self._ultimas = {p: (t, i + 1) for p, (t, i) in self._ultimas.items()}

    def _incrementar(self, text: str, confidence: float, agora: float) -> None:
        """Atualiza a linha existente em vez de criar uma duplicata."""
        _, linha = self._ultimas[text]
        if linha >= self.table.rowCount():
            return
        item_n = self.table.item(linha, 3)
        vezes = int(item_n.text()) + 1 if item_n and item_n.text().isdigit() else 2
        self.table.setItem(linha, 3, QTableWidgetItem(str(vezes)))
        self.table.setItem(linha, 0, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        anterior_conf = self.table.item(linha, 2)
        try:
            melhor = max(float(anterior_conf.text()), confidence) if anterior_conf else confidence
        except ValueError:
            melhor = confidence
        self.table.setItem(linha, 2, QTableWidgetItem(f"{melhor:.2f}"))
        self._ultimas[text] = (agora, linha)

    def _esquecer_fora_da_tabela(self) -> None:
        self._ultimas = {
            p: v for p, v in self._ultimas.items() if v[1] < self.table.rowCount()
        }

    def clear(self) -> None:
        self.table.setRowCount(0)
        self._ultimas.clear()


class StatusStrip(QWidget):
    """Faixa com o estado corrente e o resumo da configuracao."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.state = QLabel("Ocioso")
        self.detail = QLabel("")
        self.detail.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.detail.setWordWrap(True)
        layout.addWidget(self.state)
        layout.addWidget(self.detail)

    def set_state(self, text: str, color: str = "#e2e8f0") -> None:
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {color}; font-weight: 600;")

    def set_detail(self, text: str) -> None:
        self.detail.setText(text)


class SourceSelector(QWidget):
    """Escolha da fonte de video: camera detectada, arquivo ou stream de rede.

    As cameras sao descobertas pelo catalogo do sistema, sem abrir nenhuma - no
    macOS, abrir uma dispararia o pedido de permissao apenas por a janela ter
    sido montada. O dispositivo so e aberto quando o usuario clica em "Testar"
    ou inicia a execucao, que sao os momentos em que o pedido faz sentido.
    """

    source_changed = Signal(str)          # nova URI
    realtime_suggested = Signal(bool)     # camera/stream sugerem modo tempo real

    def __init__(self, uri_inicial: str = "0") -> None:
        super().__init__()
        self._dispositivos: list = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Fonte:"))

        self.combo = QComboBox()
        self.combo.setMinimumWidth(210)
        self.combo.setToolTip(
            "Cameras da maquina, arquivo de video ou stream de rede.\n"
            "As cameras sao listadas pelo catalogo do sistema, sem serem abertas."
        )
        self.combo.activated.connect(self._on_escolha)
        layout.addWidget(self.combo, stretch=1)

        self.botao_atualizar = QPushButton("Atualizar")
        self.botao_atualizar.setToolTip("Procurar cameras de novo")
        self.botao_atualizar.setMinimumWidth(96)
        self.botao_atualizar.clicked.connect(self.rescan)
        layout.addWidget(self.botao_atualizar)

        self.campo = QLineEdit(uri_inicial)
        self.campo.setPlaceholderText("arquivo.mp4, 0 (webcam) ou rtsp://...")
        self.campo.setMinimumWidth(220)
        self.campo.textChanged.connect(self._on_texto)
        layout.addWidget(self.campo, stretch=2)

        self.botao_abrir = QPushButton("Abrir...")
        self.botao_abrir.setMinimumWidth(90)
        self.botao_abrir.clicked.connect(self.escolher_arquivo)
        layout.addWidget(self.botao_abrir)

        self.botao_testar = QPushButton("Testar")
        self.botao_testar.setMinimumWidth(84)
        self.botao_testar.setToolTip("Abrir a fonte e verificar se entrega imagem")
        self.botao_testar.clicked.connect(self.testar)
        layout.addWidget(self.botao_testar)

        self.rescan(silencioso=True)
        self.set_uri(uri_inicial)

    # -- descoberta --------------------------------------------------------

    def rescan(self, silencioso: bool = False) -> None:
        """Refaz a varredura de cameras, preservando a fonte atual."""
        from ..capture.devices import describe_sources

        atual = self.campo.text().strip()
        self._dispositivos = describe_sources()

        self.combo.blockSignals(True)
        self.combo.clear()
        for d in self._dispositivos:
            self.combo.addItem(d.label, userData=d)
        n_cameras = sum(1 for d in self._dispositivos if d.kind == "camera")
        if n_cameras == 0:
            self.combo.insertItem(0, "Nenhuma camera encontrada", userData=None)
            item = cast(QStandardItemModel, self.combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
        self.combo.blockSignals(False)

        if atual:
            self.set_uri(atual)
        if not silencioso:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(
                self, "Varredura concluida",
                f"{n_cameras} camera(s) encontrada(s)."
                + ("" if n_cameras else "\n\nNo macOS, autorize o acesso em\n"
                   "Ajustes > Privacidade e Seguranca > Camera."),
            )

    # -- selecao -----------------------------------------------------------

    def _on_escolha(self, indice: int) -> None:
        dispositivo = self.combo.itemData(indice)
        if dispositivo is None:
            return
        if dispositivo.kind == "arquivo":
            self.escolher_arquivo()
            return
        if dispositivo.kind == "stream":
            self.campo.clear()
            self.campo.setPlaceholderText("rtsp://usuario:senha@192.168.1.50/stream")
            self.campo.setFocus()
            self.realtime_suggested.emit(True)
            return
        self.campo.setText(dispositivo.uri)
        self.realtime_suggested.emit(True)     # camera ao vivo pede tempo real

    def escolher_arquivo(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        caminho, _ = QFileDialog.getOpenFileName(
            self, "Selecionar video", "data/videos",
            "Videos (*.mp4 *.mov *.avi *.mkv *.m4v);;Todos os arquivos (*)",
        )
        if caminho:
            self.campo.setText(caminho)
            self.realtime_suggested.emit(False)   # arquivo: processar tudo

    def _on_texto(self, texto: str) -> None:
        self._sincronizar_combo(texto.strip())
        self.source_changed.emit(texto.strip())

    def _sincronizar_combo(self, uri: str) -> None:
        """Deixa o seletor coerente com o que esta escrito no campo."""
        from ..capture.devices import classify_source

        tipo = classify_source(uri)
        self.combo.blockSignals(True)
        for i in range(self.combo.count()):
            d = self.combo.itemData(i)
            if d is None:
                continue
            if d.kind == "camera" and tipo == "camera" and d.uri == uri:
                self.combo.setCurrentIndex(i)
                break
            if d.kind == tipo and tipo in ("arquivo", "stream"):
                self.combo.setCurrentIndex(i)
                break
        self.combo.blockSignals(False)

    # -- verificacao -------------------------------------------------------

    def testar(self) -> None:
        """Abre a fonte e informa o resultado. Acao explicita do usuario."""
        from PySide6.QtWidgets import QMessageBox

        from ..capture.devices import check_camera, classify_source

        uri = self.campo.text().strip()
        if not uri:
            QMessageBox.warning(self, "Fonte vazia", "Informe um arquivo, camera ou URL.")
            return

        tipo = classify_source(uri)
        if tipo == "camera":
            ok, mensagem = check_camera(int(uri))
        elif tipo == "arquivo":
            from pathlib import Path

            existe = Path(uri).exists()
            ok = existe
            mensagem = (
                f"Arquivo encontrado: {uri}" if existe else f"Arquivo nao encontrado: {uri}"
            )
        else:
            ok, mensagem = True, (
                "Streams de rede so podem ser verificados ao iniciar - "
                "a conexao pode levar alguns segundos."
            )

        if ok:
            QMessageBox.information(self, "Fonte verificada", mensagem)
        else:
            QMessageBox.warning(self, "Fonte indisponivel", mensagem)

    # -- acesso ------------------------------------------------------------

    def uri(self) -> str:
        return self.campo.text().strip()

    def set_uri(self, uri: str) -> None:
        if uri != self.campo.text():
            self.campo.setText(uri)
        self._sincronizar_combo(uri)

    def set_enabled(self, habilitado: bool) -> None:
        for w in (self.combo, self.campo, self.botao_abrir,
                  self.botao_atualizar, self.botao_testar):
            w.setEnabled(habilitado)
