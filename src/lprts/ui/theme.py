"""Folha de estilo da interface (tema escuro, adequado a sala de monitoramento)."""

from __future__ import annotations

DARK_QSS = """
QWidget { background-color: #0f1216; color: #e2e8f0;
          font-family: -apple-system, "SF Pro Text", "Segoe UI", sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #232a33; border-radius: 8px; margin-top: 14px; padding-top: 10px;
            font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #94a3b8; }
QPushButton { background-color: #1c232c; border: 1px solid #2b3541; border-radius: 6px;
              padding: 7px 14px; font-weight: 600; }
QPushButton:hover:!disabled { background-color: #24303c; border-color: #3b82f6; }
QPushButton:pressed:!disabled { background-color: #16202a; }
QPushButton:disabled { color: #4b5563; border-color: #1c232c; }
QPushButton#primary { background-color: #2563eb; border-color: #2563eb; color: white; }
QPushButton#primary:hover:!disabled { background-color: #3b82f6; }
QPushButton#danger { background-color: #7f1d1d; border-color: #991b1b; }
QPushButton#danger:hover:!disabled { background-color: #991b1b; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #151b22; border: 1px solid #2b3541; border-radius: 6px; padding: 6px 8px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #3b82f6; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background-color: #151b22; selection-background-color: #2563eb;
                              border: 1px solid #2b3541; }
QTableWidget { background-color: #12171d; gridline-color: #1f2630; border: none;
               alternate-background-color: #151b22; }
QHeaderView::section { background-color: #1a212a; border: none; border-bottom: 1px solid #2b3541;
                       padding: 6px; font-weight: 600; color: #94a3b8; }
QTableWidget::item:selected { background-color: #1d4ed8; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #2b3541; border-radius: 4px;
                       background-color: #151b22; }
QCheckBox::indicator:checked { background-color: #2563eb; border-color: #2563eb; }
QStatusBar { background-color: #0b0e11; border-top: 1px solid #1f2630; color: #94a3b8; }
QSplitter::handle { background-color: #1f2630; }
QScrollBar:vertical { background: #0f1216; width: 10px; }
QScrollBar::handle:vertical { background: #2b3541; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QToolTip { background-color: #1a212a; color: #e2e8f0; border: 1px solid #2b3541; padding: 4px; }
"""

LIGHT_QSS = """
QWidget { background-color: #f8fafc; color: #0f172a;
          font-family: -apple-system, "SF Pro Text", "Segoe UI", sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 14px; padding-top: 10px;
            font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #64748b; }
QPushButton { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;
              padding: 7px 14px; font-weight: 600; }
QPushButton:hover:!disabled { border-color: #2563eb; }
QPushButton#primary { background-color: #2563eb; border-color: #2563eb; color: white; }
QPushButton#danger { background-color: #dc2626; border-color: #dc2626; color: white; }
QLineEdit, QComboBox, QSpinBox { background-color: #ffffff; border: 1px solid #cbd5e1;
                                 border-radius: 6px; padding: 6px 8px; }
QTableWidget { background-color: #ffffff; gridline-color: #e2e8f0; border: none;
               alternate-background-color: #f1f5f9; }
QHeaderView::section { background-color: #f1f5f9; border: none; border-bottom: 1px solid #cbd5e1;
                       padding: 6px; font-weight: 600; color: #475569; }
"""


def stylesheet(theme: str = "dark") -> str:
    return LIGHT_QSS if theme.lower() == "light" else DARK_QSS
