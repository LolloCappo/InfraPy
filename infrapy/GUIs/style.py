# -*- coding: utf-8 -*-
"""
InfraPy GUI – style constants
------------------------------
Dark QSS stylesheet and shared UI constants.
"""

DOC_URL = "https://github.com/LolloCappo/InfraPy"
DEFAULT_COLORMAPS = ["gray", "viridis", "plasma", "inferno", "cividis"]
ROI_COLORS = ["r", "g", "b", "y", "c", "m", "w"]

DARK_QSS = """
QWidget {
    background-color: #1e1e1e;
    color: #d0d0d0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11px;
}
QMainWindow, QDialog {
    background-color: #1a1a1a;
}
QMenuBar {
    background-color: #2b2b2b;
    color: #d0d0d0;
    border-bottom: 1px solid #3a3a3a;
}
QMenuBar::item:selected { background-color: #3d7ab5; }
QMenu {
    background-color: #2b2b2b;
    color: #d0d0d0;
    border: 1px solid #3a3a3a;
}
QMenu::item:selected { background-color: #3d7ab5; }
QToolBar {
    background-color: #252525;
    border-bottom: 1px solid #3a3a3a;
    spacing: 4px;
    padding: 3px 6px;
}
QToolBar QLabel { color: #a0a0a0; padding-right: 4px; }
QPushButton {
    background-color: #333333;
    color: #d0d0d0;
    border: 1px solid #505050;
    border-radius: 4px;
    padding: 4px 10px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #3d4d5e;
    border-color: #3d7ab5;
}
QPushButton:pressed { background-color: #263040; }
QPushButton:checked {
    background-color: #3d7ab5;
    border-color: #5a9fd4;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #252525;
    color: #555555;
    border-color: #333333;
}
QLabel { color: #c0c0c0; }
QStatusBar {
    background-color: #252525;
    color: #909090;
    border-top: 1px solid #3a3a3a;
    font-size: 10px;
}
QStatusBar::item { border: none; }
QProgressBar {
    background-color: #252525;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    text-align: center;
    color: #d0d0d0;
    max-height: 14px;
}
QProgressBar::chunk {
    background-color: #3d7ab5;
    border-radius: 3px;
}
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #252525;
    color: #d0d0d0;
    border: 1px solid #505050;
    border-radius: 3px;
    padding: 2px 4px;
    min-height: 22px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #3d7ab5;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #2b2b2b;
    color: #d0d0d0;
    selection-background-color: #3d7ab5;
}
QFrame[frameShape="5"] { color: #444444; }
QScrollBar:vertical {
    background-color: #252525;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: #505050;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background-color: #3d7ab5; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background-color: #252525;
    height: 10px;
    border: none;
}
QScrollBar::handle:horizontal {
    background-color: #505050;
    border-radius: 4px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover { background-color: #3d7ab5; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QDialogButtonBox QPushButton { min-width: 70px; }
"""
