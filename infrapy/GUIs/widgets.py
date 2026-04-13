# -*- coding: utf-8 -*-
"""
InfraPy GUI – reusable widgets
--------------------------------
Custom Qt widgets used by the main window:
  - DataLoaderThread   background file-loading thread
  - TerminalWindow     stdout/stderr capture dialog
  - QRangeSlider       double-handle horizontal range slider
  - CircleParamsDialog dialog to enter circle ROI by numbers
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import QRectF, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Background data-loading thread
# ---------------------------------------------------------------------------

class DataLoaderThread(QThread):
    finished = pyqtSignal(object, Exception)

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path

    def run(self) -> None:
        try:
            from infrapy import io
            data = io.load_ir_data(Path(self.file_path))
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            data = data.transpose(0, 2, 1)
            self.finished.emit(data, None)
        except Exception as e:
            self.finished.emit(None, e)


# ---------------------------------------------------------------------------
# Terminal window (stdout / stderr sink)
# ---------------------------------------------------------------------------

class TerminalWindow(QDialog):
    """Simple terminal-like window that captures print output."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Terminal")
        self.resize(800, 200)

        layout = QVBoxLayout(self)
        self.text_area = QLabel()
        self.text_area.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_area.setStyleSheet(
            "background-color: black; color: lime; font-family: monospace;"
        )
        self.text_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.text_area.setWordWrap(True)
        layout.addWidget(self.text_area)

        self._log: list[str] = []

    # Stream-like API (compatible with sys.stdout / sys.stderr)
    def write(self, text: str) -> None:
        self._log.append(text)
        self.text_area.setText("".join(self._log))

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Double-handle range slider
# ---------------------------------------------------------------------------

class QRangeSlider(QWidget):
    """A minimal double-handle horizontal range slider emitting (start, end)."""

    valueChanged = pyqtSignal(tuple)  # emits (start, end) integers

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._start = 0
        self._end = 100
        self._dragging_start = False
        self._dragging_end = False
        self._handle_width = 8

        self.setMinimumSize(150, 30)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setMinimum(self, val: int) -> None:
        self._min = val
        if self._start < val:
            self._start = val
        if self._end < val:
            self._end = val
        self.update()

    def setMaximum(self, val: int) -> None:
        self._max = val
        if self._start > val:
            self._start = val
        if self._end > val:
            self._end = val
        self.update()

    def setValue(self, val: Tuple[int, int]) -> None:
        start, end = val
        self._start = max(self._min, min(start, self._max))
        self._end = max(self._min, min(end, self._max))
        if self._start > self._end:
            self._start = self._end
        self.update()
        self.valueChanged.emit((self._start, self._end))

    def value(self) -> Tuple[int, int]:
        return self._start, self._end

    # ------------------------------------------------------------------
    # Painting & mouse interaction
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        total_range = self._max - self._min
        if total_range <= 0:
            return

        start_x = int((self._start - self._min) / total_range * w)
        end_x = int((self._end - self._min) / total_range * w)

        # Thin track
        track_h = 4
        track_y = (h - track_h) / 2.0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(60, 60, 60))
        p.drawRoundedRect(QRectF(0, track_y, w, track_h), 2, 2)

        # Selected range highlight
        p.setBrush(QColor(61, 122, 181))
        p.drawRoundedRect(QRectF(start_x, track_y, max(1, end_x - start_x), track_h), 2, 2)

        # Circular handles
        r = h / 2 - 2
        p.setBrush(QColor(210, 210, 210))
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawEllipse(QRectF(start_x - r, 2, r * 2, h - 4))
        p.drawEllipse(QRectF(end_x - r, 2, r * 2, h - 4))

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        x = event.pos().x()
        rect = self.rect()
        total_range = self._max - self._min
        if total_range <= 0:
            return

        start_x = (self._start - self._min) / total_range * rect.width()
        end_x = (self._end - self._min) / total_range * rect.width()

        if abs(x - start_x) < self._handle_width:
            self._dragging_start = True
        elif abs(x - end_x) < self._handle_width:
            self._dragging_end = True

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not (self._dragging_start or self._dragging_end):
            return

        x = event.pos().x()
        rect = self.rect()
        total_range = self._max - self._min
        if total_range <= 0:
            return

        val = self._min + (x / rect.width()) * total_range
        val = max(self._min, min(val, self._max))

        if self._dragging_start:
            if val > self._end:
                val = self._end
            self._start = int(val)
        elif self._dragging_end:
            if val < self._start:
                val = self._start
            self._end = int(val)

        self.update()
        self.valueChanged.emit((self._start, self._end))

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._dragging_start = False
        self._dragging_end = False


# ---------------------------------------------------------------------------
# Circle ROI parameters dialog
# ---------------------------------------------------------------------------

class CircleParamsDialog(QDialog):
    """Dialog to enter center (x, y) and diameter (px) for a circle ROI."""

    def __init__(
        self,
        parent: QWidget,
        width: int,
        height: int,
        init_cx: int,
        init_cy: int,
        init_d: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Circle ROI (cx, cy, d)")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.cx_spin = QSpinBox()
        self.cy_spin = QSpinBox()
        self.d_spin = QSpinBox()

        self.cx_spin.setRange(0, max(0, width - 1))
        self.cy_spin.setRange(0, max(0, height - 1))
        self.d_spin.setRange(1, max(1, min(width, height)))

        self.cx_spin.setValue(int(init_cx))
        self.cy_spin.setValue(int(init_cy))
        self.d_spin.setValue(int(init_d))

        form.addRow("Center X (px):", self.cx_spin)
        form.addRow("Center Y (px):", self.cy_spin)
        form.addRow("Diameter (px):", self.d_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Tuple[int, int, int]:
        return int(self.cx_spin.value()), int(self.cy_spin.value()), int(self.d_spin.value())
