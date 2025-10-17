# -*- coding: utf-8 -*-
"""
InfraPy GUI
-----------
A PyQt5/pyqtgraph-based viewer for infrared (IR) data with:
- Time-domain and frequency-domain analysis (FFT and lock-in correlation)
- Spatial cropping in the main viewer (rectangular, circular, or polygonal)
- Circle ROI info readout (center & diameter in data coordinates)
- Enter circle ROI by keyboard (center x, center y, diameter)
- Export FFT magnitude map to .npy

Notes
-----
- Circle & Polygon crops: crop to bounding rectangle and set pixels outside the shape to zero.
  This preserves array rectangularity so analyses keep working without NaN handling.
- All ROI → data alignment is transform-aware via ROI.getArraySlice(...) and map via scene→imageItem.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QUrl, Qt, pyqtSignal, QPointF
from PyQt5.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QPainter,
    QPixmap,
    QImage,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QProgressDialog,
    QInputDialog,
    QSpinBox,
    QPushButton,
    QSplashScreen,
    QVBoxLayout,
    QWidget,
)

# High-DPI configuration
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

# Make local 'infrapy' importable (two levels up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Project imports
from infrapy import io  # IR data loading module
from infrapy.thermoelasticity import lock_in_analysis  # noqa: F401  (kept for future use)

import pyqtgraph as pg


# -------------------------
# Small utilities / constants
# -------------------------

DOC_URL = "https://github.com/LolloCappo/InfraPy"
DEFAULT_COLORMAPS = ["gray", "viridis", "plasma", "inferno", "cividis"]
ROI_COLORS = ["r", "g", "b", "y", "c", "m", "w"]


# -------------------------
# Terminal window (stdout/stderr sink)
# -------------------------

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

    def write(self, text: str) -> None:
        self._log.append(text)
        self.text_area.setText("".join(self._log))

    def flush(self) -> None:
        pass


# -------------------------
# Simple range slider (integer range)
# -------------------------

class QRangeSlider(QWidget):
    """A minimal double-handle horizontal range slider emitting (start, end)."""

    valueChanged = pyqtSignal(tuple)  # Emits (start, end) integers

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

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        rect = self.rect()

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(200, 200, 200))
        p.drawRect(rect)

        total_range = self._max - self._min
        if total_range <= 0:
            return

        start_x = int((self._start - self._min) / total_range * rect.width())
        end_x = int((self._end - self._min) / total_range * rect.width())

        p.setBrush(QColor(100, 150, 200))
        p.drawRect(start_x, 0, max(1, end_x - start_x), rect.height())

        p.setBrush(QColor(50, 50, 150))
        p.drawRect(start_x - self._handle_width // 2, 0, self._handle_width, rect.height())
        p.drawRect(end_x - self._handle_width // 2, 0, self._handle_width, rect.height())

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


# -------------------------
# Dialog to enter circle parameters
# -------------------------

class CircleParamsDialog(QDialog):
    """Dialog to enter center (x, y) and diameter (px) for circle ROI."""

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

        # Ranges in array coordinates
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
        layout.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def values(self) -> Tuple[int, int, int]:
        return int(self.cx_spin.value()), int(self.cy_spin.value()), int(self.d_spin.value())


# -------------------------
# Main window
# -------------------------

class IRViewerPG(QMainWindow):
    """Main InfraPy viewer window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("INFRAPY")
        self.setWindowIcon(QIcon("infrapy/GUIs/icon.png"))

        # Window sizing
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry() if screen else None
        if screen_size:
            width = int(screen_size.width() * 0.8)
            height = int(screen_size.height() * 0.8)
            self.resize(width, height)
            self.move(
                screen_size.left() + (screen_size.width() - width) // 2,
                screen_size.top() + (screen_size.height() - height) // 2,
            )

        # ---- State
        self.loaded_data: Optional[np.ndarray] = None  # (t, y, x)
        self.original_data: Optional[np.ndarray] = None
        self.current_frame: int = 0
        self.fs: Optional[float] = None
        self.use_time_axis: bool = False
        self.current_fft_index: int = 0

        # --- Crop state
        self.crop_roi: Optional[pg.ROI] = None
        self._backup_loaded_data: Optional[np.ndarray] = None
        self._backup_original_data: Optional[np.ndarray] = None
        self._last_crop_was_ellipse: bool = False
        self._last_crop_was_polygon: bool = False
        self._circle_update_lock: bool = False

        # Terminal window (redirect stdout/stderr)
        self.terminal_window = TerminalWindow(self)
        sys.stdout = self.terminal_window  # type: ignore[assignment]
        sys.stderr = self.terminal_window  # type: ignore[assignment]

        self._init_menu()
        self._init_ui()

    # ----- Window utilities

    def fit_to_screen(self) -> None:
        screen = QApplication.screenAt(self.pos())
        if not screen:
            return
        screen_size = screen.availableGeometry()
        width = int(screen_size.width() * 0.8)
        height = int(screen_size.height() * 0.8)
        self.resize(width, height)
        self.move(
            screen_size.left() + (screen_size.width() - width) // 2,
            screen_size.top() + (screen_size.height() - height) // 2,
        )

    # ----- Menus

    def _init_menu(self) -> None:
        menubar = self.menuBar()

        # File > Load
        file_menu = menubar.addMenu("File")
        load_action = QAction("Load Data", self)
        load_action.triggered.connect(self.load_ir_data)
        file_menu.addAction(load_action)

        # View
        view_menu = menubar.addMenu("View")
        colormap_menu = view_menu.addMenu("Colormap")
        self.colormaps = list(DEFAULT_COLORMAPS)
        for cmap in self.colormaps:
            action = QAction(cmap, self)
            action.triggered.connect(lambda _checked, c=cmap: self.set_colormap(c))
            colormap_menu.addAction(action)

        resize_action = QAction("Fit to Screen", self)
        resize_action.triggered.connect(self.fit_to_screen)
        view_menu.addAction(resize_action)

        terminal_action = QAction("Terminal", self)
        terminal_action.triggered.connect(self.terminal_window.show)
        view_menu.addAction(terminal_action)

        # Analysis
        analysis_menu = menubar.addMenu("Analysis")

        radiation_action = QAction("Radiation", self)
        radiation_action.triggered.connect(
            lambda: self.show_analysis_window("Radiation Analysis", "Radiation tools go here.")
        )
        analysis_menu.addAction(radiation_action)

        time_action = QAction("Time Domain Analysis", self)
        time_action.triggered.connect(self.run_time_analysis)
        analysis_menu.addAction(time_action)

        freq_domain_menu = analysis_menu.addMenu("Frequency domain")

        fft_action = QAction("Fast Fourier Transform", self)
        fft_action.triggered.connect(self.run_fft_analysis)
        freq_domain_menu.addAction(fft_action)

        corr_action = QAction("Lock-In Correlation", self)
        corr_action.triggered.connect(self.run_correlation_analysis)
        freq_domain_menu.addAction(corr_action)

        # Help
        help_menu = menubar.addMenu("Help")
        doc_action = QAction("Documentation", self)
        doc_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(DOC_URL)))
        help_menu.addAction(doc_action)

        update_action = QAction("Check for updates", self)
        update_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(DOC_URL)))
        help_menu.addAction(update_action)

    # ----- Central UI

    def _init_ui(self) -> None:
        central = QWidget()
        vlayout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vlayout.addWidget(self.progress_bar)

        layout = QHBoxLayout()
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.show()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.show()
        self.image_view.timeLine.hide()
        self.image_view.getView().setAspectLocked(True)
        layout.addWidget(self.image_view)
        vlayout.addLayout(layout)

        bottom = QHBoxLayout()

        self.trim_slider = QRangeSlider()
        self.trim_slider.setEnabled(False)
        self.trim_slider.valueChanged.connect(self.update_frame_display)
        bottom.addWidget(self.trim_slider)

        self.apply_button = QPushButton("Apply Clip")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_clipping)
        bottom.addWidget(self.apply_button)

        self.undo_button = QPushButton("Undo Clip")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_clipping)
        bottom.addWidget(self.undo_button)

        self.frame_label = QLabel("Frame: 0 / 0")
        bottom.addWidget(self.frame_label)

        self.axis_toggle = QPushButton("Show Time [s]")
        self.axis_toggle.setCheckable(True)
        self.axis_toggle.toggled.connect(self.toggle_time_axis)
        bottom.addWidget(self.axis_toggle)

        self.add_rect_roi_btn = QPushButton("Add Rect ROI")
        self.add_rect_roi_btn.clicked.connect(self.add_rect_roi)
        self.add_rect_roi_btn.setEnabled(False)
        bottom.addWidget(self.add_rect_roi_btn)

        self.add_circle_roi_btn = QPushButton("Add Circle ROI")
        self.add_circle_roi_btn.setToolTip("Circular ROI (forced 1:1)")
        self.add_circle_roi_btn.clicked.connect(self.add_circle_roi)
        self.add_circle_roi_btn.setEnabled(False)
        bottom.addWidget(self.add_circle_roi_btn)

        # Polygon ROI
        self.add_poly_roi_btn = QPushButton("Add Polygon ROI")
        self.add_poly_roi_btn.setToolTip("Custom polygon ROI")
        self.add_poly_roi_btn.clicked.connect(self.add_polygon_roi)
        self.add_poly_roi_btn.setEnabled(False)
        bottom.addWidget(self.add_poly_roi_btn)

        self.apply_crop_btn = QPushButton("Apply Crop")
        self.apply_crop_btn.setEnabled(False)
        self.apply_crop_btn.clicked.connect(self.apply_spatial_crop)
        bottom.addWidget(self.apply_crop_btn)

        self.remove_roi_btn = QPushButton("Remove ROI")
        self.remove_roi_btn.setEnabled(False)
        self.remove_roi_btn.clicked.connect(self.remove_crop_roi)
        bottom.addWidget(self.remove_roi_btn)

        self.undo_crop_btn = QPushButton("Undo Crop")
        self.undo_crop_btn.setEnabled(False)
        self.undo_crop_btn.clicked.connect(self.undo_spatial_crop)
        bottom.addWidget(self.undo_crop_btn)

        self.set_circle_btn = QPushButton("Set Circle (cx, cy, d)")
        self.set_circle_btn.setToolTip("Type center X/Y and diameter")
        self.set_circle_btn.setEnabled(False)
        self.set_circle_btn.clicked.connect(self.set_circle_by_numbers)
        bottom.addWidget(self.set_circle_btn)

        self.circle_info_label = QLabel("Circle ROI: —")
        bottom.addWidget(self.circle_info_label)

        vlayout.addLayout(bottom)

    # ----- Axis / slider updates

    def toggle_time_axis(self, checked: bool) -> None:
        self.use_time_axis = checked
        self.axis_toggle.setText("Show Frame [#]" if checked else "Show Time [s]")
        self.update_frame_display()

    # ----- Data I/O

    def load_ir_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open IR File",
            "",
            "IR Files (*.tif *.tiff *.csv *.sfmov *.npy *.npz *.hcc);;All Files (*)",
        )
        if not file_path:
            return

        try:
            self.progress_bar.setVisible(True)
            QApplication.processEvents()

            data = io.load_ir_data(Path(file_path))
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            data = data.transpose(0, 2, 1)  # keep original orientation change

            self.original_data = data
            self.loaded_data = data.copy()

            self.remove_crop_roi()
            self.circle_info_label.setText("Circle ROI: —")

            fs, ok = QInputDialog.getDouble(
                self, "Sampling Frequency", "Enter sampling frequency [Hz]:", 50.0, 0.01, 1e6, 2
            )
            if not ok:
                fs = 1.0

            self.fs = float(fs)
            self.current_frame = 0

            self.update_viewer()
            self._init_slider()
            self.undo_button.setEnabled(False)

            self.add_rect_roi_btn.setEnabled(True)
            self.add_circle_roi_btn.setEnabled(True)
            self.add_poly_roi_btn.setEnabled(True)
            self.set_circle_btn.setEnabled(True)

        except Exception as e:
            print(f"Error loading data: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def _init_slider(self) -> None:
        if self.loaded_data is None:
            return
        n = self.loaded_data.shape[0]
        self.trim_slider.setMinimum(0)
        self.trim_slider.setMaximum(n - 1)
        self.trim_slider.setValue((0, n - 1))
        self.trim_slider.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.update_frame_display()

    def update_frame_display(self) -> None:
        if self.loaded_data is None:
            return

        start, end = self.trim_slider.value()
        if start > end:
            start, end = end, start
        if not (start <= self.current_frame <= end):
            self.current_frame = start

        xvals = np.arange(self.loaded_data.shape[0]) / self.fs if (self.use_time_axis and self.fs) \
            else np.arange(self.loaded_data.shape[0])

        self.image_view.setImage(self.loaded_data, xvals=xvals, autoLevels=False)
        self.image_view.setCurrentIndex(self.current_frame)
        self.frame_label.setText(f"Frame: {self.current_frame} / {self.loaded_data.shape[0] - 1}")
        self.apply_button.setText(f"Apply Clip ({start}–{end})")

    def apply_clipping(self) -> None:
        if self.loaded_data is None:
            return
        start, end = self.trim_slider.value()
        if start > end:
            start, end = end, start
        self.loaded_data = self.loaded_data[start: end + 1]
        self.current_frame = 0
        self._init_slider()
        self.update_viewer()
        self.undo_button.setEnabled(True)

    def undo_clipping(self) -> None:
        if self.original_data is None:
            return
        self.loaded_data = self.original_data.copy()
        self.current_frame = 0
        self._init_slider()
        self.update_viewer()
        self.undo_button.setEnabled(False)

    def update_viewer(self) -> None:
        if self.loaded_data is None:
            return
        self.image_view.setImage(self.loaded_data, xvals=np.arange(self.loaded_data.shape[0]), autoLevels=True)
        self.image_view.setCurrentIndex(self.current_frame)
        self.frame_label.setText(f"Frame: {self.current_frame} / {self.loaded_data.shape[0] - 1}")

    def set_colormap(self, cmap: str) -> None:
        try:
            lut = pg.colormap.get(cmap).getLookupTable(0.0, 1.0, 256)
            self.image_view.setColorMap(pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut))
        except Exception as e:
            print(f"Error setting colormap: {e}")

    def show_analysis_window(self, title: str, text: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(400, 200)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(text))
        dlg.exec_()

    # ----- Analyses

    def run_time_analysis(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        if self.fs is None:
            print("Sampling frequency not set.")
            return
        print(f"Running Time-Domain analysis with fs={self.fs}")
        self._show_time_analysis_viewer(self.loaded_data, self.fs)

    def run_fft_analysis(self) -> None:
        if self.loaded_data is None or self.fs is None:
            print("Video or sampling frequency missing.")
            return

        data = self.loaded_data
        fs = self.fs
        n_frames, h, w = data.shape
        n_freqs = n_frames // 2 + 1

        progress = QProgressDialog("Computing FFT...", "Cancel", 0, h, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        fft_data = np.empty((n_freqs, h, w), dtype=np.complex64)
        time_series = data - data.mean(axis=0)

        for i in range(h):
            for j in range(w):
                signal = time_series[:, i, j]
                fft_result = np.fft.rfft(signal)
                fft_data[:, i, j] = fft_result
            progress.setValue(i + 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                print("FFT cancelled.")
                return

        fft_freqs = np.fft.rfftfreq(n_frames, d=1 / fs)
        fft_mag = np.abs(fft_data) * (2 / n_frames)
        fft_phase = np.angle(fft_data)

        def spectrum_callback(mean_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            spectrum = np.abs(np.fft.rfft(mean_signal - mean_signal.mean()))
            return fft_freqs, spectrum

        progress.setValue(h)
        progress.close()

        self._show_frequency_analysis_viewer(
            data, fft_freqs, fft_mag, fft_phase, spectrum_callback, enable_save_mag=True
        )

    def _show_frequency_analysis_viewer(
        self,
        original_data: np.ndarray,
        freq_vector: np.ndarray,
        mag_data: np.ndarray,
        phase_data: np.ndarray,
        spectrum_callback: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        enable_save_mag: bool = False,
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Frequency Domain Analysis")
        dlg.resize(self.size())
        vbox = QVBoxLayout(dlg)

        top = QHBoxLayout()
        vbox.addLayout(top)

        raw_view = pg.ImageView()
        raw_view.setImage(original_data, autoLevels=True)
        raw_view.ui.menuBtn.hide()
        top.addWidget(raw_view)

        mag_view = pg.ImageView()
        mag_view.setImage(mag_data, autoLevels=True)
        mag_view.ui.menuBtn.hide()
        mag_view.ui.roiBtn.hide()
        top.addWidget(mag_view)

        ph_view = pg.ImageView()
        ph_view.setImage(phase_data, autoLevels=True)
        ph_view.ui.menuBtn.hide()
        ph_view.ui.roiBtn.hide()
        top.addWidget(ph_view)

        spectrum_plot = pg.PlotWidget()
        spectrum_plot.setLabel("bottom", "Frequency [Hz]")
        spectrum_plot.setLabel("left", "Amplitude")
        vbox.addWidget(spectrum_plot)

        spectrum_curve = spectrum_plot.plot([], pen="y")
        cursor_line = pg.InfiniteLine(angle=90, movable=True, pen="r")
        spectrum_plot.addItem(cursor_line)

        roi = pg.RectROI([30, 30], [40, 40], pen=pg.mkPen("r", width=2))
        raw_view.addItem(roi)

        raw_view.setMinimumSize(400, 400)
        mag_view.setMinimumSize(400, 400)
        ph_view.setMinimumSize(400, 400)
        spectrum_plot.setMinimumHeight(150)

        if enable_save_mag:
            save_row = QHBoxLayout()
            vbox.addLayout(save_row)
            save_btn = QPushButton("Save FFT Magnitude (.npy)")
            save_btn.setToolTip("Save the full FFT magnitude array (n_freqs × H × W) to .npy")
            save_row.addWidget(save_btn)
            save_row.addStretch(1)

            def _save_mag():
                path, _ = QFileDialog.getSaveFileName(
                    dlg, "Save FFT Magnitude", "fft_magnitude.npy", "NumPy binary (*.npy)"
                )
                if path:
                    try:
                        np.save(path, mag_data)
                        print(f"Saved FFT magnitude to: {path}")
                    except Exception as e:
                        print(f"Failed to save FFT magnitude: {e}")

            save_btn.clicked.connect(_save_mag)

        def update_fft_images_from_cursor() -> None:
            freq_pos = cursor_line.value()
            idx = np.abs(freq_vector - freq_pos).argmin()
            self.current_fft_index = int(idx)
            slice_mag = mag_data[idx]
            slice_phase = phase_data[idx]
            mag_view.setImage(slice_mag[np.newaxis], autoLevels=False)
            ph_view.setImage(slice_phase[np.newaxis], autoLevels=False)

        def update_spectrum() -> None:
            try:
                h_, w_ = roi.size()
                pos = roi.pos()
                y0, x0 = math.floor(pos[0]), math.floor(pos[1])
                y1, x1 = math.ceil(pos[0] + w_), math.ceil(pos[1] + h_)
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(original_data.shape[2], x1); y1 = min(original_data.shape[1], y1)
                if x1 <= x0 or y1 <= y0:
                    print("ROI outside bounds or zero area.")
                    return
                roi_data = original_data[:, y0:y1, x0:x1]
                mean_signal = roi_data.mean(axis=(1, 2))
                freqs, spectrum = spectrum_callback(mean_signal)
                spectrum_curve.setData(freqs, spectrum)
                if self.current_fft_index < len(freqs):
                    cursor_line.setPos(freqs[self.current_fft_index])
                update_fft_images_from_cursor()
            except Exception as e:
                print("Spectrum update failed:", e)

        roi.sigRegionChanged.connect(update_spectrum)
        cursor_line.sigPositionChanged.connect(update_fft_images_from_cursor)
        update_spectrum()
        dlg.exec_()

    def run_correlation_analysis(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        if self.fs is None:
            print("Sampling frequency not set.")
            return

        data = self.loaded_data
        fs = self.fs
        n_frames, h, w = data.shape
        N = n_frames

        freq_vector = np.linspace(0, fs / 2, 200)
        n_freqs = len(freq_vector)
        t = np.arange(N) / fs

        mag_data = np.empty((n_freqs, h, w), dtype=np.float32)
        phase_data = np.empty((n_freqs, h, w), dtype=np.float32)

        progress = QProgressDialog("Computing Lock-in Correlation...", "Cancel", 0, n_freqs, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        for k, fl in enumerate(freq_vector):
            sine = np.sin(2 * np.pi * fl * t)
            cosine = np.cos(2 * np.pi * fl * t)
            sine_3d = sine[:, None, None]
            cosine_3d = cosine[:, None, None]
            X = (2 / N) * np.sum(data * cosine_3d, axis=0)
            Y = (2 / N) * np.sum(data * sine_3d, axis=0)
            mag_data[k] = np.sqrt(X**2 + Y**2)
            phase_data[k] = np.degrees(np.arctan2(Y, X))
            progress.setValue(k + 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                print("Lock-in correlation cancelled.")
                return

        progress.close()

        def spectrum_callback(mean_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            spectrum_list = []
            for fl in freq_vector:
                sine = np.sin(2 * np.pi * fl * t)
                cosine = np.cos(2 * np.pi * fl * t)
                X_roi = (2 / N) * np.sum(mean_signal * cosine)
                Y_roi = (2 / N) * np.sum(mean_signal * sine)
                spectrum_list.append(np.sqrt(X_roi**2 + Y_roi**2))
            return freq_vector, np.array(spectrum_list)

        self._show_frequency_analysis_viewer(
            data, freq_vector, mag_data, phase_data, spectrum_callback, enable_save_mag=False
        )

    def _show_time_analysis_viewer(self, data: np.ndarray, fs: float) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Time Domain Analysis")
        dlg.resize(self.size())
        layout = QVBoxLayout(dlg)

        top = QHBoxLayout()
        layout.addLayout(top)

        raw_view = pg.ImageView()
        raw_view.setImage(data, autoLevels=True)
        raw_view.ui.menuBtn.hide()
        raw_view.ui.roiBtn.hide()
        top.addWidget(raw_view)

        time_plot = pg.PlotWidget()
        time_plot.setLabel("bottom", "Time [s]")
        time_plot.setLabel("left", "Intensity")
        layout.addWidget(time_plot)

        roi_list: list[pg.ROI] = []
        curves: list[pg.PlotDataItem] = []

        ctrl_layout = QHBoxLayout()
        layout.addLayout(ctrl_layout)

        detrend_cb = QPushButton("Detrend")
        detrend_cb.setCheckable(True)
        filter_cb = QPushButton("Bandpass")
        filter_cb.setCheckable(True)
        smooth_cb = QPushButton("Smooth")
        smooth_cb.setCheckable(True)

        ctrl_layout.addWidget(detrend_cb)
        ctrl_layout.addWidget(filter_cb)
        ctrl_layout.addWidget(smooth_cb)

        t = np.arange(data.shape[0]) / fs

        def process_signal(sig: np.ndarray) -> np.ndarray:
            out = sig.copy()
            if detrend_cb.isChecked():
                out = out - np.polyval(np.polyfit(t, out, 1), t)
            if filter_cb.isChecked():
                from scipy.signal import butter, filtfilt
                b, a = butter(3, [0.1, 0.3], btype="band")
                out = filtfilt(b, a, out)
            if smooth_cb.isChecked():
                win = 5
                out = np.convolve(out, np.ones(win) / win, mode="same")
            return out

        def update_all_traces() -> None:
            for roi, curve in zip(roi_list, curves):
                pos = roi.pos()
                size = roi.size()
                x0, y0 = int(pos[0]), int(pos[1])
                w_, h_ = int(size[0]), int(size[1])
                x1, y1 = x0 + w_, y0 + h_
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(data.shape[2], x1); y1 = min(data.shape[1], y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                roi_data = data[:, y0:y1, x0:x1]
                mean_signal = roi_data.mean(axis=(1, 2))
                curve.setData(t, process_signal(mean_signal))

        def add_roi() -> None:
            color = ROI_COLORS[len(roi_list) % len(ROI_COLORS)]
            roi = pg.RectROI([30 + len(roi_list) * 10, 30], [40, 40], pen=pg.mkPen(color, width=2))
            raw_view.addItem(roi)
            roi_list.append(roi)
            curve = time_plot.plot(pen=pg.mkPen(color, width=2))
            curves.append(curve)
            roi.sigRegionChanged.connect(update_all_traces)
            update_all_traces()

        add_roi()

        raw_view.scene.sigMouseClicked.connect(
            lambda ev: add_roi() if ev.modifiers() & Qt.ControlModifier else None
        )

        detrend_cb.clicked.connect(update_all_traces)
        filter_cb.clicked.connect(update_all_traces)
        smooth_cb.clicked.connect(update_all_traces)

        dlg.exec_()

    # ---- Crop helpers -------------------------------------------------

    def _get_image_item(self):
        img_item = getattr(self.image_view, "imageItem", None)
        if img_item is None and hasattr(self.image_view, "getImageItem"):
            img_item = self.image_view.getImageItem()
        return img_item

    def _ensure_no_existing_roi(self) -> None:
        if self.crop_roi is not None:
            try:
                self.image_view.removeItem(self.crop_roi)
            except Exception:
                pass
            self.crop_roi = None
        self.apply_crop_btn.setEnabled(False)
        self.remove_roi_btn.setEnabled(False)
        self._last_crop_was_ellipse = False
        self._last_crop_was_polygon = False

    def add_rect_roi(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        self._ensure_no_existing_roi()
        _, H, W = self.loaded_data.shape
        w = max(20, int(W * 0.25))
        h = max(20, int(H * 0.25))
        x = int((W - w) / 2)
        y = int((H - h) / 2)
        self.crop_roi = pg.RectROI([x, y], [w, h], pen=pg.mkPen('y', width=2))
        self.image_view.addItem(self.crop_roi)
        self.apply_crop_btn.setEnabled(True)
        self.remove_roi_btn.setEnabled(True)

    def add_circle_roi(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        self._ensure_no_existing_roi()
        _, H, W = self.loaded_data.shape
        d = max(20, int(min(W, H) * 0.4))
        x = int((W - d) / 2)
        y = int((H - d) / 2)
        self.crop_roi = pg.EllipseROI([x, y], [d, d], pen=pg.mkPen('c', width=2))
        self.image_view.addItem(self.crop_roi)
        self.crop_roi.sigRegionChanged.connect(self._enforce_circular_roi)
        self._enforce_circular_roi()
        self._last_crop_was_ellipse = True
        self.apply_crop_btn.setEnabled(True)
        self.remove_roi_btn.setEnabled(True)
        self._update_circle_info_label()

    def add_polygon_roi(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        self._ensure_no_existing_roi()
        _, H, W = self.loaded_data.shape
        cx, cy = W / 2.0, H / 2.0
        r = max(20, min(W, H) * 0.2)
        pts = [[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]]
        self.crop_roi = pg.PolyLineROI(pts, closed=True, pen=pg.mkPen('m', width=2))
        self.image_view.addItem(self.crop_roi)
        self._last_crop_was_polygon = True
        self.apply_crop_btn.setEnabled(True)
        self.remove_roi_btn.setEnabled(True)

    def _enforce_circular_roi(self) -> None:
        if self.crop_roi is None or self._circle_update_lock:
            return
        try:
            self._circle_update_lock = True
            pos = self.crop_roi.pos()
            size = self.crop_roi.size()
            w = float(size[0]); h = float(size[1])
            if w <= 0 or h <= 0:
                return
            s = min(w, h)
            cx = float(pos[0]) + w / 2.0
            cy = float(pos[1]) + h / 2.0
            new_pos = [cx - s / 2.0, cy - s / 2.0]
            self.crop_roi.setPos(new_pos, finish=False)
            self.crop_roi.setSize([s, s], finish=False)
        finally:
            self._circle_update_lock = False
        self._update_circle_info_label()

    def _update_circle_info_label(self) -> None:
        if self.crop_roi is None or not self._last_crop_was_ellipse:
            self.circle_info_label.setText("Circle ROI: —")
            return
        bounds = self._roi_bounds_clamped()
        if bounds is None:
            self.circle_info_label.setText("Circle ROI: —")
            return
        x0, y0, x1, y1 = bounds
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        d = min(x1 - x0, y1 - y0)
        self.circle_info_label.setText(
            f"Circle ROI: center=({int(round(cx))}, {int(round(cy))}), d={int(round(d))} px"
        )

    def set_circle_by_numbers(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        _, H, W = self.loaded_data.shape

        if self.crop_roi is not None and self._last_crop_was_ellipse:
            bounds = self._roi_bounds_clamped()
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                init_cx = int(round((x0 + x1) / 2.0))
                init_cy = int(round((y0 + y1) / 2.0))
                init_d = int(round(min(x1 - x0, y1 - y0)))
            else:
                init_cx, init_cy, init_d = W // 2, H // 2, max(1, min(W, H) // 2)
        else:
            init_cx, init_cy, init_d = W // 2, H // 2, max(1, min(W, H) // 2)

        dlg = CircleParamsDialog(self, W, H, init_cx, init_cy, init_d)
        if dlg.exec_() != QDialog.Accepted:
            return
        cx, cy, d = dlg.values()

        d = max(1, min(d, min(W, H)))
        half = d / 2.0
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        x0 = max(0, min(x0, W - d))
        y0 = max(0, min(y0, H - d))

        if self.crop_roi is None or not self._last_crop_was_ellipse:
            self._ensure_no_existing_roi()
            self.crop_roi = pg.EllipseROI([x0, y0], [d, d], pen=pg.mkPen('c', width=2))
            self.image_view.addItem(self.crop_roi)
            self.crop_roi.sigRegionChanged.connect(self._enforce_circular_roi)
            self._last_crop_was_ellipse = True
            self.remove_roi_btn.setEnabled(True)
        else:
            self.crop_roi.setPos([x0, y0], finish=False)
            self.crop_roi.setSize([d, d], finish=False)

        self._enforce_circular_roi()
        self.apply_crop_btn.setEnabled(True)
        self._update_circle_info_label()

    def remove_crop_roi(self) -> None:
        self._ensure_no_existing_roi()
        self.circle_info_label.setText("Circle ROI: —")

    def _roi_bounds_clamped(self) -> Optional[tuple[int, int, int, int]]:
        if self.loaded_data is None or self.crop_roi is None:
            return None

        img2d = self.loaded_data[self.current_frame]
        img_item = self._get_image_item()
        if img_item is None:
            print("Internal error: ImageItem not found.")
            return None

        try:
            slices, _ = self.crop_roi.getArraySlice(img2d, img_item)  # (slice_y, slice_x)
            sy, sx = slices
            y0 = 0 if sy.start is None else int(sy.start)
            y1 = img2d.shape[0] if sy.stop is None else int(sy.stop)
            x0 = 0 if sx.start is None else int(sx.start)
            x1 = img2d.shape[1] if sx.stop is None else int(sx.stop)
            y0 = max(0, min(y0, img2d.shape[0])); y1 = max(0, min(y1, img2d.shape[0]))
            x0 = max(0, min(x0, img2d.shape[1])); x1 = max(0, min(x1, img2d.shape[1]))
            if x1 <= x0 or y1 <= y0:
                return None
            return x0, y0, x1, y1
        except Exception as e:
            print(f"Failed to compute ROI slice: {e}")
            return None

    # --- Polygon vertices in image coordinates (robust: scene -> imageItem)
    def _polygon_vertices_image_coords(self) -> Optional[List[Tuple[float, float]]]:
        """
        Return polygon vertices in *image array* coordinates for PolyLineROI.
        Map ROI-local handle positions -> scene -> imageItem coords for robust alignment.
        """
        if self.crop_roi is None or not isinstance(self.crop_roi, pg.PolyLineROI):
            return None

        img_item = self._get_image_item()
        if img_item is None:
            print("Internal error: ImageItem not found.")
            return None

        verts: List[Tuple[float, float]] = []
        try:
            for h in self.crop_roi.getHandles():
                handle_item = h.get("item", None)
                if handle_item is not None:
                    p_local = handle_item.pos()  # QPointF (ROI-local)
                else:
                    p_local = h.get("pos", None)
                    if p_local is None:
                        continue
                # ROI-local -> scene -> image coords
                p_scene = self.crop_roi.mapToScene(p_local)
                p_img = img_item.mapFromScene(p_scene)
                verts.append((float(p_img.x()), float(p_img.y())))
            if len(verts) < 3:
                return None
            return verts
        except Exception as e:
            print(f"Cannot extract polygon vertices (image coords): {e}")
            return None

    # --- Polygon mask rasterization via vectorized ray-casting (robust, no Qt)
    def _rasterize_polygon_mask(self, verts_xy: List[Tuple[float, float]], x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """
        Rasterize a polygon (verts in image coords) into a (h, w) uint8 mask
        using a vectorized ray-casting algorithm (inside=True -> 255).
        """
        if w <= 0 or h <= 0 or len(verts_xy) < 3:
            return np.zeros((h, w), dtype=np.uint8)

        xv = np.asarray([vx - x0 for (vx, _vy) in verts_xy], dtype=np.float64)
        yv = np.asarray([vy - y0 for (_vx, vy) in verts_xy], dtype=np.float64)

        yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing='ij')

        x1 = xv
        y1 = yv
        x2 = np.roll(xv, -1)
        y2 = np.roll(yv, -1)

        cond = ((y1[:, None, None] <= yy) & (y2[:, None, None] > yy)) | \
               ((y2[:, None, None] <= yy) & (y1[:, None, None] > yy))
        xints = (x2 - x1)[:, None, None] * (yy - y1[:, None, None]) / ((y2 - y1)[:, None, None] + 1e-12) + x1[:, None, None]

        crossings = cond & (xx < xints)
        inside = np.count_nonzero(crossings, axis=0) % 2 == 1

        return (inside.astype(np.uint8) * 255)

    def apply_spatial_crop(self) -> None:
        """
        Apply the current ROI to crop the stack (t, y, x) using precise bounds
        from ROI.getArraySlice. Rect ROI -> rectangular crop.
        Circle ROI -> rectangular crop + zero outside the circle.
        Polygon ROI -> rectangular crop + zero outside the polygon.
        """
        if self.loaded_data is None or self.crop_roi is None:
            print("No ROI to apply.")
            return
        bounds = self._roi_bounds_clamped()
        if bounds is None:
            print("ROI out of bounds or zero area.")
            return
        x0, y0, x1, y1 = bounds

        hw = self.image_view.getHistogramWidget() if hasattr(self.image_view, "getHistogramWidget") \
            else self.image_view.ui.histogram
        prev_levels = hw.getLevels()

        self._backup_loaded_data = self.loaded_data
        self._backup_original_data = self.original_data

        cropped_loaded = self.loaded_data[:, y0:y1, x0:x1]
        cropped_original = None if self.original_data is None else self.original_data[:, y0:y1, x0:x1]

        # Circle mask
        if self._last_crop_was_ellipse:
            h = y1 - y0; w = x1 - x0
            r = min(h, w) / 2.0
            cy = (h - 1) / 2.0; cx = (w - 1) / 2.0
            yy, xx = np.ogrid[:h, :w]
            circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
            mask_loaded = circle.astype(cropped_loaded.dtype, copy=False)[None, :, :]
            cropped_loaded = cropped_loaded * mask_loaded
            if cropped_original is not None:
                mask_original = circle.astype(cropped_original.dtype, copy=False)[None, :, :]
                cropped_original = cropped_original * mask_original

        # Polygon mask (always honor actual PolyLineROI)
        if isinstance(self.crop_roi, pg.PolyLineROI):
            verts = self._polygon_vertices_image_coords()
            if verts is not None:
                h = y1 - y0; w = x1 - x0
                mask_u8 = self._rasterize_polygon_mask(verts, x0, y0, w, h)  # 0/255
                mask_bool = mask_u8 > 0
                mask_loaded = mask_bool.astype(cropped_loaded.dtype, copy=False)[None, :, :]
                cropped_loaded = cropped_loaded * mask_loaded
                if cropped_original is not None:
                    mask_original = mask_bool.astype(cropped_original.dtype, copy=False)[None, :, :]
                    cropped_original = cropped_original * mask_original
            else:
                print("Polygon vertices not found; rectangular crop applied.")

        self.loaded_data = cropped_loaded
        if cropped_original is not None:
            self.original_data = cropped_original

        self._ensure_no_existing_roi()
        self.current_frame = 0
        self._init_slider()
        self.update_viewer()

        hw.setLevels(*prev_levels)

        self.undo_crop_btn.setEnabled(True)
        print(f"Cropped to x:[{x0},{x1}), y:[{y0},{y1}) ; shape -> {self.loaded_data.shape}")

    def undo_spatial_crop(self) -> None:
        if self._backup_loaded_data is None:
            print("No crop to undo.")
            return
        self.loaded_data = self._backup_loaded_data
        self.original_data = self._backup_original_data
        self._backup_loaded_data = None
        self._backup_original_data = None
        self.current_frame = 0
        self._init_slider()
        self.update_viewer()
        self.undo_crop_btn.setEnabled(False)
        print("Crop undone.")


# -------------------------
# App bootstrap with splash
# -------------------------

def show_splash_then_main() -> None:
    app = QApplication(sys.argv)

    logo = Path(__file__).parent / "icon.png"
    splash: Optional[QSplashScreen] = None
    if logo.exists():
        pix = QPixmap(str(logo)).scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

    def _show_main():
        w = IRViewerPG()
        w.show()
        if splash is not None:
            splash.finish(w)

    QTimer.singleShot(1500, _show_main)
    sys.exit(app.exec_())


if __name__ == "__main__":
    show_splash_then_main()