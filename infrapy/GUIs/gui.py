# -*- coding: utf-8 -*-
"""
InfraPy GUI
-----------
A PyQt5/pyqtgraph-based viewer for infrared (IR) data with:
- Time-domain and frequency-domain analysis (FFT and lock-in correlation)
- Spatial cropping in the main viewer (rectangular or circular)
- Circle ROI info readout (center & diameter in data coordinates)
- Enter circle ROI by keyboard (center x, center y, diameter)
- Export FFT magnitude map to .npy

Notes
-----
- Circle crop: forced 1:1 while interacting; crop uses transform-aware slicing
  so the region matches what you draw/type. Pixels outside the circle within the
  bounding box are set to zero (keeps analyses working without NaNs).
- FFT viewer includes a button to save the full magnitude cube as .npy.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QUrl, Qt, pyqtSignal

from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
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


# ----------------------------------------------
# High-DPI configuration (as early as possible)
# ----------------------------------------------

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

# Make local 'infrapy' importable (two levels up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Project imports
from infrapy import io  # IR data loading module
from infrapy.thermoelasticity import lock_in_analysis  # noqa: F401  (kept as in original)

import pyqtgraph as pg


# ---------------------------
# Small utilities / constants
# -------------------------

DOC_URL = "https://github.com/LolloCappo/InfraPy"
DEFAULT_COLORMAPS = ["gray", "viridis", "plasma", "inferno", "cividis"]
ROI_COLORS = ["r", "g", "b", "y", "c", "m", "w"]


######################

class DataLoaderThread(QThread):
    finished = pyqtSignal(object, Exception)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            from infrapy import io
            data = io.load_ir_data(Path(self.file_path))
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            data = data.transpose(0, 2, 1)
            self.finished.emit(data, None)
        except Exception as e:
            self.finished.emit(None, e)

######################

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

    # Stream-like API (compatible with sys.stdout/sys.stderr)
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

    # Public API (unchanged behavior)
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

    # Painting & interaction (unchanged visuals)
    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        rect = self.rect()

        # Background bar
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(200, 200, 200))
        p.drawRect(rect)

        total_range = self._max - self._min
        if total_range <= 0:
            return

        start_x = int((self._start - self._min) / total_range * rect.width())
        end_x = int((self._end - self._min) / total_range * rect.width())

        # Selected range bar
        p.setBrush(QColor(100, 150, 200))
        p.drawRect(start_x, 0, max(1, end_x - start_x), rect.height())

        # Handles
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

        # Window sizing relative to current screen
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
        self.loaded_data: Optional[np.ndarray] = None  # shape: (t, y, x)
        self.original_data: Optional[np.ndarray] = None
        self.current_frame: int = 0
        self.fs: Optional[float] = None
        self.use_time_axis: bool = False
        self.current_fft_index: int = 0

        # --- Crop state
        self.crop_roi: Optional[pg.ROI] = None
        self._backup_loaded_data: Optional[np.ndarray] = None
        self._backup_original_data: Optional[np.ndarray] = None
        self._last_crop_was_ellipse: bool = False  # True when circle mode active
        self._circle_update_lock: bool = False     # prevent re-entrant ROI updates

        # Terminal window (redirect stdout/stderr)
        self.terminal_window = TerminalWindow(self)
        sys.stdout = self.terminal_window  # type: ignore[assignment]
        sys.stderr = self.terminal_window  # type: ignore[assignment]

        self._init_menu()
        self._init_ui()

    # ----- Window utilities

    def fit_to_screen(self) -> None:
        """Resize and center the window on the current screen."""
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

        # Progress bar (kept hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vlayout.addWidget(self.progress_bar)

        # Image viewer row
        layout = QHBoxLayout()
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.show()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.show()
        self.image_view.timeLine.hide()
        self.image_view.getView().setAspectLocked(True)
        layout.addWidget(self.image_view)
        vlayout.addLayout(layout)

        # Bottom controls
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

        # Time axis toggle
        self.axis_toggle = QPushButton("Show Time [s]")
        self.axis_toggle.setCheckable(True)
        self.axis_toggle.toggled.connect(self.toggle_time_axis)
        bottom.addWidget(self.axis_toggle)

        # --- Crop controls
        self.add_rect_roi_btn = QPushButton("Add Rect ROI")
        self.add_rect_roi_btn.setToolTip("Add a rectangular ROI to crop the stack")
        self.add_rect_roi_btn.clicked.connect(self.add_rect_roi)
        self.add_rect_roi_btn.setEnabled(False)
        bottom.addWidget(self.add_rect_roi_btn)

        self.add_circle_roi_btn = QPushButton("Add Circle ROI")
        self.add_circle_roi_btn.setToolTip("Add a circular ROI to crop the stack (forced 1:1)")
        self.add_circle_roi_btn.clicked.connect(self.add_circle_roi)
        self.add_circle_roi_btn.setEnabled(False)
        bottom.addWidget(self.add_circle_roi_btn)

        self.apply_crop_btn = QPushButton("Apply Crop")
        self.apply_crop_btn.setToolTip("Apply the current ROI as a spatial crop to the entire stack")
        self.apply_crop_btn.setEnabled(False)
        self.apply_crop_btn.clicked.connect(self.apply_spatial_crop)
        bottom.addWidget(self.apply_crop_btn)

        self.remove_roi_btn = QPushButton("Remove ROI")
        self.remove_roi_btn.setToolTip("Remove the current ROI overlay")
        self.remove_roi_btn.setEnabled(False)
        self.remove_roi_btn.clicked.connect(self.remove_crop_roi)
        bottom.addWidget(self.remove_roi_btn)

        self.undo_crop_btn = QPushButton("Undo Crop")
        self.undo_crop_btn.setToolTip("Restore the dataset from before the last crop")
        self.undo_crop_btn.setEnabled(False)
        self.undo_crop_btn.clicked.connect(self.undo_spatial_crop)
        bottom.addWidget(self.undo_crop_btn)

        # NEW: Set Circle by numbers
        self.set_circle_btn = QPushButton("Set Circle (cx, cy, d)")
        self.set_circle_btn.setToolTip("Type center X/Y and diameter to place a precise circle ROI")
        self.set_circle_btn.setEnabled(False)
        self.set_circle_btn.clicked.connect(self.set_circle_by_numbers)
        bottom.addWidget(self.set_circle_btn)

        # Circle ROI info label (center & diameter)
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

        # Show animated progress bar
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.setFormat("Loading data...")
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        # Use a QTimer to keep UI alive during loading
        timer = QTimer()
        timer.timeout.connect(QApplication.processEvents)
        timer.start(50)  # refresh every 50 ms

        try:
            data = io.load_ir_data(Path(file_path))
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            data = data.transpose(0, 2, 1)

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
            self.set_circle_btn.setEnabled(True)

        except Exception as e:
            print(f"Error loading data: {e}")
        finally:
            timer.stop()
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)


    def _on_data_loaded(self, data, error):
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        if error:
            print(f"Error loading data: {error}")
            return

        # ✅ All UI updates happen here in the main thread
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
        self.set_circle_btn.setEnabled(True)

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

        if self.use_time_axis and self.fs:
            xvals = np.arange(self.loaded_data.shape[0]) / self.fs
        else:
            xvals = np.arange(self.loaded_data.shape[0])

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
        self.loaded_data = self.loaded_data[start : end + 1]
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
        self.image_view.setImage(
            self.loaded_data, xvals=np.arange(self.loaded_data.shape[0]), autoLevels=True
        )
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

        # Ask if segmentation is desired
        segment_choice, ok = QInputDialog.getItem(
            self,
            "FFT Segmentation",
            "Do you want to segment the time history?",
            ["No", "Yes"],
            0,
            False
        )
        if not ok:
            return

        do_segment = (segment_choice == "Yes")
        n_segments = 1
        overlap_percent = 0.0

        if do_segment:
            n_segments, ok = QInputDialog.getInt(
                self,
                "Number of Segments",
                "Enter number of segments:",
                4,
                1,
                100,
                1
            )
            if not ok:
                return

            overlap_percent, ok = QInputDialog.getDouble(
                self,
                "Overlap Percentage",
                "Enter overlap percentage (0-90):",
                50.0,
                0.0,
                90.0,
                1
            )
            if not ok:
                return

        # Ask if Hann window should be applied
        window_choice, ok = QInputDialog.getItem(
            self,
            "Windowing",
            "Apply Hann window?",
            ["No", "Yes"],
            1,
            False
        )
        if not ok:
            return
        use_hann = (window_choice == "Yes")

        # Compute segment length and start indices
        if do_segment:
            seg_len = n_frames // n_segments
            if seg_len < 2:
                print("Too many segments for the available frames.")
                return
            step = int(seg_len * (1 - overlap_percent / 100))
            if step <= 0:
                step = 1
            starts = list(range(0, n_frames - seg_len + 1, step))
        else:
            seg_len = n_frames
            starts = [0]

        n_freqs = seg_len // 2 + 1
        fft_freqs = np.fft.rfftfreq(seg_len, d=1 / fs)

        # Window and normalization factor
        if use_hann:
            window = np.hanning(seg_len)
            U = np.sum(window ** 2) / seg_len  # energy correction
            scale = 2.0 / (seg_len * U)
        else:
            window = np.ones(seg_len)
            scale = 2.0 / seg_len

        # Progress dialog
        progress = QProgressDialog("Computing FFT...", "Cancel", 0, h, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        fft_mag = np.zeros((n_freqs, h, w), dtype=np.float32)
        fft_phase = np.zeros((n_freqs, h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                signal = data[:, i, j]
                segment_mags = []
                segment_phases = []

                for start in starts:
                    seg = signal[start:start + seg_len]
                    if len(seg) < seg_len:
                        continue
                    seg = seg - seg.mean()
                    seg = seg * window
                    fft_result = np.fft.rfft(seg)
                    segment_mags.append(np.abs(fft_result) * scale)
                    segment_phases.append(np.angle(fft_result))

                if segment_mags:
                    fft_mag[:, i, j] = np.mean(segment_mags, axis=0)
                    fft_phase[:, i, j] = np.mean(segment_phases, axis=0)

            progress.setValue(i + 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                print("FFT cancelled.")
                return

        progress.close()

        # Spectrum callback for ROI
        def spectrum_callback(mean_signal: np.ndarray):
            if do_segment:
                segment_mags = []
                for start in starts:
                    seg = mean_signal[start:start + seg_len]
                    if len(seg) < seg_len:
                        continue
                    seg = seg - seg.mean()
                    seg = seg * window
                    fft_result = np.fft.rfft(seg)
                    segment_mags.append(np.abs(fft_result) * scale)
                spectrum = np.mean(segment_mags, axis=0)
            else:
                sig = mean_signal - mean_signal.mean()
                sig = sig * window
                spectrum = np.abs(np.fft.rfft(sig)) * scale
            return fft_freqs, spectrum

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

        # Top row viewers
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

        # Spectrum plot
        spectrum_plot = pg.PlotWidget()
        spectrum_plot.setLabel("bottom", "Frequency [Hz]")
        spectrum_plot.setLabel("left", "Amplitude")
        vbox.addWidget(spectrum_plot)

        spectrum_curve = spectrum_plot.plot([], pen="y")
        cursor_line = pg.InfiniteLine(angle=90, movable=True, pen="r")
        spectrum_plot.addItem(cursor_line)

        roi = pg.RectROI([30, 30], [40, 40], pen=pg.mkPen("r", width=2))
        raw_view.addItem(roi)

        # Preferred minimum sizes
        raw_view.setMinimumSize(400, 400)
        mag_view.setMinimumSize(400, 400)
        ph_view.setMinimumSize(400, 400)
        spectrum_plot.setMinimumHeight(150)

        # Save button row (optional for FFT)
        if enable_save_mag:
            save_row = QHBoxLayout()
            vbox.addLayout(save_row)

            save_btn = QPushButton("Save Mag/Phase as separate .npy")
            save_btn.setToolTip(
                "Choose a base name once; saves ..._mag.npy, ..._phase.npy"
            )
            save_row.addWidget(save_btn)
            save_row.addStretch(1)

            def _save_many():
                # 1) Ask for a base filename once
                #    We propose 'fft' so you'll get fft_mag.npy, fft_phase.npy
                path, _ = QFileDialog.getSaveFileName(
                    dlg,
                    "Save FFT outputs (base name)",
                    "fft.npy",                 # default suggestion; we will replace suffix
                    "NumPy binary (*.npy)"
                )
                if not path:
                    return

                # Normalize: strip .npy if the user typed it
                from pathlib import Path
                base = Path(path)
                if base.suffix.lower() == ".npy":
                    base = base.with_suffix("")  # remove .npy

                # 2) Build filenames
                mag_path   = base.with_name(base.name + "_mag.npy")
                phase_path = base.with_name(base.name + "_phase.npy")

                try:
                    # 3) Save magnitude & phase as separate files
                    np.save(str(mag_path), mag_data)           # (n_freqs, H, W) float32
                    np.save(str(phase_path), phase_data)       # (n_freqs, H, W) float32 (degrees)
                    print(f"Saved: {mag_path}")
                    print(f"Saved: {phase_path}")

                    # Optionally also save freqs as its own .npy for convenience:
                    freqs_path = base.with_name(base.name + "_freqs.npy")
                    np.save(str(freqs_path), freq_vector)
                    print(f"Saved: {freqs_path}")

                except Exception as e:
                    print(f"Failed to save FFT outputs: {e}")

            save_btn.clicked.connect(_save_many)
                # -- update helpers

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

                # NOTE: pos returns (x, y); maintain original mapping
                y0, x0 = math.floor(pos[0]), math.floor(pos[1])
                y1, x1 = math.ceil(pos[0] + w_), math.ceil(pos[1] + h_)

                # Clamp coordinates
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(original_data.shape[2], x1)
                y1 = min(original_data.shape[1], y1)

                if x1 <= x0 or y1 <= y0:
                    print("ROI outside bounds or zero area.")
                    return

                roi_data = original_data[:, y0:y1, x0:x1]
                mean_signal = roi_data.mean(axis=(1, 2))
                freqs, spectrum = spectrum_callback(mean_signal)
                spectrum_curve.setData(freqs, spectrum)
                spectrum_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                spectrum_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)

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

        # Frequency sweep vector (unchanged)
        freq_vector = np.linspace(0, fs / 2, 200)  # 200 points up to Nyquist
        n_freqs = len(freq_vector)

        # Time vector
        t = np.arange(N) / fs

        # Preallocate
        mag_data = np.empty((n_freqs, h, w), dtype=np.float32)
        phase_data = np.empty((n_freqs, h, w), dtype=np.float32)

        # Progress dialog
        progress = QProgressDialog("Computing Lock-in Correlation...", "Cancel", 0, n_freqs, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        # Loop over frequencies (preserved algorithm)
        for k, fl in enumerate(freq_vector):
            sine = np.sin(2 * np.pi * fl * t)
            cosine = np.cos(2 * np.pi * fl * t)
            sine_3d = sine[:, None, None]
            cosine_3d = cosine[:, None, None]

            X = (2 / N) * np.sum(data * cosine_3d, axis=0)
            Y = (2 / N) * np.sum(data * sine_3d, axis=0)
            mag_data[k] = np.sqrt(X ** 2 + Y ** 2)
            phase_data[k] = np.degrees(np.arctan2(Y, X))

            progress.setValue(k + 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                print("Lock-in correlation cancelled.")
                return

        progress.close()

        # ROI callback for spectrum
        def spectrum_callback(mean_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            spectrum_list = []
            for fl in freq_vector:
                sine = np.sin(2 * np.pi * fl * t)
                cosine = np.cos(2 * np.pi * fl * t)
                X_roi = (2 / N) * np.sum(mean_signal * cosine)
                Y_roi = (2 / N) * np.sum(mean_signal * sine)
                spectrum_list.append(np.sqrt(X_roi ** 2 + Y_roi ** 2))
            return freq_vector, np.array(spectrum_list)

        self._show_frequency_analysis_viewer(
            data, freq_vector, mag_data, phase_data, spectrum_callback, enable_save_mag=False
        )

    def _show_time_analysis_viewer(self, data: np.ndarray, fs: float) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Time Domain Analysis")
        dlg.resize(self.size())
        layout = QVBoxLayout(dlg)

        # Top row: video viewer
        top = QHBoxLayout()
        layout.addLayout(top)

        raw_view = pg.ImageView()
        raw_view.setImage(data, autoLevels=True)
        raw_view.ui.menuBtn.hide()
        raw_view.ui.roiBtn.hide()
        top.addWidget(raw_view)

        # Bottom: time-series plot
        time_plot = pg.PlotWidget()
        time_plot.setLabel("bottom", "Time [s]")
        time_plot.setLabel("left", "Intensity")
        layout.addWidget(time_plot)

        # Storage for ROIs and their plots
        roi_list: list[pg.ROI] = []
        curves: list[pg.PlotDataItem] = []

        # Controls: toggles for processing
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
            """Apply processing options to signal (detrend, bandpass, smooth)."""
            out = sig.copy()
            if detrend_cb.isChecked():
                # Linear detrend via 1st-order poly fit
                out = out - np.polyval(np.polyfit(t, out, 1), t)
            if filter_cb.isChecked():
                # Note: normalized Wn values as in original; keep behavior
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

                # Clamp to bounds
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(data.shape[2], x1)
                y1 = min(data.shape[1], y1)
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

        # Start with one ROI
        add_roi()

        # Ctrl-click on video to add ROIs (unchanged UX)
        raw_view.scene.sigMouseClicked.connect(
            lambda ev: add_roi() if ev.modifiers() & Qt.ControlModifier else None
        )

        # Update when toggles are clicked
        detrend_cb.clicked.connect(update_all_traces)
        filter_cb.clicked.connect(update_all_traces)
        smooth_cb.clicked.connect(update_all_traces)

        dlg.exec_()

    # ---- Crop helpers -------------------------------------------------

    def _get_image_item(self):
        """Get the underlying ImageItem from ImageView."""
        img_item = getattr(self.image_view, "imageItem", None)
        if img_item is None and hasattr(self.image_view, "getImageItem"):
            img_item = self.image_view.getImageItem()
        return img_item

    def _ensure_no_existing_roi(self) -> None:
        """Remove any existing crop ROI from the view."""
        if self.crop_roi is not None:
            try:
                self.image_view.removeItem(self.crop_roi)
            except Exception:
                pass
            self.crop_roi = None
        self.apply_crop_btn.setEnabled(False)
        self.remove_roi_btn.setEnabled(False)

    def add_rect_roi(self) -> None:
        """Add a rectangular ROI to the image view."""
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
        self._last_crop_was_ellipse = False
        self.apply_crop_btn.setEnabled(True)
        self.remove_roi_btn.setEnabled(True)

    def add_circle_roi(self) -> None:
        """Add a circular ROI and enforce 1:1 aspect during any resize/move."""
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
        # Force perfect circle as user interacts
        self.crop_roi.sigRegionChanged.connect(self._enforce_circular_roi)
        self._enforce_circular_roi()  # ensure initial 1:1
        self._last_crop_was_ellipse = True
        self.apply_crop_btn.setEnabled(True)
        self.remove_roi_btn.setEnabled(True)
        # Update label initially
        self._update_circle_info_label()

    def _enforce_circular_roi(self) -> None:
        """Keep the EllipseROI strictly circular (1:1), preserving its center."""
        if self.crop_roi is None or self._circle_update_lock:
            return
        try:
            self._circle_update_lock = True
            pos = self.crop_roi.pos()    # (x, y)
            size = self.crop_roi.size()  # (w, h)
            w = float(size[0])
            h = float(size[1])
            if w <= 0 or h <= 0:
                return
            s = min(w, h)  # target side
            cx = float(pos[0]) + w / 2.0
            cy = float(pos[1]) + h / 2.0
            new_pos = [cx - s / 2.0, cy - s / 2.0]
            self.crop_roi.setPos(new_pos, finish=False)
            self.crop_roi.setSize([s, s], finish=False)
        finally:
            self._circle_update_lock = False
        # Update readout after enforcing circle
        self._update_circle_info_label()

    def _update_circle_info_label(self) -> None:
        """Show circle center (x,y) and diameter (px) in data coordinates."""
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
        """Open a dialog to type center and diameter, then place a circle ROI."""
        if self.loaded_data is None:
            print("No video loaded.")
            return
        _, H, W = self.loaded_data.shape

        # Prefill with current circle if available; else image center & half min dimension
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

        # Clamp so the ROI stays inside the image
        d = max(1, min(d, min(W, H)))
        half = d / 2.0
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        # Adjust to keep inside bounds
        x0 = max(0, min(x0, W - d))
        y0 = max(0, min(y0, H - d))

        # Create or reuse the EllipseROI at the specified position/size
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
        """Remove only the ROI overlay (no data changes)."""
        self._ensure_no_existing_roi()
        self.circle_info_label.setText("Circle ROI: —")

    def _roi_bounds_clamped(self) -> Optional[tuple[int, int, int, int]]:
        """
        Return (x0, y0, x1, y1) integer bounds of current ROI in array coordinates,
        using ROI.getArraySlice for accurate mapping (accounts for transforms).
        """
        if self.loaded_data is None or self.crop_roi is None:
            return None

        # Current 2D frame (t, y, x) -> (y, x)
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

            # Clamp
            y0 = max(0, min(y0, img2d.shape[0]))
            y1 = max(0, min(y1, img2d.shape[0]))
            x0 = max(0, min(x0, img2d.shape[1]))
            x1 = max(0, min(x1, img2d.shape[1]))

            if x1 <= x0 or y1 <= y0:
                return None

            return x0, y0, x1, y1
        except Exception as e:
            print(f"Failed to compute ROI slice: {e}")
            return None

    def apply_spatial_crop(self) -> None:
        """
        Apply the current ROI to crop the stack (t, y, x) using precise bounds
        from ROI.getArraySlice. Rect ROI -> rectangular crop.
        Circle ROI -> rectangular crop + zero outside the circle (keeps analyses intact).
        """
        if self.loaded_data is None or self.crop_roi is None:
            print("No ROI to apply.")
            return

        bounds = self._roi_bounds_clamped()
        if bounds is None:
            print("ROI out of bounds or zero area.")
            return

        x0, y0, x1, y1 = bounds

        # Save current histogram levels so display doesn't "binarize" after masking
        hw = self.image_view.getHistogramWidget() if hasattr(self.image_view, "getHistogramWidget") \
            else self.image_view.ui.histogram
        prev_levels = hw.getLevels()

        # Backups for Undo Crop
        self._backup_loaded_data = self.loaded_data
        self._backup_original_data = self.original_data

        # Crop current and original datasets (t, y, x)
        cropped_loaded = self.loaded_data[:, y0:y1, x0:x1]
        cropped_original = None if self.original_data is None else self.original_data[:, y0:y1, x0:x1]

        if self._last_crop_was_ellipse:
            # Build a circular mask in the cropped rectangle, centered in the box
            h = y1 - y0
            w = x1 - x0
            r = min(h, w) / 2.0
            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
            yy, xx = np.ogrid[:h, :w]
            circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)

            # Apply mask as zeros outside the circle (preserve dtype)
            mask_loaded = circle.astype(cropped_loaded.dtype, copy=False)[None, :, :]
            cropped_loaded = cropped_loaded * mask_loaded

            if cropped_original is not None:
                mask_original = circle.astype(cropped_original.dtype, copy=False)[None, :, :]
                cropped_original = cropped_original * mask_original

        # Commit
        self.loaded_data = cropped_loaded
        if cropped_original is not None:
            self.original_data = cropped_original

        # Clean up ROI and refresh UI
        self._ensure_no_existing_roi()
        self.current_frame = 0
        self._init_slider()
        self.update_viewer()

        # Restore histogram levels so zeros outside circle don't skew contrast
        hw.setLevels(*prev_levels)

        # Enable Undo Crop
        self.undo_crop_btn.setEnabled(True)
        print(f"Cropped to x:[{x0},{x1}), y:[{y0},{y1}) ; shape -> {self.loaded_data.shape}")

    def undo_spatial_crop(self) -> None:
        """Restore dataset to state before last crop."""
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

    # Look for an icon next to this file
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

    QTimer.singleShot(1400, _show_main)
    sys.exit(app.exec_())


if __name__ == "__main__":
    show_splash_then_main()