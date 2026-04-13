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

Module layout
-------------
gui.py      — this file: IRViewerPG shell + bootstrap
style.py    — DARK_QSS stylesheet and shared constants
widgets.py  — QRangeSlider, TerminalWindow, CircleParamsDialog, DataLoaderThread
analysis.py — AnalysisMixin  (FFT, lock-in, time-domain viewers)
crop.py     — CropMixin      (rect/circle ROI, apply/undo crop)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QUrl, Qt
from PyQt5.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplashScreen,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# ----------------------------------------------
# High-DPI configuration (as early as possible)
# ----------------------------------------------

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

# Make local 'infrapy' importable when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from infrapy import io  # noqa: E402
from infrapy.thermoelasticity import lock_in_analysis  # noqa: F401,E402

import pyqtgraph as pg  # noqa: E402

try:
    from .style import DARK_QSS, DEFAULT_COLORMAPS, DOC_URL
    from .widgets import QRangeSlider, TerminalWindow
    from .analysis import AnalysisMixin
    from .crop import CropMixin
except ImportError:
    # Running gui.py directly as __main__
    sys.path.insert(0, str(Path(__file__).parent))
    from style import DARK_QSS, DEFAULT_COLORMAPS, DOC_URL  # type: ignore
    from widgets import QRangeSlider, TerminalWindow  # type: ignore
    from analysis import AnalysisMixin  # type: ignore
    from crop import CropMixin  # type: ignore


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class IRViewerPG(AnalysisMixin, CropMixin, QMainWindow):
    """Main InfraPy viewer window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("InfraPy")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "icon.png")))

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

        # ---- Crop state (used by CropMixin)
        self.crop_roi: Optional[pg.ROI] = None
        self._backup_loaded_data: Optional[np.ndarray] = None
        self._backup_original_data: Optional[np.ndarray] = None
        self._last_crop_was_ellipse: bool = False
        self._circle_update_lock: bool = False

        # Terminal window (redirect stdout/stderr)
        self.terminal_window = TerminalWindow(self)
        sys.stdout = self.terminal_window  # type: ignore[assignment]
        sys.stderr = self.terminal_window  # type: ignore[assignment]

        self._init_menu()
        self._init_toolbar()
        self._init_ui()

    # ------------------------------------------------------------------
    # Window utilities
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _init_menu(self) -> None:
        menubar = self.menuBar()

        # File
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

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _init_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        load_action = QAction("Load Data", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_ir_data)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(DEFAULT_COLORMAPS)
        self._cmap_combo.setFixedWidth(100)
        self._cmap_combo.currentTextChanged.connect(self.set_colormap)
        toolbar.addWidget(self._cmap_combo)

    # ------------------------------------------------------------------
    # Central UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        central = QWidget()
        vlayout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vlayout.addWidget(self.progress_bar)

        # Image viewer
        layout = QHBoxLayout()
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.show()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.show()
        self.image_view.timeLine.hide()
        self.image_view.getView().setAspectLocked(True)
        layout.addWidget(self.image_view)
        vlayout.addLayout(layout)

        # ---- Playback controls row
        playback_row = QHBoxLayout()
        playback_row.setSpacing(6)

        self.trim_slider = QRangeSlider()
        self.trim_slider.setEnabled(False)
        self.trim_slider.setMinimumHeight(28)
        self.trim_slider.valueChanged.connect(self.update_frame_display)
        playback_row.addWidget(self.trim_slider, stretch=1)

        self.apply_button = QPushButton("Apply Clip")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_clipping)
        playback_row.addWidget(self.apply_button)

        self.undo_button = QPushButton("Undo Clip")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_clipping)
        playback_row.addWidget(self.undo_button)

        _sep1 = QFrame()
        _sep1.setFrameShape(QFrame.VLine)
        _sep1.setFrameShadow(QFrame.Sunken)
        playback_row.addWidget(_sep1)

        self.axis_toggle = QPushButton("Show Time [s]")
        self.axis_toggle.setCheckable(True)
        self.axis_toggle.toggled.connect(self.toggle_time_axis)
        playback_row.addWidget(self.axis_toggle)

        vlayout.addLayout(playback_row)

        # ---- Crop controls row
        crop_row = QHBoxLayout()
        crop_row.setSpacing(6)

        self.add_rect_roi_btn = QPushButton("Rect ROI")
        self.add_rect_roi_btn.setToolTip("Add a rectangular ROI to crop the stack")
        self.add_rect_roi_btn.clicked.connect(self.add_rect_roi)
        self.add_rect_roi_btn.setEnabled(False)
        crop_row.addWidget(self.add_rect_roi_btn)

        self.add_circle_roi_btn = QPushButton("Circle ROI")
        self.add_circle_roi_btn.setToolTip("Add a circular ROI to crop the stack (forced 1:1)")
        self.add_circle_roi_btn.clicked.connect(self.add_circle_roi)
        self.add_circle_roi_btn.setEnabled(False)
        crop_row.addWidget(self.add_circle_roi_btn)

        self.set_circle_btn = QPushButton("Set Circle…")
        self.set_circle_btn.setToolTip("Type center X/Y and diameter to place a precise circle ROI")
        self.set_circle_btn.setEnabled(False)
        self.set_circle_btn.clicked.connect(self.set_circle_by_numbers)
        crop_row.addWidget(self.set_circle_btn)

        _sep2 = QFrame()
        _sep2.setFrameShape(QFrame.VLine)
        _sep2.setFrameShadow(QFrame.Sunken)
        crop_row.addWidget(_sep2)

        self.apply_crop_btn = QPushButton("Apply Crop")
        self.apply_crop_btn.setToolTip("Apply the current ROI as a spatial crop to the entire stack")
        self.apply_crop_btn.setEnabled(False)
        self.apply_crop_btn.clicked.connect(self.apply_spatial_crop)
        crop_row.addWidget(self.apply_crop_btn)

        self.remove_roi_btn = QPushButton("Remove ROI")
        self.remove_roi_btn.setToolTip("Remove the current ROI overlay")
        self.remove_roi_btn.setEnabled(False)
        self.remove_roi_btn.clicked.connect(self.remove_crop_roi)
        crop_row.addWidget(self.remove_roi_btn)

        self.undo_crop_btn = QPushButton("Undo Crop")
        self.undo_crop_btn.setToolTip("Restore the dataset from before the last crop")
        self.undo_crop_btn.setEnabled(False)
        self.undo_crop_btn.clicked.connect(self.undo_spatial_crop)
        crop_row.addWidget(self.undo_crop_btn)

        crop_row.addStretch()
        vlayout.addLayout(crop_row)

        # ---- Status bar: frame counter (left) + circle info (right)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.circle_info_label = QLabel("Circle ROI: —")
        self.statusBar().addWidget(self.frame_label)
        self.statusBar().addPermanentWidget(self.circle_info_label)

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def toggle_time_axis(self, checked: bool) -> None:
        self.use_time_axis = checked
        self.axis_toggle.setText("Show Frame [#]" if checked else "Show Time [s]")
        self.update_frame_display()

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_ir_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open IR File",
            "",
            "IR Files (*.tif *.tiff *.csv *.sfmov *.npy *.npz *.hcc);;All Files (*)",
        )
        if not file_path:
            return

        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Loading data...")
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        timer = QTimer()
        timer.timeout.connect(QApplication.processEvents)
        timer.start(50)

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
            self.fs = float(fs) if ok else 1.0

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

    def _on_data_loaded(self, data, error) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        if error:
            print(f"Error loading data: {error}")
            return

        self.original_data = data
        self.loaded_data = data.copy()

        self.remove_crop_roi()
        self.circle_info_label.setText("Circle ROI: —")

        fs, ok = QInputDialog.getDouble(
            self, "Sampling Frequency", "Enter sampling frequency [Hz]:", 50.0, 0.01, 1e6, 2
        )
        self.fs = float(fs) if ok else 1.0

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

        xvals = (
            np.arange(self.loaded_data.shape[0]) / self.fs
            if self.use_time_axis and self.fs
            else np.arange(self.loaded_data.shape[0])
        )

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


# ---------------------------------------------------------------------------
# App bootstrap with splash screen
# ---------------------------------------------------------------------------

def show_splash_then_main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)

    # pyqtgraph global dark config (must be set before any pg widgets are created)
    pg.setConfigOptions(background="#1e1e1e", foreground="#c0c0c0", antialias=True)

    logo = Path(__file__).parent / "icon.png"
    splash: Optional[QSplashScreen] = None
    if logo.exists():
        pix = QPixmap(str(logo)).scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
        import time
        time.sleep(3.0)

    w = IRViewerPG()
    w.show()
    if splash is not None:
        splash.finish(w)

    sys.exit(app.exec_())


if __name__ == "__main__":
    show_splash_then_main()
