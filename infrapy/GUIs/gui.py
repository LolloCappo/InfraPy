import sys
import os
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QProgressBar, QSplashScreen, QAction, QPushButton,
    QDialog, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices, QPainter, QColor

import pyqtgraph as pg

# Adjust path to access infrapy modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from infrapy import io  # Your IR data loading module
from infrapy.thermoelasticity import lock_in_analysis


class QRangeSlider(QWidget):
    valueChanged = pyqtSignal(tuple)  # Emits (start, end) integers

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._start = 0
        self._end = 100
        self._dragging_start = False
        self._dragging_end = False
        self.setMinimumSize(150, 30)
        self.setMouseTracking(True)

    def setMinimum(self, val):
        self._min = val
        if self._start < val:
            self._start = val
        if self._end < val:
            self._end = val
        self.update()

    def setMaximum(self, val):
        self._max = val
        if self._start > val:
            self._start = val
        if self._end > val:
            self._end = val
        self.update()

    def setValue(self, val):
        start, end = val
        self._start = max(self._min, min(start, self._max))
        self._end = max(self._min, min(end, self._max))
        if self._start > self._end:
            self._start = self._end
        self.update()
        self.valueChanged.emit((self._start, self._end))

    def value(self):
        return (self._start, self._end)

    def paintEvent(self, event):
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
        handle_width = 8
        p.setBrush(QColor(50, 50, 150))
        p.drawRect(start_x - handle_width // 2, 0, handle_width, rect.height())
        p.drawRect(end_x - handle_width // 2, 0, handle_width, rect.height())

    def mousePressEvent(self, event):
        x = event.pos().x()
        rect = self.rect()
        total_range = self._max - self._min
        if total_range <= 0:
            return
        start_x = (self._start - self._min) / total_range * rect.width()
        end_x = (self._end - self._min) / total_range * rect.width()
        handle_width = 8

        if abs(x - start_x) < handle_width:
            self._dragging_start = True
        elif abs(x - end_x) < handle_width:
            self._dragging_end = True

    def mouseMoveEvent(self, event):
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

    def mouseReleaseEvent(self, event):
        self._dragging_start = False
        self._dragging_end = False


class IRViewerPG(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("INFRAPY")
        self.setWindowIcon(QIcon('infrapy/GUIs/icon.png'))
        self.resize(1300, 800)

        self.loaded_data = None
        self.original_data = None
        self.current_frame = 0

        self.init_menu()
        self.init_ui()

    def init_menu(self):
        menubar = self.menuBar()

        # File > Load
        file_menu = menubar.addMenu("File")
        load_action = QAction("Load Data", self)
        load_action.triggered.connect(self.load_ir_data)
        file_menu.addAction(load_action)

        # View > Colormap
        view_menu = menubar.addMenu("View")
        colormap_menu = view_menu.addMenu("Colormap")
        self.colormaps = ["gray", "viridis", "plasma", "inferno", "cividis"]
        for cmap in self.colormaps:
            cmap_action = QAction(cmap, self)
            cmap_action.triggered.connect(lambda checked, c=cmap: self.set_colormap(c))
            colormap_menu.addAction(cmap_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        radiation_action = QAction("Radiation", self)
        radiation_action.triggered.connect(lambda: self.show_analysis_window("Radiation Analysis", "Radiation tools go here."))
        time_domain_action = QAction("Time domain", self)
        time_domain_action.triggered.connect(lambda: self.show_analysis_window("Time Domain Analysis", "Time domain tools go here."))
        freq_domain_action = QAction("Frequency domain", self)
        freq_domain_action.triggered.connect(self.show_frequency_window)
        analysis_menu.addAction(radiation_action)
        analysis_menu.addAction(time_domain_action)
        analysis_menu.addAction(freq_domain_action)

        # Help > Documentation
        help_menu = menubar.addMenu("Help")
        doc_action = QAction("Documentation", self)
        doc_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/LolloCappo/InfraPy")))
        help_menu.addAction(doc_action)
        update_action = QAction("Check for updates", self)
        update_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/LolloCappo/InfraPy")))
        help_menu.addAction(update_action)

    def init_ui(self):
        central = QWidget()
        vlayout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Progress bar
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
        vlayout.addLayout(bottom)

    def load_ir_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open IR File", "",
            "IR Files (*.tif *.tiff *.csv *.sfmov *.npy *.npz);;All Files (*)"
        )
        if not file_path:
            return

        try:
            self.progress_bar.setVisible(True)
            QApplication.processEvents()

            data = io.load_ir_data(Path(file_path))
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            data = data.transpose(0, 2, 1)

            self.original_data = data
            self.loaded_data = data.copy()
            self.current_frame = 0

            self.update_viewer()
            self.init_slider()
            self.undo_button.setEnabled(False)

        except Exception as e:
            print(f"Error loading data: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def init_slider(self):
        n = self.loaded_data.shape[0]
        self.trim_slider.setMinimum(0)
        self.trim_slider.setMaximum(n - 1)
        self.trim_slider.setValue((0, n - 1))
        self.trim_slider.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.update_frame_display()

    def update_frame_display(self):
        start, end = self.trim_slider.value()
        if start > end:
            start, end = end, start
        if not (start <= self.current_frame <= end):
            self.current_frame = start
        self.image_view.setImage(self.loaded_data, xvals=np.arange(self.loaded_data.shape[0]), autoLevels=False)
        self.image_view.setCurrentIndex(self.current_frame)
        self.frame_label.setText(f"Frame: {self.current_frame} / {self.loaded_data.shape[0] - 1}")
        self.apply_button.setText(f"Apply Clip ({start}–{end})")

    def apply_clipping(self):
        start, end = self.trim_slider.value()
        if start > end:
            start, end = end, start
        self.loaded_data = self.loaded_data[start:end+1]
        self.current_frame = 0
        self.init_slider()
        self.update_viewer()
        self.undo_button.setEnabled(True)

    def undo_clipping(self):
        self.loaded_data = self.original_data.copy()
        self.current_frame = 0
        self.init_slider()
        self.update_viewer()
        self.undo_button.setEnabled(False)

    def update_viewer(self):
        self.image_view.setImage(self.loaded_data, xvals=np.arange(self.loaded_data.shape[0]), autoLevels=True)
        self.image_view.setCurrentIndex(self.current_frame)
        self.frame_label.setText(f"Frame: {self.current_frame} / {self.loaded_data.shape[0] - 1}")

    def set_colormap(self, cmap):
        try:
            lut = pg.colormap.get(cmap).getLookupTable(0.0, 1.0, 256)
            self.image_view.setColorMap(pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut))
        except Exception as e:
            print(f"Error setting colormap: {e}")

    def show_analysis_window(self, title, text):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(400, 200)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(text))
        dlg.exec_()

    def show_frequency_window(self):
        if self.loaded_data is None:
            print("No video loaded.")
            return
        fs, ok1 = QInputDialog.getDouble(self, "Sampling Frequency", "Enter fs [Hz]:", 50.0)
        if not ok1:
            return
        fl, ok2 = QInputDialog.getDouble(self, "Excitation Frequency", "Enter fl [Hz]:", 10.0)
        if not ok2:
            return

        mag, ph = self.run_lock_in_analysis(self.loaded_data, fs, fl)
        if mag is None or ph is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Frequency Domain Analysis")
        dlg.resize(1200, 600)
        layout = QHBoxLayout(dlg)

        mag_view = pg.ImageView()
        mag_view.setImage(mag[np.newaxis, :, :], autoLevels=True)
        mag_view.setPredefinedGradient('inferno')
        mag_view.ui.roiBtn.hide()
        mag_view.ui.menuBtn.hide()
        layout.addWidget(mag_view)

        ph_view = pg.ImageView()
        ph_view.setImage(ph[np.newaxis, :, :], autoLevels=True)
        ph_view.setPredefinedGradient('bipolar')
        ph_view.ui.roiBtn.hide()
        ph_view.ui.menuBtn.hide()
        layout.addWidget(ph_view)

        dlg.exec_()

    def run_lock_in_analysis(self, data, fs, fl):
        try:
            magnitude, phase = lock_in_analysis(data, fs, fl, method='fft', band=0.5)
            return magnitude, phase
        except Exception as e:
            print(f"Error in lock-in analysis: {e}")
            return None, None


def show_splash_then_main():
    app = QApplication(sys.argv)
    logo = Path(__file__).parent / "icon.png"
    if logo.exists():
        pix = QPixmap(str(logo)).scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
    QTimer.singleShot(1500, lambda: IRViewerPG().show() or (splash.finish(app.activeWindow()) if 'splash' in locals() else None))
    sys.exit(app.exec_())


if __name__ == "__main__":
    show_splash_then_main()
