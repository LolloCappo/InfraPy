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
from PyQt5 import QtCore
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
from PyQt5.QtWidgets import QProgressDialog, QApplication
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

        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        width = int(screen_size.width() * 0.8)
        height = int(screen_size.height() * 0.8)
        self.resize(width, height)
        self.move(
            screen_size.left() + (screen_size.width() - width) // 2,
            screen_size.top() + (screen_size.height() - height) // 2
)
        self.loaded_data = None
        self.original_data = None
        self.current_frame = 0

        self.fs = None
        self.use_time_axis = False

        self.init_menu()
        self.init_ui()

    def fit_to_screen(self):
        screen = QApplication.screenAt(self.pos())
        if screen:
            screen_size = screen.availableGeometry()
            width = int(screen_size.width() * 0.8)
            height = int(screen_size.height() * 0.8)
            self.resize(width, height)
            self.move(
                screen_size.left() + (screen_size.width() - width) // 2,
                screen_size.top() + (screen_size.height() - height) // 2
            )

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

        resize_action = QAction("Fit to Screen", self)
        resize_action.triggered.connect(self.fit_to_screen)
        view_menu.addAction(resize_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        radiation_action = QAction("Radiation", self)
        analysis_menu.addAction(radiation_action)
        radiation_action.triggered.connect(lambda: self.show_analysis_window("Radiation Analysis", "Radiation tools go here."))
        time_domain_action = QAction("Time domain", self)
        time_domain_action.triggered.connect(lambda: self.show_analysis_window("Time Domain Analysis", "Time domain tools go here."))
        analysis_menu.addAction(time_domain_action)
  
        freq_domain_menu = analysis_menu.addMenu("Frequency domain")
        fft_action = QAction("Fast Fourier Transform", self)
        fft_action.triggered.connect(self.run_fft_analysis)
        freq_domain_menu.addAction(fft_action)

        corr_action = QAction("Lock-In Correlation", self)
        corr_action.triggered.connect(self.run_correlation_analysis)
        freq_domain_menu.addAction(corr_action)

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

        # Time axis toggle
        self.axis_toggle = QPushButton("Show Time [s]")
        self.axis_toggle.setCheckable(True)
        self.axis_toggle.toggled.connect(self.toggle_time_axis)
        bottom.addWidget(self.axis_toggle)
    
    def toggle_time_axis(self, checked):
        self.use_time_axis = checked
        self.axis_toggle.setText("Show Frame [#]" if checked else "Show Time [s]")
        self.update_frame_display()

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

            # Prompt for sampling frequency
            fs, ok = QInputDialog.getDouble(self, "Sampling Frequency", "Enter sampling frequency (Hz):", 50.0, 0.01, 1e6, 2)
            if not ok:
                fs = 1.0  # Default to 1 to avoid divide-by-zero
            self.fs = fs

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
        if self.use_time_axis and self.fs:
            xvals = np.arange(self.loaded_data.shape[0]) / self.fs
        else:
            xvals = np.arange(self.loaded_data.shape[0])
        self.image_view.setImage(self.loaded_data, xvals=xvals, autoLevels=False)

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

    def run_fft_analysis(self):
        if self.loaded_data is None:
            print("No video loaded.")
            return
        if not hasattr(self, 'fs') or self.fs is None:
            print("Sampling frequency not set.")
            return

        data = self.loaded_data
        fs = self.fs
        n_frames = data.shape[0]

        # Compute FFT over time axis
        def compute_fft_with_progress(data, fs, parent=None):
            n_frames, height, width = data.shape
            n_freqs = n_frames // 2 + 1
            fft_data = np.empty((n_freqs, height, width), dtype=np.complex64)

            progress = QProgressDialog("Computing FFT...", "Cancel", 0, height, parent)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setMinimumDuration(0)

            for y in range(height):
                QApplication.processEvents()  # allow UI update
                if progress.wasCanceled():
                    return None  # You can handle cancel gracefully
                fft_data[:, y, :] = np.fft.rfft(data[:, y, :] - np.mean(data[:, y, :]), axis=0)
                progress.setValue(y)

            progress.setValue(height)
            fft_freqs = np.fft.rfftfreq(n_frames, d=1/fs)
            return fft_data, fft_freqs
        
        fft_freqs = np.fft.rfftfreq(n_frames, d=1/fs)

        fft_data, fft_freqs = compute_fft_with_progress(data, fs, parent=self)
        if fft_data is None:
            print("FFT canceled.")
            return
        
        fft_mag = np.abs(fft_data) * 2/n_frames
        fft_phase = np.angle(fft_data)

        # Store for interaction
        self.current_fft_index = 0
        self.fft_freqs = fft_freqs

        # Dialog setup
        dlg = QDialog(self)
        dlg.setWindowTitle("FFT Analysis")
        dlg.resize(1600, 800)
        layout = QVBoxLayout(dlg)

        # Top row: 3 ImageView widgets
        top_row = QHBoxLayout()
        layout.addLayout(top_row)

        raw_view = pg.ImageView()
        raw_view.setImage(data, autoLevels=True)
        raw_view.ui.menuBtn.hide()
        top_row.addWidget(raw_view)

        mag_view = pg.ImageView()
        mag_view.setImage(fft_mag, autoLevels=True)
        mag_view.ui.menuBtn.hide()
        mag_view.ui.roiBtn.hide()
        top_row.addWidget(mag_view)

        ph_view = pg.ImageView()
        ph_view.setImage(fft_phase, autoLevels=True)
        ph_view.ui.menuBtn.hide()
        ph_view.ui.roiBtn.hide()
        top_row.addWidget(ph_view)

        # Bottom: spectrum plot with draggable cursor
        spectrum_plot = pg.PlotWidget()
        spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        spectrum_plot.setLabel("left", "Amplitude")
        layout.addWidget(spectrum_plot)
        spectrum_curve = spectrum_plot.plot([], pen='y')

        cursor_line = pg.InfiniteLine(angle=90, movable=True, pen='r')
        spectrum_plot.addItem(cursor_line)

        # ROI on original video
        roi = pg.RectROI([30, 30], [40, 40], pen='r')
        raw_view.addItem(roi)

        # Function to update spectrum from ROI
        def update_spectrum():
            try:
                region = np.array([
                    roi.getArrayRegion(data[i], raw_view.imageItem)
                    for i in range(data.shape[0])
                ])
                if region.ndim != 3:
                    return
                mean_signal = region.mean(axis=(1, 2))  # shape: (n_frames,)
                spectrum = np.abs(np.fft.rfft(mean_signal)) * 2/data.shape[0]
                spectrum_curve.setData(fft_freqs, spectrum)

                # Move cursor to current index
                if self.current_fft_index < len(fft_freqs):
                    cursor_line.setPos(fft_freqs[self.current_fft_index])
                    update_fft_images_from_cursor()
            except Exception as e:
                print(f"FFT spectrum update failed: {e}")

        # Function to update mag/phase maps from frequency selection
        def update_fft_images_from_cursor():
            freq_pos = cursor_line.value()
            idx = (np.abs(fft_freqs - freq_pos)).argmin()
            self.current_fft_index = idx
            if 0 <= idx < fft_mag.shape[0]:
                mag_slice = fft_mag[idx]
                ph_slice = fft_phase[idx]
                mag_view.setImage(mag_slice[np.newaxis, :, :], autoLevels=False)
                ph_view.setImage(ph_slice[np.newaxis, :, :], autoLevels=False)

        # Connect callbacks
        roi.sigRegionChanged.connect(update_spectrum)
        cursor_line.sigPositionChanged.connect(update_fft_images_from_cursor)

        # Initial draw
        update_spectrum()

        dlg.exec_()


        # Synchronize frequency slider with mag/phase viewers
        def on_frame_changed(index):
            if 0 <= index < fft_mag.shape[0]:
                mag_slice = fft_mag[index]
                ph_slice = fft_phase[index]
                mag_view.setImage(mag_slice[np.newaxis, :, :], autoLevels=False)
                ph_view.setImage(ph_slice[np.newaxis, :, :], autoLevels=False)

        raw_view.timeLine.sigPositionChanged.connect(lambda: on_frame_changed(int(raw_view.currentIndex)))
        on_frame_changed(0)

        dlg.exec_()

    def run_correlation_analysis(self):
        if self.loaded_data is None:
            print("No video loaded.")
            return
        if not hasattr(self, 'fs') or self.fs is None:
            print("Sampling frequency not set.")
            return
    
        print(f"Running Correlation analysis with fs={self.fs}")
        # TODO: Implement Correlation analysis & visualization

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
