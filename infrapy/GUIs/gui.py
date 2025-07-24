import sys
import os
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QProgressBar, QSplashScreen, QAction
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QIcon
import pyqtgraph as pg

# Adjust path to access infrapy.io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from infrapy import io  # Ensure this is your actual IR data loading module


class IRViewerPG(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("INFRAPY")
        self.setWindowIcon(QIcon('icon.png'))
        self.resize(1300, 800)

        self.loaded_data = None
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

    def init_ui(self):
        central = QWidget()
        vlayout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Progress bar below menu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vlayout.addWidget(self.progress_bar)

        # Viewer + label
        layout = QHBoxLayout()
        vlayout.addLayout(layout)

        # ImageView
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.getView().setAspectLocked(True)
        layout.addWidget(self.image_view)

        # Right panel
        controls = QVBoxLayout()
        controls.addStretch()
        layout.addLayout(controls)

    def load_ir_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open IR File", "",
            "IR Files (*.tif *.tiff *.csv *.sfmov *.npy *.npz);;All Files (*)"
        )
        if not file_path:
            return

        path = Path(file_path)
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            QApplication.processEvents()

            self.loaded_data = io.load_ir_data(path)
            self.progress_bar.setValue(50)

            if self.loaded_data.ndim == 2:
                self.loaded_data = self.loaded_data[np.newaxis, :, :]

            # Transpose to correct orientation
            self.loaded_data = self.loaded_data.transpose(0, 2, 1)

            self.image_view.setImage(self.loaded_data, xvals=np.arange(self.loaded_data.shape[0]))

            self.progress_bar.setValue(100)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def set_colormap(self, cmap_name):
        try:
            lut = pg.colormap.get(cmap_name).getLookupTable(0.0, 1.0, 256)
            self.image_view.setColorMap(pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut))
        except Exception as e:
            print(f"Error setting colormap {cmap_name}: {e}")


def show_splash_then_main():
    app = QApplication(sys.argv)

    splash = None
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        pixmap = QPixmap(str(logo_path)).scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

    def launch_main():
        app.main_window = IRViewerPG()
        app.main_window.show()
        if splash:
            splash.finish(app.main_window)

    QTimer.singleShot(1500, launch_main)
    sys.exit(app.exec_())


if __name__ == "__main__":
    show_splash_then_main()
