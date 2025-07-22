import napari
from magicgui import magicgui
from qtpy.QtWidgets import (
    QApplication, QFileDialog, QProgressBar, QWidget, QVBoxLayout, QSplashScreen
)
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from skimage.draw import polygon
import time

# === Initialize QApplication ===
app = QApplication.instance()
if app is None:
    app = QApplication([])

# === Load splash logo safely ===
logo_path = Path(__file__).parent / "logo_loading.png"
splash_pix = QPixmap(str(logo_path))
splash_pix = splash_pix.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
# === Show splash screen ===
splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
splash.show()
app.processEvents()

# Simulate loading time
time.sleep(1.5)

# === Adjust Python path to find your infrapy module ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from infrapy import io  # Make sure this path is correct for your project

# === Create Napari viewer ===
viewer = napari.Viewer(title="InfraPy - Infrared Image Processing")
viewer.theme = "light"

# === Close splash after GUI is ready ===
splash.finish(viewer.window._qt_window)

# === Progress bar ===
progress_bar = QProgressBar()
progress_bar.setRange(0, 100)
progress_bar.setValue(0)
progress_bar.setVisible(False)

# === Global variables ===
loaded_data = None
fig, ax = plt.subplots()
canvas_widget = None


# === ROI statistics ===
def extract_roi_stats(shapes_layer, data):
    stats = {}
    for i, shape in enumerate(shapes_layer.data):
        try:
            if shape.shape[0] < 3:
                print(f"⚠️ Skipping ROI {i+1}: fewer than 3 points.")
                continue

            # Ensure coordinates are within bounds
            rr, cc = polygon(shape[:, 0], shape[:, 1], shape=data.shape[1:])
            mask = np.zeros(data.shape[1:], dtype=bool)
            mask[rr, cc] = True

            values = [
                data[frame][mask].mean() if np.any(mask) else np.nan
                for frame in range(data.shape[0])
            ]
            stats[i] = values

        except Exception as e:
            print(f"❌ Failed to process ROI {i+1}: {e}")
    return stats

def update_plot(stats):
    ax.clear()
    for roi_idx, values in stats.items():
        ax.plot(values, label=f"ROI {roi_idx + 1}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Pixel Value")
    ax.set_title("Mean Pixel Value Over Time")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    canvas_widget.draw_idle()


def on_shapes_change(event):
    if loaded_data is None:
        return
    shapes_layer = event.source
    stats = extract_roi_stats(shapes_layer, loaded_data)
    update_plot(stats)


# === Matplotlib plot panel ===
class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        global canvas_widget
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_widget = self.canvas

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


# === Data loader widget ===
@magicgui(call_button="Load IR Data")
def data_loader_widget():
    global loaded_data

    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setNameFilter("IR Files (*.tif *.tiff *.csv *.sfmov *.npy *.npz);;All files (*)")

    if dialog.exec_():
        selected = dialog.selectedFiles()[0]
        path = Path(selected)

        try:
            progress_bar.setVisible(True)
            progress_bar.setValue(10)

            data = io.load_ir_data(path)

            progress_bar.setValue(90)

            name = path.stem if path.is_file() else path.name
            if data.shape[0] == 1:
                viewer.add_image(data[0], name=name, colormap="inferno")
            else:
                viewer.add_image(data, name=name, colormap="inferno")

            loaded_data = data

            if "ROIs" not in viewer.layers:
                shapes = viewer.add_shapes(name="ROIs", shape_type="polygon",
                                           edge_color="yellow", face_color="transparent", opacity=0.5)
                shapes.events.data.connect(on_shapes_change)
            else:
                viewer.layers["ROIs"].data = []

            progress_bar.setValue(100)

        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
        finally:
            progress_bar.setVisible(False)
            progress_bar.setValue(0)


# === Add widgets to Napari GUI ===
plot_widget = PlotWidget()
viewer.window.add_dock_widget(data_loader_widget, area='right', name="Load IR Data")
viewer.window.add_dock_widget(plot_widget, area='right', name="ROI Statistics")
viewer.window.add_dock_widget(progress_bar, area='bottom', name="Loading Progress")

# === Start Napari ===
napari.run()
