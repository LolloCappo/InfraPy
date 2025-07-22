import napari
from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog, QProgressBar
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from infrapy import io

# Create the viewer
viewer = napari.Viewer()
viewer.window._qt_viewer.setWindowTitle("InfraPy")
viewer.theme = "light"

# Add a progress bar below the dock
progress_bar = QProgressBar()
progress_bar.setRange(0, 100)
progress_bar.setValue(0)
progress_bar.setVisible(False)  # Only show during loading

@magicgui(call_button="Load IR Data")
def data_loader_widget():
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

            progress_bar.setValue(100)

        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
        finally:
            progress_bar.setVisible(False)
            progress_bar.setValue(0)

# Add the widget and progress bar to the viewer
viewer.window.add_dock_widget(data_loader_widget, area='right', name="Load IR Data")
viewer.window.add_dock_widget(progress_bar, area='bottom', name="Loading Progress")

# Run the app
napari.run()
