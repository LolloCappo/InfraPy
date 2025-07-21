from nicegui import ui
from pathlib import Path
import numpy as np
import sys
from pathlib import Path

# Add project root (parent of 'gui') to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from infrapy.io import load_snapshot, load_sequence
# Adjust this import based on your actual project structure
from infrapy.io import load_snapshot, load_sequence


selected_path = None  # global for current file selection


def handle_file_selection(e):
    global selected_path
    selected_path = Path(e.name)
    file_label.set_text(f"Selected file: {selected_path.name}")


def load_file():
    if selected_path is None:
        ui.notify("Please select a file first!", color="negative")
        return

    try:
        # Try loading as sequence
        if selected_path.is_dir() or selected_path.suffix in [".tif", ".tiff", ".npy", ".npz"]:
            data = load_sequence(selected_path)
            dims = data.shape
            ui.notify(f"Sequence loaded: shape {dims}", color="positive")
        else:
            data = load_snapshot(selected_path)
            dims = data.shape
            ui.notify(f"Snapshot loaded: shape {dims}", color="positive")

    except Exception as ex:
        ui.notify(f"Error loading file: {str(ex)}", color="negative")


# UI Layout
ui.markdown("# 🔥 InfraPy Viewer")
ui.label("Select a file or folder to load infrared data")

ui.upload(on_upload=handle_file_selection, label="Select File", auto_upload=True)
file_label = ui.label("No file selected yet.")

ui.button("📂 Load File", on_click=load_file, color="primary")

# Run app
ui.run(title="InfraPy GUI", reload=False)
