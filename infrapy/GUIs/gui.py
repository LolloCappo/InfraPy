from nicegui import ui
from pathlib import Path
import sys
import time
import tempfile

# Add your InfraPy source root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from infrapy.io import load_snapshot, load_sequence


progress = ui.linear_progress(show_value=True).props('instant-feedback')
file_label = ui.label("No file uploaded yet.")


def handle_file_upload(e):
    """Triggered when a file is uploaded."""
    try:
        content = e.content.read()

        # Save to a temporary file
        suffix = Path(e.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            uploaded_path = Path(tmp.name)

        file_label.set_text(f"Uploaded: {e.name}")
        load_and_report(uploaded_path)

    except Exception as ex:
        ui.notify(f"Upload failed: {str(ex)}", color="negative")


def load_and_report(path: Path):
    """Load the file and show progress."""
    try:
        progress.set_value(0.1)
        ui.notify("Loading started...", color="info")
        time.sleep(0.2)

        if path.suffix.lower() in [".tif", ".tiff", ".npy", ".npz"]:
            data = load_sequence(path)
        else:
            data = load_snapshot(path)

        dims = data.shape
        progress.set_value(1.0)
        ui.notify(f"Loaded: shape {dims}", color="positive")

    except Exception as ex:
        progress.set_value(0.0)
        ui.notify(f"Error loading file: {str(ex)}", color="negative")


# UI Layout
ui.markdown("## InfraPy GUI")
ui.label("Upload a file to automatically load it")
ui.upload(
    on_upload=handle_file_upload,
    label="Upload and Load",
    auto_upload=True,
    max_files=1,
)

ui.run(title="InfraPy GUI", reload=False)
