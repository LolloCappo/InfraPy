import numpy as np
import imageio
import os
import tifffile
from pathlib import Path

def read_array(filepath):
    """Reads a .npy or .npz NumPy array."""
    filepath = Path(filepath)
    if filepath.suffix == ".npy":
        return np.load(filepath)
    elif filepath.suffix == ".npz":
        return dict(np.load(filepath))
    else:
        raise ValueError(f"Unsupported NumPy file format: {filepath.suffix}")

def read_tiff_stack(filepath):
    """Reads a multi-frame TIFF stack."""
    return tifffile.imread(filepath)

def read_image_sequence(folder_path, extension=".png"):
    """Reads a folder of images as a 3D array (frames, height, width)."""
    folder = Path(folder_path)
    files = sorted(folder.glob(f"*{extension}"))
    if not files:
        raise FileNotFoundError(f"No {extension} files found in {folder}")
    return np.stack([imageio.imread(file) for file in files], axis=0)

def read_file(filepath):
    """General-purpose reader that routes to appropriate format handler."""
    filepath = Path(filepath)
    if filepath.suffix in [".npy", ".npz"]:
        return read_array(filepath)
    elif filepath.suffix.lower() in [".tif", ".tiff"]:
        return read_tiff_stack(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
