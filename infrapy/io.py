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

def save_array(data, filepath):
    """Save a NumPy array to .npy or .npz."""
    filepath = Path(filepath)
    if filepath.suffix == ".npy":
        np.save(filepath, data)
    elif filepath.suffix == ".npz":
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            raise ValueError("For .npz, input must be a dict of arrays.")
    else:
        raise ValueError(f"Unsupported NumPy file format: {filepath.suffix}")

def save_tiff_stack(data, filepath):
    """Save a 3D array (frames, H, W) as a TIFF stack."""
    tifffile.imwrite(filepath, data)

def save_image_sequence(data, folder_path, prefix="frame", extension=".png"):
    """Save 3D array (frames, H, W) to image sequence."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(data):
        filename = folder / f"{prefix}_{i:04d}{extension}"
        imageio.imwrite(filename, frame.astype(np.uint8))

def save_file(data, filepath):
    """General-purpose saver."""
    filepath = Path(filepath)
    if filepath.suffix in [".npy", ".npz"]:
        save_array(data, filepath)
    elif filepath.suffix in [".tif", ".tiff"]:
        save_tiff_stack(data, filepath)
    else:
        raise ValueError(f"Unsupported file format for saving: {filepath.suffix}")