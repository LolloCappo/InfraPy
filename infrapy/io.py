from pathlib import Path
import numpy as np
import tifffile
import imageio.v2 as imageio  # safe image reader

def load_snapshot(filepath):
    """
    Load a single infrared snapshot as a 2D NumPy array.

    Supports:
    - Single-frame TIFF
    - Single .npy file
    - Single .png/.jpg/.bmp image

    Returns:
        2D NumPy array
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix in [".npy"]:
        array = np.load(filepath)
        if array.ndim != 2:
            raise ValueError("Expected a 2D array for snapshot.")
        return array

    elif filepath.suffix in [".tif", ".tiff"]:
        img = tifffile.imread(filepath)
        if img.ndim != 2:
            raise ValueError("Expected a single-frame TIFF (2D array).")
        return img

    elif filepath.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
        img = imageio.imread(filepath)
        if img.ndim != 2:
            raise ValueError("Expected grayscale image for IR snapshot.")
        return img.astype(np.float32)

    else:
        raise ValueError(f"Unsupported snapshot format: {filepath.suffix}")


def load_sequence(path):
    """
    Load a sequence of infrared frames (3D array: frames, height, width).

    Supports:
    - Multi-frame TIFF
    - .npy or .npz files (with 3D arrays)
    - Folder of .png images (alphabetical order)

    Returns:
        3D NumPy array (frames, height, width)
    """
    path = Path(path)

    if path.is_dir():
        files = sorted(path.glob("*.png"))
        if not files:
            raise FileNotFoundError("No image files (.png) found in folder.")
        stack = np.stack([imageio.imread(f).astype(np.float32) for f in files], axis=0)
        return stack

    elif path.suffix in [".npy", ".npz"]:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            # Look for a key with 3D data
            for key in data:
                if data[key].ndim == 3:
                    return data[key]
            raise ValueError("No 3D array found in .npz file.")
        elif isinstance(data, np.ndarray) and data.ndim == 3:
            return data
        else:
            raise ValueError("Expected a 3D array in .npy/.npz file.")

    elif path.suffix in [".tif", ".tiff"]:
        stack = tifffile.imread(path)
        if stack.ndim != 3:
            raise ValueError("Expected multi-frame TIFF (3D array).")
        return stack.astype(np.float32)

    else:
        raise ValueError(f"Unsupported sequence format: {path.suffix}")


# Optional: Aliases for saving
def save_array(data, filepath):
    """
    Save a 2D or 3D NumPy array to .npy or .npz format.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".npy":
        np.save(filepath, data)
    elif filepath.suffix == ".npz":
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            raise ValueError("For .npz, data must be a dict of arrays.")
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")
