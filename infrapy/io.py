from pathlib import Path
import numpy as np
import tifffile
import sdypy.io.sfmov as sfmov

def load_ir_data(path):
    """
    Load infrared data from supported formats:
    - Single or multi-frame TIFF, CSV, SFMOV, NPY, NPZ
    - Folder of TIFF, CSV, or SFMOV files (each treated as 1 frame)

    Always returns:
        ndarray: 3D array with shape (num_frames, height, width)
    """
    path = Path(path)

    def ensure_3d(arr):
        """Make sure output is always (frames, height, width)"""
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        elif arr.ndim == 3:
            return arr
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    if path.is_dir():
        # Folder: look for tiff, csv, sfmov files
        files = sorted(
            f for f in path.iterdir()
            if f.suffix.lower() in [".tif", ".tiff", ".csv", ".sfmov"]
        )
        if not files:
            raise FileNotFoundError(f"No supported files found in folder: {path}")

        frames = []
        for f in files:
            suffix = f.suffix.lower()
            if suffix in [".tif", ".tiff"]:
                img = tifffile.imread(f)
                img = ensure_3d(img)
                frames.extend(img)
            elif suffix == ".csv":
                arr = np.loadtxt(f, delimiter=',')
                arr = ensure_3d(arr)
                frames.append(arr[0])  # single CSV is one frame
            elif suffix == ".sfmov":
                arr = sfmov.get_data(f)
                arr = ensure_3d(arr)
                frames.extend(arr)
            else:
                raise ValueError(f"Unsupported file in folder: {f}")
        return np.stack(frames, axis=0)

    # Single file
    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        arr = tifffile.imread(path)
        return ensure_3d(arr)

    elif suffix == ".csv":
        arr = np.loadtxt(path, delimiter=',')
        return ensure_3d(arr)

    elif suffix == ".sfmov":
        arr = sfmov.get_data(path)
        return ensure_3d(arr)

    elif suffix in [".npy", ".npz"]:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            for key in data:
                arr = data[key]
                return ensure_3d(arr)
            raise ValueError("No arrays found in .npz file.")
        else:
            return ensure_3d(data)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def save_ir_data(array, filepath, key="data"):
    """
    Save a NumPy array to .npy or .npz format.

    Args:
        array (np.ndarray): The array to save (2D or 3D).
        filepath (str or Path): Destination file path (.npy or .npz).
        key (str): Key name to use if saving to .npz (default: "data").

    Raises:
        ValueError: If file extension is not .npy or .npz.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".npy":
        np.save(filepath, array)
    elif suffix == ".npz":
        np.savez_compressed(filepath, **{key: array})
    else:
        raise ValueError(f"Unsupported file format for saving: {suffix}. Use .npy or .npz.")
