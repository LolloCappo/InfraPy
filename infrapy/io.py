from pathlib import Path
import numpy as np
import tifffile
import sdypy.io.sfmov as sfmov

def load_ir_data(path):
    """
    Load infrared data from supported formats:
    - Single or multi-frame TIFF, CSV, SFMOV, NPY, NPZ
    - Folder of TIFF, CSV, or SFMOV files (each treated as 1 frame)

    Parameters
    ----------
    path : str or Path
        Path to the file or directory containing infrared data.

    Returns
    ----------
    ndarray: 3D array with shape (num_frames, height, width)
    """
    path = Path(path)

    def ensure_3d(arr):
        """Make sure output is always (frames, height, width) with single grayscale channel."""
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            # Single frame, 2D image
            return arr[np.newaxis, ...]
        elif arr.ndim == 3:
            # Could be (frames, height, width) or (height, width, channels)
            if arr.shape[-1] in [3, 4]:
                # (height, width, channels) -> convert to grayscale and add frame axis
                arr_gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                return arr_gray[np.newaxis, ...]
            else:
                # (frames, height, width)
                return arr
        elif arr.ndim == 4:
            # (frames, height, width, channels)
            if arr.shape[-1] in [3, 4]:
                # Convert each frame to grayscale using weighted sum over channels
                arr_gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                return arr_gray
            else:
                raise ValueError(f"Unsupported array shape: {arr.shape}")
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

def save_ir_data(array, filepath, key="data", overwrite=True):
    """
    Save a 2D or 3D NumPy array to disk in .npy or compressed .npz format.

    Parameters
    ----------
    array : np.ndarray
        Array to save. Expected shape is (height, width) or (frames, height, width).
    filepath : str or Path
        Destination file path. Must have extension .npy or .npz.
    key : str, optional
        Key name for the array if saving as .npz (default is "data").
    overwrite : bool, optional
        Whether to overwrite existing files (default True). If False and file exists, raises FileExistsError.

    Raises
    ------
    ValueError
        If file extension is not .npy or .npz.
    TypeError
        If input array is not a NumPy ndarray.
    FileExistsError
        If overwrite is False and file already exists.

    Returns
    -------
    Path
        The full path to the saved file as a Path object.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input 'array' must be a NumPy ndarray.")

    if array.ndim not in [2, 3]:
        raise ValueError(f"Input array must be 2D or 3D, got {array.ndim}D.")

    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File {filepath} already exists and overwrite=False.")

    if suffix == ".npy":
        np.save(filepath, array)
    elif suffix == ".npz":
        np.savez_compressed(filepath, **{key: array})
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .npy or .npz.")

    return filepath
