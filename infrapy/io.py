from pathlib import Path
from typing import Union, Optional
import numpy as np
import tifffile
import pandas as pd
import pysfmov as sfmov
from fasthcc import read_hcc
from tqdm import tqdm


def read_ir(
    filenames: Union[str, list[str]],
    skip_frames: int = 1,
    return_timestamps: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Read infrared camera data from one or more .hcc or .sfmov files.

    Parameters
    ----------
    filenames : str or list of str
        Path(s) to .hcc or .sfmov file(s).
    skip_frames : int
        Frame decimation factor (1 = no skipping).
    return_timestamps : bool
        If True, include formatted timestamps in the metadata DataFrame.

    Returns
    -------
    data : ndarray, shape (frames, height, width)
    meta_data : DataFrame
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    all_data: list[np.ndarray] = []
    all_meta_data: list[pd.DataFrame] = []

    hcc_files = [f for f in filenames if f.endswith(".hcc")]
    sfmov_files = [f for f in filenames if f.endswith(".sfmov")]

    for filename in sfmov_files:
        try:
            data = sfmov.get_data(filename)
            meta_data = sfmov.get_meta_data(filename)
            return data, meta_data
        except Exception as e:
            raise RuntimeError(f"Error processing {filename}: {e}") from e

    for filename in hcc_files:
        try:
            frames_slice = slice(None, None, skip_frames) if skip_frames > 1 else None
            data, meta = read_hcc(filename, frames=frames_slice, metadata=True, calibrated=True)
            meta_df = pd.DataFrame(meta)
            if return_timestamps:
                meta_df["Formatted_Timestamp"] = pd.to_datetime(meta_df["POSIXTime"], unit="s")
            all_data.append(data)
            all_meta_data.append(meta_df)
        except Exception as e:
            raise RuntimeError(f"Error processing {filename}: {e}") from e

    if hcc_files:
        return np.concatenate(all_data, axis=0), pd.concat(all_meta_data, ignore_index=True)

    raise ValueError("No valid .hcc or .sfmov files were found.")


def save_ir_data(
    array: np.ndarray,
    filepath: Union[str, Path],
    key: str = "data",
    overwrite: bool = True,
) -> Path:
    """
    Save a 2D or 3D NumPy array to .npy or compressed .npz format.

    Parameters
    ----------
    array : ndarray
        Array to save. Shape (height, width) or (frames, height, width).
    filepath : str or Path
        Destination path. Extension must be .npy or .npz.
    key : str
        Key name when saving as .npz.
    overwrite : bool
        If False and the file exists, raises FileExistsError.

    Returns
    -------
    Path
        Full path to the saved file.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("'array' must be a NumPy ndarray.")
    if array.ndim not in (2, 3):
        raise ValueError(f"Array must be 2D or 3D, got {array.ndim}D.")

    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"{filepath} already exists and overwrite=False.")

    if suffix == ".npy":
        np.save(filepath, array)
    elif suffix == ".npz":
        np.savez_compressed(filepath, **{key: array})
    else:
        raise ValueError(f"Unsupported format: '{suffix}'. Use .npy or .npz.")

    return filepath


def load_ir_data(
    path: Union[str, Path],
    sort_by: str = "name",
    normalize: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Load infrared data from a file or directory of files.

    Supported formats: .tif/.tiff, .csv, .sfmov, .npy, .npz, .hcc.
    When given a directory, all supported files are loaded and stacked.

    Parameters
    ----------
    path : str or Path
        File or directory to load from.
    sort_by : str
        File ordering when loading a directory: "name" or "mtime".
    normalize : bool
        If True, normalize each frame to [0, 1].
    verbose : bool
        If True, print progress information.

    Returns
    -------
    ndarray, shape (frames, height, width)
    """
    path = Path(path)

    def ensure_3d(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                return gray[np.newaxis, ...]
            return arr
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    def sort_files(files: list[Path]) -> list[Path]:
        if sort_by == "mtime":
            return sorted(files, key=lambda f: f.stat().st_mtime)
        return sorted(files, key=lambda f: f.name)

    def normalize_frame(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else arr

    frames: list[np.ndarray] = []

    if path.is_dir():
        supported = {".tif", ".tiff", ".csv", ".sfmov", ".hcc"}
        files = sort_files([f for f in path.iterdir() if f.suffix.lower() in supported])
        if not files:
            raise FileNotFoundError(f"No supported files found in: {path}")
        if verbose:
            print(f"Found {len(files)} files in {path}")

        for f in tqdm(files, desc="Loading IR data"):
            suffix = f.suffix.lower()
            if verbose:
                print(f"  {f.name}")
            if suffix in (".tif", ".tiff"):
                frames.extend(ensure_3d(tifffile.imread(f)))
            elif suffix == ".csv":
                frames.extend(ensure_3d(np.loadtxt(f, delimiter=",")))
            elif suffix == ".sfmov":
                frames.extend(ensure_3d(sfmov.get_data(f)))

        hcc_files = [str(f) for f in files if f.suffix.lower() == ".hcc"]
        if hcc_files:
            if verbose:
                print(f"Processing {len(hcc_files)} HCC files…")
            hcc_data, _ = read_ir(hcc_files)
            frames.extend(ensure_3d(hcc_data))

    else:
        suffix = path.suffix.lower()

        if suffix in (".tif", ".tiff"):
            tiff_files = sort_files(
                list(path.parent.glob("*.tif")) + list(path.parent.glob("*.tiff"))
            )
            if verbose:
                print(f"Found {len(tiff_files)} TIFFs in {path.parent}")
            for f in tqdm(tiff_files, desc="Loading TIFFs"):
                frames.extend(ensure_3d(tifffile.imread(f)))

        elif suffix == ".csv":
            frames.extend(ensure_3d(np.loadtxt(path, delimiter=",")))

        elif suffix == ".sfmov":
            frames.extend(ensure_3d(sfmov.get_data(str(path))))

        elif suffix == ".hcc":
            hcc_files = sort_files(list(path.parent.glob("*.hcc")))
            if not hcc_files:
                raise FileNotFoundError(f"No .hcc files found in {path.parent}")
            if verbose:
                print(f"Found {len(hcc_files)} HCC files in {path.parent}")
            data, _ = read_ir([str(f) for f in hcc_files])
            frames.extend(ensure_3d(data))

        elif suffix in (".npy", ".npz"):
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                for key in data:
                    frames.extend(ensure_3d(data[key]))
                if not frames:
                    raise ValueError("No arrays found in .npz file.")
            else:
                frames.extend(ensure_3d(data))

        else:
            raise ValueError(f"Unsupported file type: '{suffix}'")

    if normalize:
        frames = [normalize_frame(f) for f in frames]

    return np.stack(frames, axis=0)
