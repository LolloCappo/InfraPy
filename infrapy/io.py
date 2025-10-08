from pathlib import Path
import numpy as np
import tifffile
import sdypy.io.sfmov as sfmov


import pandas as pd
import numpy as np
import pysfmov as sfmov
import TelopsToolbox.utils.image_processing as ip
from tqdm import tqdm
from TelopsToolbox.hcc.readIRCam import read_ircam

def read_ir(filenames, skip_frames=1, return_timestamps=False):
    """
    Reads infrared camera data from one or more .hcc or .sfmov files and converts it to a stack of images.
    This function supports infrared images acquired using Telops (.hcc) and Flir (.sfmov) IR cameras.

    Args:
        filenames (str or list): Path(s) to the infrared camera data file(s) (.hcc or .sfmov).
        skip_frames (int): Number of frames to skip when processing. Defaults to 1 (no skipping).
        return_timestamps (bool): If True, include timestamps from metadata. Defaults to False.

    Returns:
        If `.hcc` files: 
            - combined_data (np.ndarray): A 3D array representing the stack of 2D infrared images.
                                          Shape: (num_frames, height, width).
            - combined_meta_data (pd.DataFrame): Metadata for each frame.
        If `.sfmov` file:
            - data (np.ndarray): A 3D array representing the infrared images from that file.
                                Shape: (num_frames, height, width)
            - meta_data (pd.DataFrame): Metadata for that file.
    """
    # Ensure filenames is a list
    if isinstance(filenames, str):
        filenames = [filenames]

    # Initialize storage for `.hcc` files only
    all_data = []
    all_meta_data = []
    
    hcc_files = [f for f in filenames if f.endswith('.hcc')]
    sfmov_files = [f for f in filenames if f.endswith('.sfmov')]

    # Process `.sfmov` files individually
    for filename in sfmov_files:
        try:
            data = sfmov.get_data(filename)
            meta_data = sfmov.get_meta_data(filename)
            return data, meta_data  # Direct return for `.sfmov`
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the file {filename}: {e}")

    # Process `.hcc` files (if any)
    for filename in hcc_files:
        try:
            frame_data, frame_headers, special_pixels, non_special_pixels = read_ircam(filename)
            image_stack = []
            meta_data = pd.DataFrame(frame_headers)

            for i in tqdm(range(0, len(frame_data), skip_frames), desc=f"Loading frames from {filename}", unit="frame"):
                this_header = meta_data.iloc[i]
                processed_image = ip.form_image(this_header, frame_data[i]).squeeze()
                image_stack.append(processed_image)

            data = np.stack(image_stack)

            if return_timestamps:
                meta_data['Formatted_Timestamp'] = pd.to_datetime(meta_data['POSIXTime'], unit='s')

            # Append results to the cumulative lists
            all_data.append(data)
            all_meta_data.append(meta_data)

        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the file {filename}: {e}")

    # If `.hcc` files were processed, combine their data
    if hcc_files:
        combined_data = np.concatenate(all_data, axis=0)
        combined_meta_data = pd.concat(all_meta_data, ignore_index=True)
        return combined_data, combined_meta_data

    # If no valid files were processed, raise an error
    raise ValueError("No valid `.hcc` or `.sfmov` files were found.")


def load_ir_data(path):
    """
    Load infrared data from supported formats:
    - Single or multi-frame TIFF, CSV, SFMOV, NPY, NPZ, HCC
    - Folder of TIFF, CSV, SFMOV, or HCC files (each treated as 1 frame or multi-frame)

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
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        elif arr.ndim == 3:
            if arr.shape[-1] in [3, 4]:
                arr_gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                return arr_gray[np.newaxis, ...]
            else:
                return arr
        elif arr.ndim == 4:
            if arr.shape[-1] in [3, 4]:
                arr_gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                return arr_gray
            else:
                raise ValueError(f"Unsupported array shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    if path.is_dir():
        files = sorted(
            f for f in path.iterdir()
            if f.suffix.lower() in [".tif", ".tiff", ".csv", ".sfmov", ".hcc"]
        )
        if not files:
            raise FileNotFoundError(f"No supported files found in folder: {path}")

        frames = []
        hcc_files = [str(f) for f in files if f.suffix.lower() == ".hcc"]
        other_files = [f for f in files if f.suffix.lower() != ".hcc"]

        # Process non-HCC files
        for f in other_files:
            suffix = f.suffix.lower()
            if suffix in [".tif", ".tiff"]:
                img = tifffile.imread(f)
                img = ensure_3d(img)
                frames.extend(img)
            elif suffix == ".csv":
                arr = np.loadtxt(f, delimiter=',')
                arr = ensure_3d(arr)
                frames.append(arr[0])
            elif suffix == ".sfmov":
                arr = sfmov.get_data(f)
                arr = ensure_3d(arr)
                frames.extend(arr)

        # Process HCC files
        if hcc_files:
            hcc_data, _ = read_ir(hcc_files)
            hcc_data = ensure_3d(hcc_data)
            frames.extend(hcc_data)

        return np.stack(frames, axis=0)

    # Single file
    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        arr = tifffile.imread(path)
        return ensure_3d(arr)

    elif suffix == ".csv":
        arr = np.loadtxt(path, delimiter=',')
        return ensure_3d(arr)

    elif suffix == ".hcc":
        # Load all .hcc files in the same folder
        hcc_files = sorted(str(f) for f in path.parent.glob("*.hcc"))
        if not hcc_files:
            raise FileNotFoundError(f"No other .hcc files found in folder: {path.parent}")
        data, _ = read_ir(hcc_files)
        return ensure_3d(data)

    elif suffix == ".hcc":
        data, _ = read_ir(str(path))
        return ensure_3d(data)

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
