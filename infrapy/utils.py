import numpy as np
from scipy.signal import resample
from scipy.ndimage import zoom


def celsius_to_kelvin(temp_c: np.ndarray) -> np.ndarray:
    """
    Convert Celsius temperature to Kelvin.

    Parameters
    ----------
    temp_c : ndarray
        Temperature values in degrees Celsius.

    Returns
    -------
    ndarray
        Temperature values in Kelvin.
    """
    return temp_c + 273.15


def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
    """
    Convert Kelvin temperature to Celsius.

    Parameters
    ----------
    temp_k : ndarray
        Temperature values in Kelvin.

    Returns
    -------
    ndarray
        Temperature values in degrees Celsius.
    """
    return temp_k - 273.15


def compute_temporal_snr(data: np.ndarray) -> np.ndarray:
    """
    Compute temporal signal-to-noise ratio (SNR = mean / std) across frames.

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
        Input data array.

    Returns
    -------
    ndarray, shape (height, width)
        SNR map.
    """
    mean_signal = np.mean(data, axis=0)
    std_noise = np.std(data, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(std_noise != 0, mean_signal / std_noise, 0)
    return snr


def normalize_temporal_signal(data: np.ndarray) -> np.ndarray:
    """
    Normalize pixel intensity values over time to [0, 1] range per pixel.

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
        Thermal data.

    Returns
    -------
    ndarray, same shape
        Normalized data.
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    range_val = max_val - min_val
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (data - min_val) / np.where(range_val != 0, range_val, 1)
    return norm


def resample_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resample a sequence of thermal frames to a new number of frames.

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
        Input thermal data.
    target_frames : int
        Desired number of output frames.

    Returns
    -------
    ndarray, shape (target_frames, height, width)
        Resampled data.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3D (frames, height, width).")
    n_frames, height, width = data.shape
    data_flat = data.reshape(n_frames, -1)
    resampled = resample(data_flat, target_frames, axis=0)
    return resampled.reshape(target_frames, height, width)


def split_sequence(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """
    Split a frame sequence into overlapping windows.

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
        Input thermal data.
    window_size : int
        Number of frames per window.
    step : int, optional
        Step size between windows. Default is 1.

    Returns
    -------
    ndarray, shape (num_windows, window_size, height, width)
    """
    n_frames, height, width = data.shape
    num_windows = (n_frames - window_size) // step + 1
    windows = np.zeros((num_windows, window_size, height, width), dtype=data.dtype)
    for i in range(num_windows):
        start = i * step
        windows[i] = data[start : start + window_size]
    return windows


def build_time_vector(n_frames: int, fps: float) -> np.ndarray:
    """
    Generate a time vector corresponding to frame indices.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    fps : float
        Frame rate in Hz.

    Returns
    -------
    ndarray, shape (n_frames,)
        Time vector in seconds.
    """
    return np.arange(n_frames) / fps


def interpolate_to_match(
    big_matrix: np.ndarray,
    small_matrix: np.ndarray,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Interpolate small_matrix so its shape matches big_matrix.

    Works for 2D (H, W) or 3D arrays (N, H, W) or (H, W, C).

    Parameters
    ----------
    big_matrix : ndarray
        Reference array with the target shape.
    small_matrix : ndarray
        Array to resize.
    method : {'nearest', 'bilinear', 'bicubic'}
        Interpolation method.

    Returns
    -------
    ndarray
        Resized version of small_matrix with shape matching big_matrix.
    """
    method_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
    if method not in method_map:
        raise ValueError(f"Invalid method '{method}'. Choose from {list(method_map.keys())}.")

    zoom_factors = [
        big_matrix.shape[i] / small_matrix.shape[i] if i < len(big_matrix.shape) else 1.0
        for i in range(len(small_matrix.shape))
    ]
    return zoom(small_matrix, zoom_factors, order=method_map[method])
