import numpy as np
from scipy.signal import resample

def celsius_to_kelvin(temp_c):
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

def kelvin_to_celsius(temp_k):
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

def compute_temporal_snr(data):
    """
    Compute temporal signal-to-noise ratio (SNR = mean / std) across frames.

    Parameters
    ----------
    data : ndarray
        Input data array of shape (frames, height, width).

    Returns
    -------
    ndarray
        SNR map of shape (height, width).
    """
    mean_signal = np.mean(data, axis=0)
    std_noise = np.std(data, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(std_noise != 0, mean_signal / std_noise, 0)
    return snr

def normalize_temporal_signal(data):
    """
    Normalize pixel intensity values over time to [0, 1] range per pixel.

    Parameters
    ----------
    data : ndarray
        Thermal data of shape (frames, height, width).

    Returns
    -------
    ndarray
        Normalized data of same shape.
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    range_val = max_val - min_val
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = (data - min_val) / np.where(range_val != 0, range_val, 1)
    return norm

def resample_frames(data, target_frames):
    """
    Resample a sequence of thermal frames to a new number of frames.

    Parameters
    ----------
    data : ndarray
        Input thermal data array with shape (frames, height, width).
    target_frames : int
        Desired number of output frames.

    Returns
    -------
    ndarray
        Resampled data with shape (target_frames, height, width).
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3D (frames, height, width).")
    n_frames, height, width = data.shape
    data_flat = data.reshape(n_frames, -1)
    resampled = resample(data_flat, target_frames, axis=0)
    return resampled.reshape(target_frames, height, width)

def split_sequence(data, window_size, step=1):
    """
    Split a frame sequence into overlapping windows.

    Parameters
    ----------
    data : ndarray
        Input thermal data array with shape (frames, height, width).
    window_size : int
        Number of frames per window.
    step : int, optional
        Step size between windows. Default is 1.

    Returns
    -------
    ndarray
        Array of shape (num_windows, window_size, height, width).
    """
    n_frames, height, width = data.shape
    num_windows = (n_frames - window_size) // step + 1
    windows = np.zeros((num_windows, window_size, height, width), dtype=data.dtype)
    for i in range(num_windows):
        start = i * step
        windows[i] = data[start:start + window_size]
    return windows

def build_time_vector(n_frames, fps):
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
    ndarray
        Time vector of shape (n_frames,), in seconds.
    """
    return np.arange(n_frames) / fps
