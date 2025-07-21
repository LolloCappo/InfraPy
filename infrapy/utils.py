import numpy as np
from scipy.signal import resample
from typing import Tuple
import logging
from tqdm import tqdm

# ---------------------------
# LOGGING SETUP
# ---------------------------

def setup_logger(name: str = "infrapy", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger for InfraPy modules.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

# Example usage:
# logger = setup_logger()
# logger.info("Starting process...")

# ---------------------------
# PROGRESS WRAPPER
# ---------------------------

def progress_iterator(iterable, desc="Processing", total=None):
    """
    Wrap any iterable with a tqdm progress bar.
    """
    return tqdm(iterable, desc=desc, total=total)

# ---------------------------
# UNIT CONVERSIONS
# ---------------------------

def celsius_to_kelvin(temp_c: np.ndarray) -> np.ndarray:
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15

def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15

# ---------------------------
# SIGNAL ANALYSIS UTILITIES
# ---------------------------

def compute_snr(signal: np.ndarray, axis=-1) -> float:
    """
    Compute signal-to-noise ratio (SNR = mean / std) along given axis.
    """
    mean_signal = np.mean(signal, axis=axis)
    std_noise = np.std(signal, axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(std_noise != 0, mean_signal / std_noise, 0)
    return snr

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal between 0 and 1."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)

# ---------------------------
# FRAME RESAMPLING / ALIGNMENT
# ---------------------------

def resample_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resample a 3D array (frames, H, W) to a new number of frames.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3D (frames, height, width).")
    n_frames, height, width = data.shape
    data_reshaped = data.reshape(n_frames, -1)
    resampled = resample(data_reshaped, target_frames, axis=0)
    return resampled.reshape(target_frames, height, width)

# ---------------------------
# FRAME SEQUENCE HELPERS
# ---------------------------

def split_sequence(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """
    Split frame sequence into overlapping windows.
    Returns shape: (num_windows, window_size, H, W)
    """
    n_frames, height, width = data.shape
    num_windows = (n_frames - window_size) // step + 1
    windows = np.zeros((num_windows, window_size, height, width), dtype=data.dtype)
    for i in range(num_windows):
        start = i * step
        windows[i] = data[start:start + window_size]
    return windows

# ---------------------------
# TIME AXIS HANDLING
# ---------------------------

def build_time_vector(n_frames: int, fps: float) -> np.ndarray:
    """Generate a time vector for a given number of frames and frame rate."""
    return np.arange(n_frames) / fps
