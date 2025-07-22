import numpy as np

def synth_lockin_animation(magnitude, phase=None, f=10.0, fs=100, n_frames=100):
    """
    Create a synthetic animation of the lock-in magnitude modulated by a sine wave.

    Parameters:
    -----------
    magnitude : ndarray [H, W]
        Magnitude map (from lock-in).
    phase : ndarray [H, W] or None
        Phase map in degrees. If None, uses zero phase everywhere.
    f : float
        Frequency of oscillation [Hz].
    fs : float
        Sampling frequency of animation [Hz].
    n_frames : int
        Number of frames in the animation.

    Returns:
    --------
    animation : ndarray [n_frames, H, W]
        Stack of synthetic oscillating frames.
    """

    H, W = magnitude.shape
    t = np.arange(n_frames) / fs  # time vector

    # Convert phase to radians if given
    if phase is None:
        phi = 0
    else:
        phi = np.deg2rad(phase)

    # Broadcast time and spatial data
    time_sine = np.sin(2 * np.pi * f * t)[:, None, None]  # [n_frames, 1, 1]
    if phase is not None:
        animation = magnitude[None, :, :] * np.sin(2 * np.pi * f * t[:, None, None] + phi[None, :, :])
    else:
        animation = magnitude[None, :, :] * time_sine

    return animation
