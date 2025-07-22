import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def LIA_animation(magnitude, phase=None, f=10.0, fs=100, n_frames=100, cmap='inferno'):
    """
    Create and display a synthetic lock-in animation using magnitude and optional phase maps.

    Parameters:
    -----------
    magnitude : ndarray [H, W]
        Magnitude map from lock-in analysis.
    phase : ndarray [H, W] or None
        Phase map in degrees. If None, zero phase is used.
    f : float
        Frequency of oscillation [Hz].
    fs : float
        Sampling frequency of the animation [Hz].
    n_frames : int
        Number of frames in the animation.
    cmap : str
        Colormap for the animation.

    Returns:
    --------
    ani : matplotlib.animation.FuncAnimation
        The animation object, useful for saving as .mp4/.gif.
    frames : ndarray [n_frames, H, W]
        The synthetic frame stack (oscillating thermal signal).
    """

    H, W = magnitude.shape
    t = np.arange(n_frames) / fs  # time vector

    if phase is None:
        phi = 0
    else:
        phi = np.deg2rad(phase)

    # Create animated frame stack
    if phase is not None:
        frames = magnitude[None, :, :] * np.sin(2 * np.pi * f * t[:, None, None] + phi[None, :, :])
    else:
        time_sine = np.sin(2 * np.pi * f * t)[:, None, None]
        frames = magnitude[None, :, :] * time_sine

    # Display animation
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, animated=True)
    ax.set_title("Synthetic Lock-In Animation")
    plt.colorbar(im, ax=ax)

    def update(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fs, blit=True)
    plt.show()

    return ani, frames
