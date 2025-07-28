import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from ipywidgets import interact, FloatSlider, fixed
import ipywidgets as widgets
from infrapy.thermoelasticity import lock_in_analysis

def animate_tsa(magnitude, phase=None, f=10.0, fs=100, n_frames=100, 
                cmap='viridis', save_path=None, dpi=150, speed_factor=2.0):
    """
    Generate and optionally save a synthetic TSA animation (e.g., thermoelastic response).

    Parameters
    ----------
    magnitude : ndarray [H, W]
        Magnitude map from lock-in analysis.
    phase : ndarray [H, W] or None
        Phase map in degrees. If None, zero phase is used.
    f : float
        Frequency of oscillation [Hz].
    fs : float
        Sampling frequency for frame generation [Hz].
    n_frames : int
        Total number of frames in the animation.
    cmap : str
        Colormap to use for visualization.
    save_path : str or Path or None
        If provided, save the animation to this path (.mp4 or .gif).
    dpi : int
        Resolution of the output animation.
    speed_factor : float
        Multiplier to slow down animation playback (higher = slower playback).

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    frames : ndarray [n_frames, H, W]
        Stack of generated frames.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    H, W = magnitude.shape
    t = np.arange(n_frames) / fs  # time vector

    if phase is None:
        phi = np.zeros_like(magnitude, dtype=np.float32)
    else:
        phi = np.deg2rad(phase).astype(np.float32)

    # Generate frame stack
    frames = np.empty((n_frames, H, W), dtype=np.float32)
    for i, time_point in enumerate(t):
        frames[i] = magnitude * np.sin(2 * np.pi * f * time_point + phi)

    # Setup figure but don't show
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, animated=True)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Signal amplitude')

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    # Compute adjusted interval for slower playback
    interval_ms = 1000 / fs * speed_factor

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=True)

    if save_path is not None:
        save_path = Path(save_path)
        ext = save_path.suffix.lower()
        if ext not in ['.mp4', '.gif']:
            raise ValueError("save_path extension must be .mp4 or .gif")
        if ext == '.mp4':
            writer = animation.writers['ffmpeg'](fps=fs / speed_factor, bitrate=1800)
            ani.save(save_path, writer=writer, dpi=dpi)
        elif ext == '.gif':
            ani.save(save_path, writer='imagemagick', fps=fs / speed_factor, dpi=dpi)

    plt.close(fig)  # prevent display

    return ani, frames


def plot_tsa_maps(magnitude, phase, 
                  cmap_mag='viridis', mag_lim=None, 
                  cmap_phase='bwr', ph_lim=None, 
                  title=None):
    """
    Plot TSA magnitude and phase maps side by side with colorbars.

    Parameters:
    -----------
    magnitude : ndarray [H, W]
        Lock-in magnitude map (e.g., amplitude of oscillation).
    phase : ndarray [H, W]
        Lock-in phase map in degrees.
    cmap_mag : str
        Colormap for the magnitude map (default: 'viridis').
    mag_lim : list or tuple [vmin, vmax] or None
        Color scale limits for the magnitude plot.
    cmap_phase : str
        Colormap for the phase map (default: 'bwr').
    ph_lim : list or tuple [vmin, vmax] or None
        Color scale limits for the phase plot.
    title : str or None
        Optional title for the full figure.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Set magnitude limits
    if mag_lim is None:
        mag_lim = [np.nanmin(magnitude), np.nanmax(magnitude)]
    im0 = axes[0].imshow(magnitude, cmap=cmap_mag, vmin=mag_lim[0], vmax=mag_lim[1])
    axes[0].set_title("Magnitude")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Set phase limits
    if ph_lim is None:
        ph_lim = [-180, 180]
    im1 = axes[1].imshow(phase, cmap=cmap_phase, vmin=ph_lim[0], vmax=ph_lim[1])
    axes[1].set_title("Phase (degrees)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Optional title
    if title:
        fig.suptitle(title, fontsize=14)

    # Hide axes
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
