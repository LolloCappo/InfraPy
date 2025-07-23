import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def animate_tsa(magnitude, phase=None, f=10.0, fs=100, n_frames=100, 
                cmap='bwr', save_path=None, dpi=150):
    """
    Create and display a synthetic lock-in animation using magnitude and optional phase maps.
    Optionally save the animation as an mp4 or gif file.

    Parameters
    ----------
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
        Colormap for the animation. Default is 'bwr', centered at zero.
    save_path : str or Path or None
        If specified, save the animation to this path (.mp4 or .gif).
        If None, only display the animation.
    dpi : int
        Resolution of saved animation.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object, useful for further saving or manipulation.
    frames : ndarray [n_frames, H, W]
        The synthetic frame stack (oscillating thermal signal).
    """

    H, W = magnitude.shape
    t = np.arange(n_frames) / fs  # time vector

    if phase is None:
        phi = np.zeros((H, W), dtype=np.float32)
    else:
        phi = np.deg2rad(phase).astype(np.float32)

    # Create frame stack [n_frames, H, W]
    frames = np.empty((n_frames, H, W), dtype=np.float32)
    for i, time_point in enumerate(t):
        frames[i, :, :] = magnitude * np.sin(2 * np.pi * f * time_point + phi)

    # Plot setup
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, vmin=-np.max(magnitude), vmax=np.max(magnitude), animated=True)
    ax.set_title("Thermoelastic Signal Animation")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Signal amplitude')

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fs, blit=True)

    if save_path is not None:
        save_path = Path(save_path)
        ext = save_path.suffix.lower()
        if ext not in ['.mp4', '.gif']:
            raise ValueError("save_path extension must be .mp4 or .gif")
        if ext == '.mp4':
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fs, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(save_path, writer=writer, dpi=dpi)
        elif ext == '.gif':
            ani.save(save_path, writer='imagemagick', dpi=dpi)

    else:
        plt.show()

    return ani, frames

def plot_tsa_maps(magnitude, phase, cmap_mag='viridis', cmap_phase='bwr', title=None):
    """
    Plot TSA magnitude and phase maps side by side with colorbars.

    Parameters:
    -----------
    magnitude : ndarray [H, W]
        Lock-in magnitude map (e.g., amplitude of oscillation).
    phase : ndarray [H, W]
        Lock-in phase map in degrees.
    cmap_mag : str
        Colormap for the magnitude map (default: 'inferno').
    cmap_phase : str
        Colormap for the phase map (default: 'twilight' is good for cyclic data).
    title : str or None
        Optional title for the full figure.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(magnitude, cmap=cmap_mag)
    axes[0].set_title("Magnitude")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phase, cmap=cmap_phase, vmin=-180, vmax=180)
    axes[1].set_title("Phase (degrees)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()