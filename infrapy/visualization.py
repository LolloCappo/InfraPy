from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_tsa(
    magnitude: np.ndarray,
    phase: Optional[np.ndarray] = None,
    f: float = 10.0,
    fs: float = 100.0,
    n_frames: int = 100,
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    speed_factor: float = 2.0,
) -> tuple[animation.FuncAnimation, np.ndarray]:
    """
    Generate and optionally save a synthetic TSA animation.

    Parameters
    ----------
    magnitude : ndarray, shape (H, W)
        Magnitude map from lock-in analysis.
    phase : ndarray, shape (H, W), optional
        Phase map in degrees. Defaults to zero phase.
    f : float
        Oscillation frequency [Hz].
    fs : float
        Sampling frequency for frame generation [Hz].
    n_frames : int
        Number of frames in the animation.
    cmap : str
        Matplotlib colormap name.
    save_path : str or Path, optional
        If given, save the animation (.mp4 or .gif).
    dpi : int
        Output resolution.
    speed_factor : float
        Playback slow-down multiplier (higher = slower).

    Returns
    -------
    ani : FuncAnimation
    frames : ndarray, shape (n_frames, H, W)
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    H, W = magnitude.shape
    t = np.arange(n_frames) / fs
    phi = np.zeros_like(magnitude, dtype=np.float32) if phase is None else np.deg2rad(phase).astype(np.float32)

    frames = np.empty((n_frames, H, W), dtype=np.float32)
    for i, tp in enumerate(t):
        frames[i] = magnitude * np.sin(2 * np.pi * f * tp + phi)

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, animated=True)
    ax.axis("off")
    fig.colorbar(im, ax=ax).set_label("Signal amplitude")

    def update(idx: int) -> list:
        im.set_array(frames[idx])
        return [im]

    interval_ms = 1000 / fs * speed_factor
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=True)

    if save_path is not None:
        save_path = Path(save_path)
        ext = save_path.suffix.lower()
        if ext not in (".mp4", ".gif"):
            raise ValueError("save_path must end with .mp4 or .gif")
        if ext == ".mp4":
            writer = animation.writers["ffmpeg"](fps=fs / speed_factor, bitrate=1800)
            ani.save(save_path, writer=writer, dpi=dpi)
        else:
            ani.save(save_path, writer="imagemagick", fps=fs / speed_factor, dpi=dpi)

    plt.close(fig)
    return ani, frames


def plot_tsa_maps(
    magnitude: np.ndarray,
    phase: np.ndarray,
    cmap_mag: str = "viridis",
    mag_lim: Optional[tuple[float, float]] = None,
    cmap_phase: str = "bwr",
    ph_lim: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot TSA magnitude and phase maps side by side.

    Parameters
    ----------
    magnitude : ndarray, shape (H, W)
        Lock-in magnitude map.
    phase : ndarray, shape (H, W)
        Lock-in phase map in degrees.
    cmap_mag : str
        Colormap for magnitude (default: 'viridis').
    mag_lim : (vmin, vmax), optional
        Colour scale limits for magnitude.
    cmap_phase : str
        Colormap for phase (default: 'bwr').
    ph_lim : (vmin, vmax), optional
        Colour scale limits for phase. Defaults to (-180, 180).
    title : str, optional
        Figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmin_mag = np.nanmin(magnitude) if mag_lim is None else mag_lim[0]
    vmax_mag = np.nanmax(magnitude) if mag_lim is None else mag_lim[1]
    im0 = axes[0].imshow(magnitude, cmap=cmap_mag, vmin=vmin_mag, vmax=vmax_mag)
    axes[0].set_title("Magnitude")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmin_ph = -180.0 if ph_lim is None else ph_lim[0]
    vmax_ph = 180.0 if ph_lim is None else ph_lim[1]
    im1 = axes[1].imshow(phase, cmap=cmap_phase, vmin=vmin_ph, vmax=vmax_ph)
    axes[1].set_title("Phase (degrees)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
