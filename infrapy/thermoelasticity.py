from typing import Optional, Union
import numpy as np


def lock_in_analysis(
    data: np.ndarray,
    fs: float,
    fl: float,
    method: str = "fft",
    band: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lock-in analysis for thermoelastic signal demodulation.

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
        Sequence of thermal images.
    fs : float
        Sampling frequency [Hz].
    fl : float
        Excitation load frequency [Hz].
    method : {'fft', 'correlation'}
        Demodulation method.
        - 'correlation': lock-in amplifier style (sine/cosine demodulation).
        - 'fft': FFT-based extraction of the bin closest to fl.
    band : float
        Frequency bandwidth [Hz] around fl for FFT method. Ignored for correlation.

    Returns
    -------
    magnitude : ndarray, shape (height, width)
    phase : ndarray, shape (height, width), in degrees

    Notes
    -----
    The thermoelastic signal T(t) at a pixel is modelled as:

        T(t) = T_DC + ΔT_e · sin(2π f_l t + φ_e) + harmonics + noise(t)

    References
    ----------
    Pitarresi, G. (2015). Lock-in signal post-processing techniques in
    infra-red thermography for materials structural evaluation.
    Experimental Mechanics, 55(4), 667–680.
    """
    N, H, W = data.shape
    t = np.arange(N) / fs

    if method == "correlation":
        sine_3d = np.sin(2 * np.pi * fl * t)[:, None, None]
        cosine_3d = np.cos(2 * np.pi * fl * t)[:, None, None]
        X = (2 / N) * np.sum(data * cosine_3d, axis=0)
        Y = (2 / N) * np.sum(data * sine_3d, axis=0)
        magnitude = np.sqrt(X ** 2 + Y ** 2)
        phase = np.degrees(np.arctan2(Y, X))

    elif method == "fft":
        fft_data = np.fft.rfft(data, axis=0)
        freqs = np.fft.rfftfreq(N, d=1 / fs)

        if band > 0:
            band_mask = (freqs >= fl - band / 2) & (freqs <= fl + band / 2)
            if not np.any(band_mask):
                raise ValueError(f"No frequency components found within ±{band/2} Hz of {fl} Hz")
            selected_freqs = freqs[band_mask]
            selected_fft = fft_data[band_mask]
            idx = np.argmin(np.abs(selected_freqs - fl))
            fft_val = selected_fft[idx]
        else:
            idx = np.argmin(np.abs(freqs - fl))
            fft_val = fft_data[idx]

        magnitude = 2 * np.abs(fft_val) / N
        phase = np.degrees(np.angle(fft_val))

    else:
        raise ValueError("method must be 'correlation' or 'fft'.")

    return magnitude, phase


def spectral(
    data: np.ndarray,
    fs: float,
    method: str = "fft",
    foi: Optional[float] = None,
    segment_length: Optional[int] = None,
    overlap: float = 0.25,
    zero_pad: bool = True,
    apply_window: bool = True,
    segment: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spectral thermoelasticity analysis (Welch-style FFT or Digital Lock-In Correlation).

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
    fs : float
        Sampling frequency [Hz].
    method : {'fft', 'lockin'}
    foi : float, optional
        Frequency of interest [Hz]. Required for 'lockin' method.
    segment_length : int, optional
        Frames per segment. Required when segment=True.
    overlap : float
        Fractional overlap between segments [0, 1).
    zero_pad : bool
        Zero-pad data to the next full second boundary.
    apply_window : bool
        Apply a Hann window to each segment.
    segment : bool
        Use Welch-style segmentation.

    Returns
    -------
    amplitude_map : ndarray
        FFT: shape (freq_bins, H, W). Lock-in: shape (H, W).
    freq : ndarray
        Frequency axis (Hz). Lock-in returns a single-element array [foi].
    """
    frames = data.shape[0]

    if zero_pad:
        target_length = int(np.ceil(frames / fs) * fs)
        if frames < target_length:
            pad_width = ((0, target_length - frames), (0, 0), (0, 0))
            data = np.pad(data, pad_width, mode="constant", constant_values=np.nan)
            frames = target_length

    seg_len = int(segment_length) if segment and segment_length is not None else frames
    window = np.hanning(seg_len) if apply_window else np.ones(seg_len)

    if segment:
        if segment_length is None:
            raise ValueError("segment_length must be provided when segment=True.")
        step = int(seg_len * (1 - overlap))
        if step <= 0:
            raise ValueError("Overlap too high — step becomes zero or negative.")
        segments = [(s, s + seg_len) for s in range(0, frames - seg_len + 1, step)]
    else:
        segments = [(0, frames)]

    if method == "fft":
        amplitudes = []
        for start, end in segments:
            seg = data[start:end] - np.nanmean(data[start:end], axis=0)
            seg = seg * window[:, None, None]
            amp = np.abs(np.fft.rfft(seg, axis=0)) * 2 / (end - start)
            amplitudes.append(amp)
        freq = np.fft.rfftfreq(seg_len, 1 / fs)
        return np.nanmean(amplitudes, axis=0), freq

    elif method == "lockin":
        if foi is None:
            raise ValueError("foi must be provided for the 'lockin' method.")
        t = np.arange(seg_len) / fs
        ref_cos = np.cos(2 * np.pi * foi * t)
        ref_sin = np.sin(2 * np.pi * foi * t)
        amplitudes = []
        for start, end in segments:
            seg = data[start:end] - np.nanmean(data[start:end], axis=0)
            seg = seg * window[:, None, None]
            X = np.nansum(seg * ref_cos[:, None, None], axis=0)
            Y = np.nansum(seg * ref_sin[:, None, None], axis=0)
            amplitudes.append(np.sqrt(X ** 2 + Y ** 2) * 2 / (end - start))
        return np.nanmean(amplitudes, axis=0), np.array([foi])

    else:
        raise ValueError("method must be 'fft' or 'lockin'.")


def get_strain(
    eps: np.ndarray,
    configuration: Optional[str] = None,
) -> float:
    """
    Compute equivalent strain from strain-gauge data.

    Parameters
    ----------
    eps : ndarray, shape (time,) or (time, gauges)
        Strain gauge time series.
    configuration : {'90', '120'}, optional
        Rosette configuration for 3-gauge setups.

    Returns
    -------
    float
        Equivalent strain value.
    """
    eps = np.asarray(eps)
    if eps.ndim == 1 or eps.shape[1] == 1:
        return float(np.max(eps))
    if eps.shape[0] < eps.shape[1]:
        raise ValueError("Input seems transposed; rows should be time, columns gauges.")
    if eps.shape[1] == 2:
        return float((np.max(eps[:, 0]) + np.max(eps[:, 1])) / 2)
    if eps.shape[1] == 3:
        if configuration not in {"90", "120"}:
            raise ValueError("configuration must be '90' or '120' for a 3-gauge rosette.")
        if configuration == "90":
            return float(abs(np.max(eps[:, 1]) - (np.max(eps[:, 0]) + np.max(eps[:, 2])) / 2))
        return float(abs((np.max(eps[:, 0]) - np.max(eps[:, 2])) / np.sqrt(3)))
    raise ValueError(f"Unsupported number of gauges: {eps.shape[1]}")


def from_material(material: str) -> float:
    """
    Return the thermoelastic coefficient for a standard material.

    Parameters
    ----------
    material : str
        One of: 'steel', 'aluminium', 'titanium', 'epoxy', 'magnesium', 'glass'.

    Returns
    -------
    float
        Thermoelastic coefficient [°C / Pa].
    """
    coefficients: dict[str, float] = {
        "steel": 3.5e-12,
        "aluminium": 8.8e-12,
        "titanium": 3.5e-12,
        "epoxy": 6.2e-11,
        "magnesium": 1.4e-11,
        "glass": 3.85e-12,
    }
    key = material.lower()
    if key not in coefficients:
        raise ValueError(f"Material '{material}' not recognised. Choose from: {list(coefficients)}")
    return coefficients[key]


def from_strain_gauge(
    data: np.ndarray,
    fs: float,
    fl: float,
    E: float,
    nu: float,
    strain: float,
    location: tuple[int, int, int, int],
    method: str = "correlation",
    **kwargs,
) -> float:
    """
    Calibrate the thermoelastic coefficient from strain-gauge measurements.

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
        Thermal image sequence [°C].
    fs : float
        Sampling frequency [Hz].
    fl : float
        Load frequency [Hz].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    strain : float
        Measured equivalent strain.
    location : (x, y, w, h)
        ROI over the strain-gauge bonding area.
    method : {'correlation', 'fft'}

    Returns
    -------
    float
        Thermoelastic coefficient km [°C / Pa].
    """
    if method not in ("correlation", "fft"):
        raise ValueError("method must be 'correlation' or 'fft'.")

    magnitude, _ = lock_in_analysis(data, fs, fl, method=method, **kwargs)
    x, y, w, h = location
    mag_avg = float(np.mean(magnitude[y : y + h, x : x + w]))
    return mag_avg * (1 - nu) / (E * strain)


def thermoelastic_calibration(
    method: str,
    *,
    material: Optional[str] = None,
    strain_data: Optional[np.ndarray] = None,
    strain_config: Optional[str] = None,
    thermal_data: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    fl: Optional[float] = None,
    E: Optional[float] = None,
    nu: Optional[float] = None,
    location: Optional[tuple[int, int, int, int]] = None,
) -> float:
    """
    Unified thermoelastic calibration dispatcher.

    Parameters
    ----------
    method : {'material', 'direct_strain', 'strain_gauge'}
    material : str
        Material name (for 'material' method).
    strain_data : ndarray
        Strain-gauge time series (for 'direct_strain' and 'strain_gauge').
    strain_config : {'90', '120'}, optional
        Rosette configuration for 3-gauge setups.
    thermal_data : ndarray
        Thermal image stack (for 'strain_gauge').
    fs, fl : float
        Sampling and load frequencies [Hz] (for 'strain_gauge').
    E, nu : float
        Young's modulus [Pa] and Poisson's ratio (for 'strain_gauge').
    location : (x, y, w, h)
        Strain-gauge ROI (for 'strain_gauge').

    Returns
    -------
    float
        Thermoelastic coefficient [°C / Pa] or equivalent strain (unitless).
    """
    method = method.lower()

    if method == "material":
        if material is None:
            raise ValueError("'material' must be provided.")
        return from_material(material)

    if method == "direct_strain":
        if strain_data is None:
            raise ValueError("'strain_data' must be provided.")
        return get_strain(strain_data, configuration=strain_config)

    if method == "strain_gauge":
        missing = [
            name for name, val in [
                ("thermal_data", thermal_data), ("fs", fs), ("fl", fl),
                ("E", E), ("nu", nu), ("strain_data", strain_data), ("location", location),
            ] if val is None
        ]
        if missing:
            raise ValueError(f"Missing required arguments for 'strain_gauge': {missing}")
        strain = get_strain(strain_data, configuration=strain_config)  # type: ignore[arg-type]
        return from_strain_gauge(thermal_data, fs, fl, E, nu, strain, location)  # type: ignore[arg-type]

    raise ValueError("method must be 'material', 'direct_strain', or 'strain_gauge'.")
