import numpy as np

def lock_in_analysis(data, fs, fl, method='correlation', band=0.5):
    """
    Lock-in analysis for thermoelastic signals demodulation.

    Parameters:
    -----------
    data : ndarray [frames, height, width]
        Sequence of thermal images (time series).

    fs : float
        Sampling frequency of thermal video [Hz].

    fl : float
        Frequency of excitation load [Hz].

    method : str, optional
        Method to use for lock-in:
        - 'correlation' : correlation method using sine and cosine demodulation (equivalent to lock-in amplifier).
        - 'fft'         : FFT-based method extracting the frequency component closest to fl.
        - 'lsqfit'      : Least squares fitting method fitting a sum of sinusoids at fl.

    band : float, optional
        Frequency bandwidth around fl to consider in 'fft' method [Hz]. Ignored for other methods.

    Returns:
    --------
    magnitude : ndarray [height, width]
        Magnitude of the locked-in signal (thermal units).

    phase : ndarray [height, width]
        Phase of the locked-in signal in degrees.

    Notes:
    ------
    The thermoelastic signal T(t) at a pixel can be modeled as:

        T(t) = T_DC + ΔT_e * sin(2π f_l t + φ_e) + 
               Σ_{n=2}^k ΔT_i * sin(2π n f_l t + φ_i) + noise(t)

    where:
    - T_DC is the mean (DC) temperature,
    - ΔT_e and φ_e are amplitude and phase of the excitation frequency component,
    - ΔT_i and φ_i are amplitudes and phases of harmonics,
    - noise(t) is measurement noise.

    The 'correlation' method extracts the in-phase (X) and quadrature (Y) components by correlating with sine and cosine at f_l.

    The 'fft' method uses frequency domain filtering, selecting the frequency bin closest to f_l.

    The 'lsqfit' method fits the model in the time domain using least squares.

    References:
    -----------
    - Pitarresi, G., Cappello, R., & Catalanotti, G. (2020). Quantitative thermoelastic stress analysis by means of low-cost setups.
    Optics and Lasers in Engineering, 134, 106158.

    """
    N, H, W = data.shape
    t = np.arange(N) / fs  # time vector

    if method == 'correlation':
        # Correlation (Lock-in Amplifier style)
        sine = np.sin(2 * np.pi * fl * t)
        cosine = np.cos(2 * np.pi * fl * t)

        # Reshape for broadcasting
        sine_3d = sine[:, None, None]
        cosine_3d = cosine[:, None, None]

        X = (2 / N) * np.sum(data * cosine_3d, axis=0)  # In-phase component
        Y = (2 / N) * np.sum(data * sine_3d, axis=0)    # Quadrature component

        magnitude = np.sqrt(X**2 + Y**2)
        phase = np.degrees(np.arctan2(Y, X))

    elif method == 'fft':
        # FFT method
        fft_data = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(N, d=1/fs)

        # Select positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_data = fft_data[pos_mask, :, :]

        # Frequency band selection
        band_mask = (freqs >= fl - band/2) & (freqs <= fl + band/2)
        selected_freqs = freqs[band_mask]
        selected_fft = fft_data[band_mask, :, :]

        # Find closest frequency to fl
        idx = np.argmin(np.abs(selected_freqs - fl))
        fft_val = selected_fft[idx]

        # Extract real and imaginary parts
        magnitude = 2 * np.abs(fft_val) / N
        phase = np.degrees(np.angle(fft_val))

    elif method == 'lsqfit':
        # Least squares fitting method
        # Model: A*cos(2π f_l t) + B*sin(2π f_l t) + C (DC term)
        cos_comp = np.cos(2 * np.pi * fl * t)[:, None, None]
        sin_comp = np.sin(2 * np.pi * fl * t)[:, None, None]
        ones_comp = np.ones_like(cos_comp)

        # Build design matrix [N x 3]
        # We'll solve separately for each pixel: y = Xb + noise, b = [A, B, C]
        X_mat = np.stack([cos_comp, sin_comp, ones_comp], axis=-1)  # shape (N,H,W,3)

        # Reshape for vectorized least squares
        X_reshaped = X_mat.reshape(N, -1, 3)  # (N, H*W, 3)
        y = data.reshape(N, -1)               # (N, H*W)

        # Solve for b using least squares for each pixel
        # b = (X.T X)^-1 X.T y for each pixel separately
        b = np.empty((y.shape[1], 3))
        for i in range(y.shape[1]):
            Xi = X_reshaped[:, i, :]  # (N,3)
            yi = y[:, i]              # (N,)
            coeffs, _, _, _ = np.linalg.lstsq(Xi, yi, rcond=None)
            b[i, :] = coeffs

        A = b[:, 0]
        B = b[:, 1]
        # C = b[:, 2]  # DC offset, not used here

        magnitude = 2 * np.sqrt(A**2 + B**2)
        phase = np.degrees(np.arctan2(B, A))

        magnitude = magnitude.reshape(H, W)
        phase = phase.reshape(H, W)

    else:
        raise ValueError("Method must be 'correlation', 'fft', or 'lsqfit'")

    return magnitude, phase
