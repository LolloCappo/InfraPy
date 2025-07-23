import numpy as np

def lock_in_analysis(data, fs, fl, method='fft', band=0.5):
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
        - 'fft'         : FFT-based method extracting the frequency component closest to fl. Default method
        - 'lsf'         : Least squares fitting method fitting a sum of sinusoids at fl.

    band : float, optional
        Frequency bandwidth around fl to consider in 'fft' method [Hz]. Ignored for other methods.

    Returns:
    --------
    magnitude : ndarray [height, width]
        Magnitude of the locked-in signal (input units).

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

    The 'lsf' method fits the model in the time domain using least squares.

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
        # FFT method using rfft for real input signals
        fft_data = np.fft.rfft(data, axis=0)  # shape: [N_freqs, H, W]
        freqs = np.fft.rfftfreq(N, d=1/fs)    # positive frequencies only

        # Frequency band selection
        band_mask = (freqs >= fl - band/2) & (freqs <= fl + band/2)
        selected_freqs = freqs[band_mask]
        selected_fft = fft_data[band_mask, :, :]

        # Find closest frequency to fl
        idx = np.argmin(np.abs(selected_freqs - fl))
        fft_val = selected_fft[idx]

        # Extract magnitude and phase
        magnitude = 2 * np.abs(fft_val) / N
        phase = np.degrees(np.angle(fft_val))

    elif method == 'lsf':
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
        raise ValueError("Method must be 'correlation', 'fft', or 'lsf'")

    return magnitude, phase

def get_strain(eps, configuration=None):
    """
    Calculate equivalent strain from strain-gauge data.
    See detailed docs above.
    """
    eps = np.asarray(eps)
    if eps.ndim == 1 or eps.shape[1] == 1:
        return np.max(eps)
    if eps.shape[0] < eps.shape[1]:
        raise ValueError("Input shape seems transposed; rows should be time, columns gauges.")
    if eps.shape[1] == 2:
        return (np.max(eps[:, 0]) + np.max(eps[:, 1])) / 2
    if eps.shape[1] == 3:
        if configuration not in {'90', '120'}:
            raise ValueError("configuration must be '90' or '120' degrees for 3-gauge rosette.")
        if configuration == '90':
            return abs(np.max(eps[:,1]) - (np.max(eps[:,0]) + np.max(eps[:,2])) / 2)
        if configuration == '120':
            return abs((np.max(eps[:,0]) - np.max(eps[:,2])) / np.sqrt(3))

def from_material(material):
    """
    Lookup thermoelastic coefficient for standard materials.
    """
    coefficients = {
        'steel': 3.5e-12,
        'aluminium': 8.8e-12,
        'titanium': 3.5e-12,
        'epoxy': 6.2e-11,
        'magnesium': 1.4e-11,
        'glass': 3.85e-12
    }
    material = material.lower()
    if material not in coefficients:
        raise ValueError(f"Material '{material}' not recognized. Choose from: {list(coefficients.keys())}")
    return coefficients[material]

def from_strain_gauge(data, fs, fl, E, nu, strain, location, method='correlation', **kwargs):
    """
    Obtain the thermoelastic coefficient through strain-gauge calibration procedure
    
    Parameters:
    -----------
    data : ndarray [frames, height, width]
        Sequence of thermal images [°C]
    fs : float
        Sampling frequency [Hz]
    fl : float
        Load frequency of harmonic excitation [Hz]
    E : float
        Young Modulus of the material [Pa]
    nu : float
        Poisson's ratio
    strain : float
        Strain measured using strain-gauges [e]
    location : tuple (x, y, w, h)
        ROI coordinates of the area where the strain-gauges are bonded
    method : str
        Method for thermal amplitude extraction, e.g. 'correlation', 'fft', 'lsf'
    **kwargs:
        Additional arguments passed to the lock_in_analysis function
    
    Returns:
    --------
    km : float
        Thermoelastic coefficient [°C / Pa]
    """

    x, y, w, h = location
    
    if method == 'correlation':
        magnitude, _ = lock_in_analysis(data, fs, fl, method='correlation', **kwargs)
    elif method == 'fft':
        magnitude, _ = lock_in_analysis(data, fs, fl, method='fft', **kwargs)
    elif method == 'lsf':
        magnitude, _ = lock_in_analysis(data, fs, fl, method='lsf', **kwargs)
    else:
        raise ValueError("Invalid method. Choose from 'correlation', 'fft', 'lsf'.")
    
    # Average magnitude in the ROI
    mag_avg = np.mean(magnitude[y:y+h, x:x+w])

    # Thermoelastic coefficient computation
    km = (mag_avg * (1 - nu)) / (E * strain)
    
    return km

def thermoelastic_calibration(
    method,
    *,
    material=None,
    strain_data=None,
    strain_config=None,
    thermal_data=None,
    fs=None,
    fl=None,
    E=None,
    nu=None,
    location=None,
):
    """
    Unified thermoelastic calibration function.
    
    Parameters
    ----------
    method : str
        Calibration method, one of:
        - 'material': Lookup thermoelastic coefficient from material name.
        - 'direct_strain': Compute equivalent strain from strain gauge data.
        - 'strain_gauge': Calibrate thermoelastic coefficient using thermal data and strain gauge data.
    
    material : str, optional
        Material name for 'material' method.
    
    strain_data : ndarray, optional
        Strain gauge data (time series x gauges).
    
    strain_config : {'90', '120'}, optional
        Strain gauge rosette configuration for 3-gauge setups.
    
    thermal_data : ndarray, optional
        Thermal images sequence [frames, height, width].
    
    fs : float, optional
        Sampling frequency (Hz).
    
    fl : float, optional
        Loading frequency (Hz).
    
    E : float, optional
        Young’s modulus (Pa).
    
    nu : float, optional
        Poisson’s ratio (dimensionless).
    
    location : tuple, optional
        ROI location (x, y, w, h) for strain gauge position in thermal data.
    
    Returns
    -------
    float
        Thermoelastic coefficient [°C / Pa] for 'material' and 'strain_gauge',
        or equivalent strain (unitless) for 'direct_strain'.
    
    Raises
    ------
    ValueError if required inputs for the selected method are missing.
    """
    method = method.lower()
    
    if method == 'material':
        if material is None:
            raise ValueError("Material name must be provided for 'material' method.")
        return from_material(material)
    
    elif method == 'direct_strain':
        if strain_data is None:
            raise ValueError("strain_data must be provided for 'direct_strain' method.")
        return get_strain(strain_data, configuration=strain_config)
    
    elif method == 'strain_gauge':
        if any(arg is None for arg in (thermal_data, fs, fl, E, nu, strain_data, location)):
            raise ValueError("thermal_data, fs, fl, E, nu, strain_data, and location must all be provided for 'strain_gauge' method.")
        strain = get_strain(strain_data, configuration=strain_config)
        return from_strain_gauge(thermal_data, fs, fl, E, nu, strain, location)
    
    else:
        raise ValueError("Method must be one of: 'material', 'direct_strain', 'strain_gauge'.")

