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

    References:
    -----------
    - Pitarresi, G. "Lock-in signal post-processing techniques in infra-red thermography for materials structural evaluation."
        Experimental Mechanics 55.4 (2015): 667-680.

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

        # Select frequency bin
        if band > 0:
            # Frequency band selection
            band_mask = (freqs >= fl - band/2) & (freqs <= fl + band/2)
            if not np.any(band_mask):
                raise ValueError(f"No frequency components found within ±{band/2} Hz of {fl} Hz")
            selected_freqs = freqs[band_mask]
            selected_fft = fft_data[band_mask, :, :]
            idx = np.argmin(np.abs(selected_freqs - fl))
            fft_val = selected_fft[idx]
        else:
            # Just find the closest bin
            idx = np.argmin(np.abs(freqs - fl))
            fft_val = fft_data[idx]

        # Extract magnitude and phase
        magnitude = 2 * np.abs(fft_val) / N
        phase = np.degrees(np.angle(fft_val))
        
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
        Method for thermal amplitude extraction, e.g. 'correlation', 'fft'
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
    else:
        raise ValueError("Invalid method. Choose from 'correlation', 'fft'.")
    
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

