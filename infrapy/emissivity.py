import numpy as np
from scipy.constants import h, c, k
from scipy.integrate import quad

# Physical constants (Planck's law)
C1 = 2 * h * c**2  # First radiation constant (W·µm^4/m^2/sr)
C2 = h * c / k     # Second radiation constant (µm·K)

# Default spectral band for LWIR cameras (typical: 7.5–13 μm)
DEFAULT_BAND = (7.5e-6, 13e-6)  # meters
DEFAULT_EFFECTIVE_WAVELENGTH = 10e-6  # meters (10 µm)

def planck_law(wavelength_m, temperature_K):
    """Spectral radiance at a given wavelength and temperature (W/sr/m²/m)."""
    exponent = C2 / (wavelength_m * temperature_K)
    return (C1 / wavelength_m**5) / (np.exp(exponent) - 1)

def band_integrated_radiance(temperature_K, band=DEFAULT_BAND):
    """Integrate Planck's law over a wavelength band (in meters)."""
    wl_min, wl_max = band
    return quad(lambda wl: planck_law(wl, temperature_K), wl_min, wl_max, limit=500)[0]

def radiance_from_temperature(temperature_K, method="band", **kwargs):
    """
    Compute total radiance from a blackbody at given temperature.

    Parameters:
    - temperature_K: scalar or array of temperatures in Kelvin
    - method: 'band' or 'monochromatic'
    - kwargs: optional arguments:
        - band (tuple): (min_wl, max_wl) in meters
        - wavelength (float): single wavelength in meters
    """
    T = np.asarray(temperature_K)
    if method == "monochromatic":
        wl = kwargs.get("wavelength", DEFAULT_EFFECTIVE_WAVELENGTH)
        return planck_law(wl, T)
    elif method == "band":
        band = kwargs.get("band", DEFAULT_BAND)
        return np.array([band_integrated_radiance(temp, band) for temp in T])
    else:
        raise ValueError("Invalid method. Choose 'band' or 'monochromatic'.")

def temperature_from_radiance(radiance, method="band", **kwargs):
    """
    Estimate temperature from measured radiance using numerical inversion.
    Only works for scalar radiance values.

    Parameters:
    - radiance: measured radiance (W/sr/m²)
    - method: 'band' or 'monochromatic'
    - kwargs: optional arguments for integration or wavelength
    """
    from scipy.optimize import root_scalar

    def residual(T):
        return radiance_from_temperature(T, method=method, **kwargs) - radiance

    sol = root_scalar(residual, bracket=[100, 1500], method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("Temperature inversion failed.")

def correct_emissivity(measured_radiance, emissivity=0.95, ambient_temp_K=300.0, method="band", **kwargs):
    """
    Apply emissivity correction to convert apparent radiance into true surface radiance.

    Parameters:
    - measured_radiance: apparent radiance from sensor
    - emissivity: object surface emissivity (0–1)
    - ambient_temp_K: background reflected temp (usually room temp)
    - method: 'band' or 'monochromatic'
    """
    reflected = radiance_from_temperature(ambient_temp_K, method=method, **kwargs)
    true_radiance = (measured_radiance - (1 - emissivity) * reflected) / emissivity
    return true_radiance
