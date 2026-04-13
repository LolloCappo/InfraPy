from typing import Union
import numpy as np
from scipy.constants import h, c, k
from scipy.integrate import quad

# Physical constants (Planck's law)
C1 = 2 * h * c**2  # First radiation constant (W·m^4/m^2/sr)
C2 = h * c / k     # Second radiation constant (m·K)

# Default spectral band for LWIR cameras (typical: 7.5–13 μm)
DEFAULT_BAND = (7.5e-6, 13e-6)          # meters
DEFAULT_EFFECTIVE_WAVELENGTH = 10e-6    # meters (10 µm)


def planck_law(wavelength_m: float, temperature_K: Union[float, np.ndarray]) -> np.ndarray:
    """
    Spectral radiance at a given wavelength and temperature.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in metres.
    temperature_K : float or ndarray
        Temperature in Kelvin.

    Returns
    -------
    ndarray
        Spectral radiance in W/sr/m²/m.
    """
    exponent = C2 / (wavelength_m * np.asarray(temperature_K))
    return (C1 / wavelength_m**5) / (np.exp(exponent) - 1)


def band_integrated_radiance(
    temperature_K: float,
    band: tuple[float, float] = DEFAULT_BAND,
) -> float:
    """
    Integrate Planck's law over a wavelength band.

    Parameters
    ----------
    temperature_K : float
        Temperature in Kelvin (scalar).
    band : (float, float)
        (min_wavelength, max_wavelength) in metres.

    Returns
    -------
    float
        Integrated radiance in W/sr/m².
    """
    wl_min, wl_max = band
    return quad(lambda wl: planck_law(wl, temperature_K), wl_min, wl_max, limit=500)[0]


def radiance_from_temperature(
    temperature_K: Union[float, np.ndarray],
    method: str = "band",
    **kwargs,
) -> np.ndarray:
    """
    Compute total radiance from a blackbody at the given temperature.

    Parameters
    ----------
    temperature_K : float or ndarray
        Temperature(s) in Kelvin.
    method : {'band', 'monochromatic'}
        Computation method.
    **kwargs
        band : (float, float) — wavelength band in metres (for 'band').
        wavelength : float   — single wavelength in metres (for 'monochromatic').

    Returns
    -------
    ndarray
        Radiance in W/sr/m².
    """
    T = np.asarray(temperature_K)
    if method == "monochromatic":
        wl = kwargs.get("wavelength", DEFAULT_EFFECTIVE_WAVELENGTH)
        return planck_law(wl, T)
    elif method == "band":
        band = kwargs.get("band", DEFAULT_BAND)
        scalar_input = T.ndim == 0
        T_flat = np.atleast_1d(T).ravel()
        result = np.array([band_integrated_radiance(float(t), band) for t in T_flat])
        return float(result[0]) if scalar_input else result.reshape(T.shape)
    else:
        raise ValueError("method must be 'band' or 'monochromatic'.")


def temperature_from_radiance(
    radiance: Union[float, np.ndarray],
    method: str = "band",
    t_min: float = 100.0,
    t_max: float = 1500.0,
    **kwargs,
) -> Union[float, np.ndarray]:
    """
    Estimate temperature from measured radiance using numerical inversion.

    Works for both scalar and array inputs.

    Parameters
    ----------
    radiance : float or ndarray
        Measured radiance in W/sr/m².
    method : {'band', 'monochromatic'}
        Computation method passed to radiance_from_temperature.
    t_min, t_max : float
        Search bracket for the Brent root-finding (Kelvin).
    **kwargs
        Forwarded to radiance_from_temperature.

    Returns
    -------
    float or ndarray
        Estimated temperature(s) in Kelvin.
    """
    from scipy.optimize import brentq

    def _invert_scalar(rad: float) -> float:
        def residual(T: float) -> float:
            return float(radiance_from_temperature(T, method=method, **kwargs)) - rad
        try:
            return brentq(residual, t_min, t_max)
        except ValueError as exc:
            raise RuntimeError(
                f"Temperature inversion failed for radiance={rad:.3e}. "
                "Check that the value lies within the expected range."
            ) from exc

    rad = np.asarray(radiance)
    scalar_input = rad.ndim == 0
    rad_flat = np.atleast_1d(rad).ravel()
    result = np.array([_invert_scalar(float(r)) for r in rad_flat])
    return float(result[0]) if scalar_input else result.reshape(rad.shape)


def correct_emissivity(
    measured_radiance: Union[float, np.ndarray],
    emissivity: float = 0.95,
    ambient_temp_K: float = 300.0,
    method: str = "band",
    **kwargs,
) -> Union[float, np.ndarray]:
    """
    Apply emissivity correction to convert apparent radiance to true surface radiance.

    Parameters
    ----------
    measured_radiance : float or ndarray
        Apparent radiance from sensor.
    emissivity : float
        Object surface emissivity in (0, 1].
    ambient_temp_K : float
        Background reflected temperature in Kelvin.
    method : {'band', 'monochromatic'}
        Computation method.
    **kwargs
        Forwarded to radiance_from_temperature.

    Returns
    -------
    float or ndarray
        True surface radiance.
    """
    reflected = radiance_from_temperature(ambient_temp_K, method=method, **kwargs)
    return (measured_radiance - (1 - emissivity) * reflected) / emissivity
