import numpy as np


def thermal_delta_temperature(data: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Compute temperature variation over time using different statistical methods.

    Parameters
    ----------
    data : ndarray, shape (frames, height, width)
        Thermal data.
    method : {'peak', 'rms', 'std'}
        Method for computing ΔT:
        - 'peak': peak-to-peak variation
        - 'rms': root mean square
        - 'std': standard deviation

    Returns
    -------
    delta_T : ndarray, shape (height, width)
        Temperature variation map.
    """
    if method == "peak":
        return np.max(data, axis=0) - np.min(data, axis=0)
    elif method == "rms":
        return np.sqrt(np.mean(data ** 2, axis=0))
    elif method == "std":
        return np.std(data, axis=0)
    else:
        raise ValueError("method must be one of: 'peak', 'rms', 'std'")


def thermal_spatial_gradient(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial temperature gradients (∇T) in x and y directions.

    Parameters
    ----------
    data : ndarray, shape (height, width)
        2D thermal image.

    Returns
    -------
    grad_x, grad_y : tuple of ndarray
        Spatial gradients in x and y directions.
    """
    grad_y, grad_x = np.gradient(data)
    return grad_x, grad_y
