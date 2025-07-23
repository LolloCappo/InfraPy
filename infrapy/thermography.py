import numpy as np

def thermal_delta_temperature(data, method='peak'):
    """
    Compute temperature variation over time using different statistical methods.

    Parameters
    ----------
    data : ndarray
        Thermal data [frames, height, width]
    method : {'peak', 'rms', 'std'}
        Method for computing ΔT:
        - 'peak': peak-to-peak variation
        - 'rms': root mean square
        - 'std': standard deviation

    Returns
    -------
    delta_T : ndarray
        [height, width] Temperature variation map
    """
    if method == 'peak':
        return np.max(data, axis=0) - np.min(data, axis=0)
    elif method == 'rms':
        return np.sqrt(np.mean(data**2, axis=0))
    elif method == 'std':
        return np.std(data, axis=0)
    else:
        raise ValueError("Method must be one of: 'peak', 'rms', 'std'")

def thermal_spatial_gradient(data):
    """
    Compute spatial temperature gradients (∇T) in x and y directions.

    Parameters
    ----------
    data : ndarray
        2D thermal image [height, width]

    Returns
    -------
    grad_x, grad_y : tuple of ndarray
        Spatial gradients in x and y directions
    """
    grad_y, grad_x = np.gradient(data)
    return grad_x, grad_y