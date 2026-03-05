import pandas as pd
import numpy as np
import pysfmov as sfmov
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tqdm import tqdm

def ism(map1, map2, *, demean=True, on_fail=np.nan):
    """
    Compute an Image Similarity Metric equivalent to MAC:
        ISM = ( (v1•v2)^2 ) / ( (v1•v1)*(v2•v2) )
    Robust to NaNs (ignored pairwise) and optional demeaning.

    Parameters
    ----------
    map1, map2 : array-like, same shape
    demean : bool, default True
        Subtract the mean of valid entries before computing ISM (helps with DC offsets).
    on_fail : float, default np.nan
        Returned when insufficient valid data or zero norms.

    Returns
    -------
    ism : float in [0, 1] or `on_fail`
    """
    a = np.asarray(map1, dtype=float).ravel()
    b = np.asarray(map2, dtype=float).ravel()

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return on_fail

    a = a[mask]
    b = b[mask]

    if demean:
        a = a - a.mean()
        b = b - b.mean()

    aa = np.dot(a, a)
    bb = np.dot(b, b)
    if aa <= 0 or bb <= 0:
        return on_fail

    ab = np.dot(a, b)
    ism = (ab * ab) / (aa * bb)
    return float(np.clip(ism, 0.0, 1.0))

def weighted_median(ensemble, weights):
    """Weighted median that ignores NaNs in the ensemble."""
    mask = np.isfinite(ensemble)
    if mask.sum() == 0:
        return np.nan  # no valid data

    e = ensemble[mask]
    w = weights[mask]

    # Normalize weights to avoid bias
    w = w / w.sum()

    best_val = e[0]
    min_cost = np.inf

    for v in e:
        cost = np.sum(w * np.abs(e - v))
        if cost < min_cost:
            min_cost = cost
            best_val = v

    return best_val

def process_wti(measurements, ground_truth=None, max_iter=5, convergence_threshold=0.99):
    """
    Multi-Measurement Thermoelastic Response Identification with history tracking.
    
    Parameters:
    -----------
    measurements : np.ndarray, shape (C, H, W)
        Stack of C measurements
    ground_truth : np.ndarray, shape (H, W), optional
        Laboratory reference for validation
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        MAC threshold for convergence (default 0.99)
        
    Returns:
    --------
    reference : np.ndarray, shape (H, W)
        Final filtered thermoelastic response
    weights : np.ndarray, shape (C,)
        Final weights for each measurement
    history : dict
        Dictionary containing:
        - 'references': list of reference maps at each iteration
        - 'weights': list of weight arrays at each iteration
        - 'mac_sequential': MAC between consecutive references
        - 'mac_to_ground_truth': MAC to ground truth (if provided)
    """
    C, H, W = measurements.shape

    # Initialize history tracking
    reference_history = []
    weight_history = []
    mac_sequential = []
    mac_to_ground_truth = []

    # Iteration 0: unweighted median
    weights = np.ones(C) / C
    reference = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            reference[i, j] = weighted_median(measurements[:, i, j], weights)

    reference_history.append(reference.copy())
    weight_history.append(weights.copy())

    if ground_truth is not None:
        mac_to_ground_truth.append(ism(reference, ground_truth))

    # Iterations 1 to max_iter: weighted median
    for iteration in range(max_iter):
        prev_reference = reference.copy()

        # Compute ISM weights
        ism_values = np.array([ism(measurements[c], reference) for c in range(C)])
        weights = (ism_values / ism_values.sum()
                   if ism_values.sum() > 0 else np.ones(C) / C)

        # Update reference
        for i in range(H):
            for j in range(W):
                reference[i, j] = weighted_median(measurements[:, i, j], weights)

        # Track history
        reference_history.append(reference.copy())
        weight_history.append(weights.copy())
        mac_seq = ism(reference, prev_reference)
        mac_sequential.append(mac_seq)

        if ground_truth is not None:
            mac_to_ground_truth.append(ism(reference, ground_truth))

        # Check convergence
        if mac_seq >= convergence_threshold:
            print(f"Converged at iteration {iteration + 1} (MAC = {mac_seq:.4f})")
            break

    history = {
        'references': reference_history,
        'weights': weight_history,
        'mac_sequential': np.array(mac_sequential),
        'mac_to_ground_truth': np.array(mac_to_ground_truth)
                               if ground_truth is not None else None
    }

    return reference, weights, history

def wti_diagnostics(measurements, reference):
    """
    Compute pixel-wise variance and residual maps given measurements and a reference.

    Parameters:
    -----------
    measurements : np.ndarray, shape (C, H, W)
        Stack of C measurements
    reference : np.ndarray, shape (H, W)
        Reference map (e.g. output of wti)

    Returns:
    --------
    variance_map : np.ndarray, shape (H, W)
        Pixel-wise variance across measurements
    residual_map : np.ndarray, shape (H, W)
        Sum of absolute residuals between each measurement and the reference
    """
    C, H, W = measurements.shape

    variance_map = np.nanvar(measurements, axis=0)

    residual_map = np.full((H, W), np.nan)
    for c in range(C):
        diff = np.abs(measurements[c] - reference)

        mask_init = np.isnan(residual_map) & np.isfinite(diff)
        residual_map[mask_init] = diff[mask_init]

        mask_add = np.isfinite(residual_map) & np.isfinite(diff)
        residual_map[mask_add] += diff[mask_add]

    return variance_map, residual_map

def plot_weight_evolution(weight_history, condition_labels=None):
    C = len(weight_history[0])
    iterations = np.arange(len(weight_history))
    
    if condition_labels is None:
        condition_labels = [f'Measurement {i+1}' for i in range(C)]
    
    full_palette = plt.cm.inferno_r([0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    colors = full_palette[:C]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    bottoms = np.zeros(len(iterations))
    for c in range(C):
        weights_c = np.array([weight_history[k][c] for k in range(len(weight_history))])
        ax.bar(
            iterations, weights_c,
            bottom=bottoms,
            width=0.3,
            color=colors[c],
            label=condition_labels[c],
            edgecolor='white',
            linewidth=0.5
        )
        bottoms += weights_c

    ax.set_xlabel('WTI iteration $k$', fontsize=14)
    ax.set_ylabel(r'Weight $z_c^{(k)}$', fontsize=14)
    ax.set_ylim(0, 1.05)
    #ax.grid(True, which='major', alpha=0.2, axis='y')
    ax.legend(loc='upper left', fontsize=12)
    ax.set_xticks(iterations)
    ax.tick_params(labelsize=12)
    ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    fontsize=11,
    frameon=True
)

    plt.tight_layout()
    plt.show()