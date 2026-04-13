from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def ism(
    map1: np.ndarray,
    map2: np.ndarray,
    *,
    demean: bool = True,
    on_fail: float = np.nan,
) -> float:
    """
    Compute an Image Similarity Metric equivalent to MAC:
        ISM = ( (v1·v2)² ) / ( (v1·v1) · (v2·v2) )

    Robust to NaNs (ignored pairwise) and optional demeaning.

    Parameters
    ----------
    map1, map2 : array-like, same shape
    demean : bool
        Subtract the mean of valid entries before computing ISM.
    on_fail : float
        Returned when insufficient valid data or zero norms.

    Returns
    -------
    float in [0, 1] or `on_fail`
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
    return float(np.clip((ab * ab) / (aa * bb), 0.0, 1.0))


def weighted_median(ensemble: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median that ignores NaNs in the ensemble.

    Parameters
    ----------
    ensemble : ndarray, shape (C,)
    weights : ndarray, shape (C,)

    Returns
    -------
    float
    """
    mask = np.isfinite(ensemble)
    if mask.sum() == 0:
        return np.nan

    e = ensemble[mask]
    w = weights[mask]
    w = w / w.sum()

    costs = np.array([np.sum(w * np.abs(e - v)) for v in e])
    return float(e[np.argmin(costs)])


def _weighted_median_map(measurements: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Vectorized weighted median over all pixels simultaneously.

    Parameters
    ----------
    measurements : ndarray, shape (C, H, W)
    weights : ndarray, shape (C,)

    Returns
    -------
    ndarray, shape (H, W)
    """
    C, H, W = measurements.shape
    finite_mask = np.isfinite(measurements)  # (C, H, W)
    valid_count = finite_mask.sum(axis=0)    # (H, W)

    # diff[k, c, h, w] = |meas[k, h, w] - meas[c, h, w]|
    diff = np.abs(measurements[:, np.newaxis] - measurements[np.newaxis, :])  # (C, C, H, W)

    # Zero out pairs where either entry is NaN
    finite_pair = finite_mask[:, np.newaxis] & finite_mask[np.newaxis, :]  # (C, C, H, W)
    diff = np.where(finite_pair, diff, 0.0)

    # cost[c, h, w] = sum_k(w[k] * diff[k, c, h, w])
    cost = np.einsum("k,kchw->chw", weights, diff)  # (C, H, W)

    # Candidates that are NaN get infinite cost
    cost = np.where(finite_mask, cost, np.inf)

    best_c = np.argmin(cost, axis=0)  # (H, W)
    h_idx = np.arange(H)[:, np.newaxis]
    w_idx = np.arange(W)[np.newaxis, :]
    result = measurements[best_c, h_idx, w_idx]

    return np.where(valid_count > 0, result, np.nan)


def process_wti(
    measurements: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    max_iter: int = 5,
    convergence_threshold: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Multi-Measurement Thermoelastic Response Identification with history tracking.

    Parameters
    ----------
    measurements : ndarray, shape (C, H, W)
        Stack of C measurements.
    ground_truth : ndarray, shape (H, W), optional
        Laboratory reference for validation.
    max_iter : int
        Maximum number of iterations.
    convergence_threshold : float
        ISM threshold for convergence (default 0.99).

    Returns
    -------
    reference : ndarray, shape (H, W)
        Final filtered thermoelastic response.
    weights : ndarray, shape (C,)
        Final weights for each measurement.
    history : dict
        - 'references': list of reference maps at each iteration
        - 'weights': list of weight arrays at each iteration
        - 'mac_sequential': ISM between consecutive references
        - 'mac_to_ground_truth': ISM to ground truth (if provided)
    """
    C, H, W = measurements.shape

    reference_history: list[np.ndarray] = []
    weight_history: list[np.ndarray] = []
    mac_sequential: list[float] = []
    mac_to_ground_truth: list[float] = []

    # Iteration 0: uniform weights
    weights = np.ones(C) / C
    reference = _weighted_median_map(measurements, weights)

    reference_history.append(reference.copy())
    weight_history.append(weights.copy())

    if ground_truth is not None:
        mac_to_ground_truth.append(ism(reference, ground_truth))

    for iteration in range(max_iter):
        prev_reference = reference.copy()

        ism_values = np.array([ism(measurements[c], reference) for c in range(C)])
        total = ism_values.sum()
        weights = ism_values / total if total > 0 else np.ones(C) / C

        reference = _weighted_median_map(measurements, weights)

        reference_history.append(reference.copy())
        weight_history.append(weights.copy())

        mac_seq = ism(reference, prev_reference)
        mac_sequential.append(mac_seq)

        if ground_truth is not None:
            mac_to_ground_truth.append(ism(reference, ground_truth))

        if mac_seq >= convergence_threshold:
            print(f"Converged at iteration {iteration + 1} (ISM = {mac_seq:.4f})")
            break

    history: dict = {
        "references": reference_history,
        "weights": weight_history,
        "mac_sequential": np.array(mac_sequential),
        "mac_to_ground_truth": np.array(mac_to_ground_truth) if ground_truth is not None else None,
    }
    return reference, weights, history


def wti_diagnostics(
    measurements: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pixel-wise variance and residual maps.

    Parameters
    ----------
    measurements : ndarray, shape (C, H, W)
    reference : ndarray, shape (H, W)

    Returns
    -------
    variance_map : ndarray, shape (H, W)
    residual_map : ndarray, shape (H, W)
        Sum of absolute residuals between each measurement and the reference.
    """
    variance_map = np.nanvar(measurements, axis=0)
    diffs = np.abs(measurements - reference[np.newaxis])  # (C, H, W)
    residual_map = np.nansum(diffs, axis=0)
    # Where all measurements were NaN, restore NaN
    all_nan = ~np.any(np.isfinite(measurements), axis=0)
    residual_map = np.where(all_nan, np.nan, residual_map)
    return variance_map, residual_map


def plot_weight_evolution(
    weight_history: list[np.ndarray],
    condition_labels: Optional[list[str]] = None,
) -> None:
    """
    Plot stacked bar chart of weight evolution across WTI iterations.

    Parameters
    ----------
    weight_history : list of ndarray, each shape (C,)
    condition_labels : list of str, optional
    """
    C = len(weight_history[0])
    iterations = np.arange(len(weight_history))

    if condition_labels is None:
        condition_labels = [f"Measurement {i + 1}" for i in range(C)]

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
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += weights_c

    ax.set_xlabel("WTI iteration $k$", fontsize=14)
    ax.set_ylabel(r"Weight $z_c^{(k)}$", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(iterations)
    ax.tick_params(labelsize=12)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=11,
        frameon=True,
    )
    plt.tight_layout()
    plt.show()
