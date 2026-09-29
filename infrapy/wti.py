"""
Weighted Thermoelastic Identification (WTI).

Fuses C repeated full-field response maps of the same component (e.g. thermoelastic
amplitude maps acquired under different environmental conditions) into one map. Each
measurement is weighted by its Image Similarity Metric (ISM) to the current consensus,
and the consensus is the pixel-wise weighted median of the measurements.

Nothing here is specific to infrared data: any stack of maps shaped (C, H, W) on a
common grid can be fused (see :func:`stack_maps`).

Typical workflow::

    from infrapy import thermoelasticity, wti

    maps = []
    for data, fs, foi in acquisitions:              # (frames, H, W) sequences
        seg = int(data.shape[0] / 4)
        amp, freq = thermoelasticity.spectral(data, fs, method="fft",
                                              segment_length=seg, overlap=0.25)
        maps.append(amp[np.argmin(np.abs(freq - foi))])

    V = wti.stack_maps(maps)
    reference, weights, history = wti.process_wti(V, max_iter=10)
    variance_map, residual_map = wti.wti_diagnostics(V, reference)
"""

from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from infrapy.utils import interpolate_to_match


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


def weighted_median(ensemble: np.ndarray, weights: np.ndarray):
    """
    Weighted median along the first axis, ignoring NaNs.

    At each position, returns the member value ``v`` minimising
    ``sum_c w_c * |e_c - v|`` over the finite members, with the weights renormalised
    over those members. Ties go to the first member in ensemble order; positions with
    no finite member are NaN.

    Vectorised over positions, with the same per-position arithmetic (including
    summation order) as the original per-pixel implementation, so results are
    identical to it, ties included.

    Parameters
    ----------
    ensemble : ndarray, shape (C,) or (C, ...)
    weights : ndarray, shape (C,)

    Returns
    -------
    float for 1-D input, otherwise ndarray of shape ``ensemble.shape[1:]``.
    """
    e = np.asarray(ensemble)
    w_all = np.asarray(weights)
    C = e.shape[0]
    if w_all.shape != (C,):
        raise ValueError(f"weights must have shape ({C},), got {w_all.shape}")

    flat = e.reshape(C, -1)
    valid = np.isfinite(flat)
    n_valid = valid.sum(axis=0)
    out = np.full(flat.shape[1], np.nan)

    # Finite members first, keeping ensemble order, so every group of positions with
    # the same member count is one contiguous (positions, n) block.
    order = np.argsort(~valid, axis=0, kind="stable")

    for n in np.unique(n_valid):
        if n == 0:
            continue
        cols = np.flatnonzero(n_valid == n)
        idx = order[:n, cols].T
        vals = flat[idx, cols[:, None]]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = w_all[idx]
            w = w / np.sum(w, axis=1, keepdims=True)

            best = vals[:, 0].copy()
            min_cost = np.full(len(cols), np.inf)
            for k in range(n):
                cost = np.sum(w * np.abs(vals - vals[:, k:k + 1]), axis=1)
                better = cost < min_cost
                best[better] = vals[better, k]
                min_cost[better] = cost[better]
        out[cols] = best

    if e.ndim == 1:
        return float(out[0])
    return out.reshape(e.shape[1:])


def stack_maps(
    maps: Sequence[np.ndarray],
    like: Optional[np.ndarray] = None,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Bring maps of possibly different sizes onto one grid and stack them.

    Parameters
    ----------
    maps : sequence of ndarray (H_c, W_c)
        One response map per measurement.
    like : ndarray (H, W), optional
        Map whose grid is used (e.g. a laboratory reference). If omitted, the first
        map sets the grid and is kept unchanged; the others are interpolated to it.
    method : {'nearest', 'bilinear', 'bicubic'}
        Interpolation method (see :func:`infrapy.utils.interpolate_to_match`).

    Returns
    -------
    ndarray, shape (C, H, W)
    """
    if like is None:
        first, rest = maps[0], maps[1:]
        return np.stack([first] + [interpolate_to_match(first, m, method) for m in rest])
    return np.stack([interpolate_to_match(like, m, method) for m in maps])


def process_wti(
    measurements: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    max_iter: int = 5,
    convergence_threshold: float = 0.99,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Weighted Thermoelastic Identification with history tracking.

    Iteration 0 is the uniform weighted median. Each following iteration sets
    ``z_c = ISM(V_c, R) / sum_c ISM(V_c, R)`` and recomputes ``R`` as the weighted
    median. Stops when ``ISM(R_k, R_{k-1}) >= convergence_threshold`` or after
    ``max_iter`` weighted iterations.

    Parameters
    ----------
    measurements : ndarray, shape (C, H, W)
        Stack of C measurements on a common grid (see :func:`stack_maps`).
    ground_truth : ndarray, shape (H, W), optional
        Laboratory reference for validation. Only tracked in ``history``; it does not
        influence the result.
    max_iter : int
        Maximum number of weighted iterations.
    convergence_threshold : float
        ISM threshold for convergence (default 0.99). ``1.0`` iterates until the
        reference stops changing or ``max_iter`` is reached.
    verbose : bool
        Print a message on convergence.

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

    Notes
    -----
    If ISM is undefined for any measurement (fewer than 2 pixels overlapping the
    reference, or a constant map), the weight sum is NaN and all weights fall back to
    uniform for that iteration.
    """
    measurements = np.asarray(measurements)
    C, H, W = measurements.shape

    reference_history: list[np.ndarray] = []
    weight_history: list[np.ndarray] = []
    mac_sequential: list[float] = []
    mac_to_ground_truth: list[float] = []

    # Iteration 0: uniform weights
    weights = np.ones(C) / C
    reference = weighted_median(measurements, weights)

    reference_history.append(reference.copy())
    weight_history.append(weights.copy())

    if ground_truth is not None:
        mac_to_ground_truth.append(ism(reference, ground_truth))

    for iteration in range(max_iter):
        prev_reference = reference.copy()

        ism_values = np.array([ism(measurements[c], reference) for c in range(C)])
        total = ism_values.sum()
        weights = ism_values / total if total > 0 else np.ones(C) / C

        reference = weighted_median(measurements, weights)

        reference_history.append(reference.copy())
        weight_history.append(weights.copy())

        mac_seq = ism(reference, prev_reference)
        mac_sequential.append(mac_seq)

        if ground_truth is not None:
            mac_to_ground_truth.append(ism(reference, ground_truth))

        if mac_seq >= convergence_threshold:
            if verbose:
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
) -> plt.Figure:
    """
    Plot stacked bar chart of weight evolution across WTI iterations.

    Parameters
    ----------
    weight_history : list of ndarray, each shape (C,)
    condition_labels : list of str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    C = len(weight_history[0])
    iterations = np.arange(len(weight_history))

    if condition_labels is None:
        condition_labels = [f"Measurement {i + 1}" for i in range(C)]

    if C <= 6:
        colors = plt.cm.inferno_r([0.15, 0.30, 0.45, 0.60, 0.75, 0.90])[:C]
    else:
        colors = plt.cm.inferno_r(np.linspace(0.15, 0.90, C))

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
    return fig
