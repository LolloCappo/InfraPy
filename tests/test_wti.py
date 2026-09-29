"""
WTI tests.

The equivalence tests check that infrapy reproduces the original implementation used
for the paper (tests/reference_impl.py) exactly, not just approximately.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from infrapy import thermoelasticity, utils, wti
from . import reference_impl as ref

# Fully masked pixels (outside the ROI) trigger "Mean of empty slice" etc., as in the original.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def make_stack(C, H=14, W=11, nan_frac=0.2, dtype=np.float32, seed=0, integer=False):
    rng = np.random.default_rng(seed)
    if integer:
        # few distinct values -> many exact ties in the weighted median
        X = rng.integers(0, 4, size=(C, H, W)).astype(float)
    else:
        shape = np.outer(np.sin(np.linspace(0, np.pi, H)), np.sin(np.linspace(0, 2 * np.pi, W)))
        X = (shape[None] * rng.uniform(0.5, 1.5, (C, 1, 1))
             + rng.normal(0, 1, (C, H, W)) * rng.uniform(0.05, 1.0, (C, 1, 1)))
    X[rng.random(X.shape) < nan_frac] = np.nan
    X[:, 0, 0] = np.nan  # a pixel with no finite member
    return X.astype(dtype)


def make_sequence(frames=2350, H=6, W=5, fs=400.0, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / fs
    shape = rng.random((H, W))
    data = (20 + shape[None] * np.sin(2 * np.pi * 115.0 * t)[:, None, None]
            + rng.normal(0, 0.3, (frames, H, W)))
    data[:, 0, :2] = np.nan  # masked pixels, as after a circular crop
    return data.astype(dtype)


WEIGHTS = {
    "uniform": lambda C, rng: np.ones(C) / C,
    "random": lambda C, rng: (w := rng.random(C)) / w.sum(),
    "some_zero": lambda C, rng: np.where(np.arange(C) % 2 == 0, 0.0, 1.0 / C),
    "all_zero": lambda C, rng: np.zeros(C),
}


# --------------------------------------------------------------------------------------
# Equivalence with the original implementation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("C", [1, 2, 3, 4, 5, 6, 7, 8, 9, 12])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("wkind", list(WEIGHTS))
@pytest.mark.parametrize("integer", [False, True])
def test_weighted_median_matches_original(C, dtype, wkind, integer):
    stack = make_stack(C, dtype=dtype, seed=C, integer=integer)
    weights = WEIGHTS[wkind](C, np.random.default_rng(C))
    expected = np.array([[ref.weighted_median(stack[:, i, j], weights)
                          for j in range(stack.shape[2])] for i in range(stack.shape[1])], dtype=float)
    assert np.array_equal(wti.weighted_median(stack, weights), expected, equal_nan=True)


def test_weighted_median_1d():
    e = np.array([3.0, np.nan, 1.0, 2.0, 10.0])
    w = np.array([0.1, 0.5, 0.2, 0.3, 0.4])
    assert wti.weighted_median(e, w) == ref.weighted_median(e, w)
    assert np.isnan(wti.weighted_median(np.full(3, np.nan), np.ones(3)))


@pytest.mark.parametrize("C", [2, 3, 5, 6, 8])
@pytest.mark.parametrize("with_gt", [False, True])
@pytest.mark.parametrize("threshold", [0.99, 1.0])
def test_process_wti_matches_original(C, with_gt, threshold):
    stack = make_stack(C, seed=100 + C)
    gt = make_stack(1, nan_frac=0.0, seed=7)[0] if with_gt else None

    r_ref, w_ref, var_ref, _, h_ref = ref.mmtri_with_history(
        stack, ground_truth=gt, max_iter=10, convergence_threshold=threshold)
    r, w, h = wti.process_wti(stack, ground_truth=gt, max_iter=10,
                              convergence_threshold=threshold, verbose=False)

    assert np.array_equal(r, r_ref, equal_nan=True)
    assert np.array_equal(w, w_ref)
    assert len(h["references"]) == len(h_ref["references"])
    for a, b in zip(h["references"], h_ref["references"]):
        assert np.array_equal(a, b, equal_nan=True)
    for a, b in zip(h["weights"], h_ref["weights"]):
        assert np.array_equal(a, b)
    assert np.array_equal(h["mac_sequential"], h_ref["mac_sequential"], equal_nan=True)
    if with_gt:
        assert np.array_equal(h["mac_to_ground_truth"], h_ref["mac_to_ground_truth"], equal_nan=True)
    else:
        assert h["mac_to_ground_truth"] is None

    variance_map, _ = wti.wti_diagnostics(stack, r)
    assert np.array_equal(variance_map, var_ref, equal_nan=True)


def test_residual_map_vs_original():
    """
    wti_diagnostics returns sum_c |V_c - R|. The original loop counted the first
    finite measurement at each pixel twice; this pins down that difference.
    """
    stack = make_stack(5, seed=11)
    r, *_, res_orig, _ = ref.mmtri_with_history(stack, max_iter=10)
    _, residual_map = wti.wti_diagnostics(stack, r)

    diffs = np.abs(stack - r[None])
    no_data = np.isnan(r)
    assert np.allclose(residual_map, np.where(no_data, np.nan, np.nansum(diffs, axis=0)), equal_nan=True)

    first = np.argmax(np.isfinite(diffs), axis=0)
    first_diff = np.take_along_axis(diffs, first[None], axis=0)[0]
    assert np.allclose(res_orig, residual_map + first_diff, equal_nan=True)


def test_process_wti_nan_ism_falls_back_to_uniform():
    stack = make_stack(4, seed=3)
    stack[2] = np.nan  # ISM undefined for this measurement
    r_ref, *_ = ref.mmtri_with_history(stack, max_iter=3)
    r, w, _ = wti.process_wti(stack, max_iter=3, verbose=False)
    assert np.array_equal(r, r_ref, equal_nan=True)
    assert np.array_equal(w, np.ones(4) / 4)


@pytest.mark.parametrize("method", ["fft", "lockin"])
@pytest.mark.parametrize("apply_window", [True, False])
@pytest.mark.parametrize("zero_pad", [True, False])
@pytest.mark.parametrize("segment", [True, False])
@pytest.mark.parametrize("frames", [2350, 2400])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectral_matches_original(method, apply_window, zero_pad, segment, frames, dtype):
    data = make_sequence(frames=frames, dtype=dtype)
    kw = dict(method=method, foi=115.0, segment_length=int(frames / 4), overlap=0.25,
              zero_pad=zero_pad, apply_window=apply_window, segment=segment)
    a_ref, f_ref = ref.thermoelasticity(data, 400.0, **kw)
    a, f = thermoelasticity.spectral(data, 400.0, **kw)
    assert np.array_equal(a, a_ref, equal_nan=True)
    assert np.array_equal(f, f_ref)


def test_ism_matches_original():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.normal(size=(2, 30, 20))
        a[rng.random(a.shape) < 0.1] = np.nan
        for demean in (True, False):
            assert wti.ism(a, b, demean=demean) == ref.compute_ism(a, b, demean=demean)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
def test_stack_maps_matches_paper_pattern(method):
    rng = np.random.default_rng(2)
    maps = [rng.random((24, 24)), rng.random((24, 24)), rng.random((16, 20)), rng.random((30, 12))]
    # Paper_figures_v2.ipynb: keep the first map, interpolate the others to it
    expected = np.stack([maps[0]] + [ref.interpolate_to_match(maps[0], m, method) for m in maps[1:]])
    assert np.array_equal(wti.stack_maps(maps, method=method), expected)

    like = rng.random((20, 18))
    V = wti.stack_maps(maps, like=like, method=method)
    assert V.shape == (4, 20, 18)
    assert np.array_equal(V[2], ref.interpolate_to_match(like, maps[2], method))


# --------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------

def test_spectral_window_scaling():
    """Hann-windowed amplitudes are not gain-corrected: a sine of amplitude A reads ~A/2."""
    fs, f0 = 1000, 125.0
    t = np.arange(4000) / fs
    data = np.sin(2 * np.pi * f0 * t)[:, None, None] * np.ones((1, 2, 2))
    for method in ("fft", "lockin"):
        for window, expected in ((False, 1.0), (True, 0.5)):
            amp, freq = thermoelasticity.spectral(data, fs, method=method, foi=f0,
                                                  segment_length=1000, apply_window=window)
            value = amp[np.argmin(np.abs(freq - f0))] if method == "fft" else amp
            assert value[0, 0] == pytest.approx(expected, abs=1e-3)


def test_process_wti_downweights_corrupted_measurements():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[-1:1:48j, -1:1:48j]
    truth = np.abs(np.sin(2 * np.pi * x) * np.cos(np.pi * y))
    noise = [0.02, 0.05, 0.1, 0.4, 0.8]
    V = np.stack([truth + rng.normal(0, s, truth.shape) for s in noise])
    V[4] += 3 * np.exp(-((x - 0.4) ** 2 + (y - 0.3) ** 2) / 0.02)  # reflection-like hot spot

    reference, weights, history = wti.process_wti(V, ground_truth=truth, max_iter=10, verbose=False)

    assert np.all(np.diff(weights) < 0)  # cleaner measurement -> larger weight
    assert weights.sum() == pytest.approx(1.0)
    assert wti.ism(reference, truth) > wti.ism(np.median(V, axis=0), truth)
    assert history["mac_to_ground_truth"][-1] >= history["mac_to_ground_truth"][0]


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_weight_evolution_many_measurements():
    history = [np.ones(8) / 8, np.linspace(1, 2, 8) / np.linspace(1, 2, 8).sum()]
    fig = wti.plot_weight_evolution(history)
    assert len(fig.axes[0].patches) == 16
