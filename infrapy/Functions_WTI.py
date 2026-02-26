import pandas as pd
import numpy as np
import pysfmov as sfmov
import matplotlib.pyplot as plt
import TelopsToolbox.utils.image_processing as ip
from scipy.ndimage import zoom
from tqdm import tqdm
from TelopsToolbox.hcc.readIRCam import read_ircam

def read_ir(filenames, skip_frames=1, return_timestamps=False):
    """
    Reads infrared camera data from one or more .hcc or .sfmov files and converts it to a stack of images.
    This function supports infrared images acquired using Telops (.hcc) and Flir (.sfmov) IR cameras.

    Args:
        filenames (str or list): Path(s) to the infrared camera data file(s) (.hcc or .sfmov).
        skip_frames (int): Number of frames to skip when processing. Defaults to 1 (no skipping).
        return_timestamps (bool): If True, include timestamps from metadata. Defaults to False.

    Returns:
        If `.hcc` files: 
            - combined_data (np.ndarray): A 3D array representing the stack of 2D infrared images.
                                          Shape: (num_frames, height, width).
            - combined_meta_data (pd.DataFrame): Metadata for each frame.
        If `.sfmov` file:
            - data (np.ndarray): A 3D array representing the infrared images from that file.
                                Shape: (num_frames, height, width)
            - meta_data (pd.DataFrame): Metadata for that file.
    """
    # Ensure filenames is a list
    if isinstance(filenames, str):
        filenames = [filenames]

    # Initialize storage for `.hcc` files only
    all_data = []
    all_meta_data = []
    
    hcc_files = [f for f in filenames if f.endswith('.hcc')]
    sfmov_files = [f for f in filenames if f.endswith('.sfmov')]

    # Process `.sfmov` files individually
    for filename in sfmov_files:
        try:
            data = sfmov.get_data(filename)
            meta_data = sfmov.get_meta_data(filename)
            return data, meta_data  # Direct return for `.sfmov`
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the file {filename}: {e}")

    # Process `.hcc` files (if any)
    for filename in hcc_files:
        try:
            frame_data, frame_headers, special_pixels, non_special_pixels = read_ircam(filename)
            image_stack = []
            meta_data = pd.DataFrame(frame_headers)

            for i in tqdm(range(0, len(frame_data), skip_frames), desc=f"Loading frames from {filename}", unit="frame"):
                this_header = meta_data.iloc[i]
                processed_image = ip.form_image(this_header, frame_data[i]).squeeze()
                image_stack.append(processed_image)

            data = np.stack(image_stack)

            if return_timestamps:
                meta_data['Formatted_Timestamp'] = pd.to_datetime(meta_data['POSIXTime'], unit='s')

            # Append results to the cumulative lists
            all_data.append(data)
            all_meta_data.append(meta_data)

        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the file {filename}: {e}")

    # If `.hcc` files were processed, combine their data
    if hcc_files:
        combined_data = np.concatenate(all_data, axis=0)
        combined_meta_data = pd.concat(all_meta_data, ignore_index=True)
        return combined_data, combined_meta_data

    # If no valid files were processed, raise an error
    raise ValueError("No valid `.hcc` or `.sfmov` files were found.")

def interpolate_to_match(big_matrix: np.ndarray, small_matrix: np.ndarray, method: str = "bilinear") -> np.ndarray:
    """
    Interpolates the smaller matrix so its shape matches the bigger matrix.
    Works for 2D (H, W) or 3D arrays (N, H, W) or (H, W, C).
    
    Parameters:
        big_matrix (np.ndarray): The reference matrix with the target shape.
        small_matrix (np.ndarray): The matrix to resize.
        method (str): Interpolation method. Options: 'nearest', 'bilinear', 'bicubic'.
    
    Returns:
        np.ndarray: Resized version of small_matrix with shape equal to big_matrix.
    """
    # Map method to scipy order
    method_map = {
        "nearest": 0,
        "bilinear": 1,
        "bicubic": 3
    }
    if method not in method_map:
        raise ValueError(f"Invalid method '{method}'. Choose from {list(method_map.keys())}.")
    
    # Compute zoom factors for each axis
    zoom_factors = []
    for i in range(len(small_matrix.shape)):
        if i < len(big_matrix.shape):
            zoom_factors.append(big_matrix.shape[i] / small_matrix.shape[i])
        else:
            zoom_factors.append(1.0)  # keep extra dimensions unchanged
    
    # Perform interpolation
    resized_matrix = zoom(small_matrix, zoom_factors, order=method_map[method])
    
    return resized_matrix

def thermoelasticity(data, fs, method="fft", foi=None, segment_length=None, overlap=0.25,
                         zero_pad=True, apply_window=True, segment=True):
    """
    Perform Thermoelasticity analysis using FFT (Welch-style) or Digital Lock-In Correlation (DLIC).
    
    Parameters:
        data : ndarray (frames, height, width)
        fs : int, sampling frequency
        method : str, "fft" or "lockin"
        foi : float, frequency of interest (Hz) [required for lockin]
        segment_length : int, frames per segment (ignored if segment=False)
        overlap : float, fraction overlap (0 to <1)
        zero_pad : bool, whether to zero-pad to next full second
        apply_window : bool, whether to apply Hann window
        segment : bool, whether to segment the data (Welch-style)
    
    Returns:
        amplitude_map : ndarray
            For FFT: amplitude at all frequencies (freq_bins, h, w)
            For lockin: amplitude map (h, w)
        freq : ndarray
            For FFT: frequency array
            For lockin: array with single value [foi]
    """
    frames = data.shape[0]

    # Zero-padding
    if zero_pad:
        target_length = int(np.ceil(frames / fs) * fs)
        if frames < target_length:
            pad_width = ((0, target_length - frames), (0, 0), (0, 0))
            data = np.pad(data, pad_width, mode='constant', constant_values=np.nan)
            frames = target_length

    # Window
    if apply_window:
        window = np.hanning(segment_length if segment else frames)
    else:
        window = np.ones(segment_length if segment else frames)

    # Segmentation setup
    if segment:
        segment_length = int(segment_length)
        step = int(segment_length * (1 - overlap))
        if step <= 0:
            raise ValueError("Overlap too high, step becomes zero or negative.")
        segments = [(start, start + segment_length) for start in range(0, frames - segment_length + 1, step)]
    else:
        segments = [(0, frames)]

    if method == "fft":
        amplitudes = []
        for start, end in segments:
            segment_data = data[start:end]
            segment_data = segment_data - np.nanmean(segment_data, axis=0)
            segment_data = segment_data * window[:, None, None]

            fft_segment = np.fft.rfft(segment_data, axis=0)
            amplitude_segment = np.abs(fft_segment) * 2 / (end - start)
            amplitudes.append(amplitude_segment)

        amplitude_avg = np.nanmean(amplitudes, axis=0)
        freq = np.fft.rfftfreq(segment_length if segment else frames, 1/fs)
        return amplitude_avg, freq

    elif method == "lockin":
        if foi is None:
            raise ValueError("foi (frequency of interest) must be provided for lockin method.")

        t = np.arange(segment_length if segment else frames) / fs
        ref_cos = np.cos(2 * np.pi * foi * t)
        ref_sin = np.sin(2 * np.pi * foi * t)

        amplitudes = []
        for start, end in segments:
            segment_data = data[start:end]
            segment_data = segment_data - np.nanmean(segment_data, axis=0)
            segment_data = segment_data * window[:, None, None]

            X = np.nansum(segment_data * ref_cos[:, None, None], axis=0)
            Y = np.nansum(segment_data * ref_sin[:, None, None], axis=0)

            amplitude = np.sqrt(X**2 + Y**2) * 2 / (end - start)
            amplitudes.append(amplitude)

        amplitude_avg = np.nanmean(amplitudes, axis=0)
        return amplitude_avg, np.array([foi])

    else:
        raise ValueError("Invalid method. Choose 'fft' or 'lockin'.")
    
def compute_ism(map1, map2, *, demean=True, on_fail=np.nan):
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

def mmtri_with_history(measurements, ground_truth=None, max_iter=5, 
                       convergence_threshold=0.99):
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
    filtered_map : np.ndarray, shape (H, W)
        Final filtered thermoelastic response
    weights : np.ndarray, shape (C,)
        Final weights for each measurement
    variance_map : np.ndarray, shape (H, W)
        Pixel-wise variance
    residual_map : np.ndarray, shape (H, W)
        Sum of absolute residuals
    history : dict
        Dictionary containing:
        - 'references': list of reference maps at each iteration
        - 'weights': list of weight arrays at each iteration
        - 'mac_sequential': MAC between consecutive references
        - 'mac_to_ground_truth': MAC to ground truth (if provided)
    """
    C, H, W = measurements.shape
    
    # Replace any NaN/Inf with 0
    #measurements = np.nan_to_num(measurements, nan=0.0, posinf=0.0, neginf=0.0)
    
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
        mac_to_ground_truth.append(compute_ism(reference, ground_truth))
    
    # Iterations 1 to max_iter: weighted median
    for iteration in range(max_iter):
        prev_reference = reference.copy()
        
        # Compute ISM weights
        ism_values = np.array([compute_ism(measurements[c], reference) 
                              for c in range(C)])
        weights = (ism_values / ism_values.sum() 
                  if ism_values.sum() > 0 else np.ones(C) / C)
        
        # Update reference
        for i in range(H):
            for j in range(W):
                reference[i, j] = weighted_median(measurements[:, i, j], weights)
        
        # Track history
        reference_history.append(reference.copy())
        weight_history.append(weights.copy())
        mac_seq = compute_ism(reference, prev_reference)
        mac_sequential.append(mac_seq)
        
        if ground_truth is not None:
            mac_to_ground_truth.append(compute_ism(reference, ground_truth))
        
        # Check convergence
        if mac_seq >= convergence_threshold:
            print(f"Converged at iteration {iteration + 1} (MAC = {mac_seq:.4f})")
            break
    
    # Compute variance and residual maps
    variance_map = np.nanvar(measurements, axis=0)
    
    # Initialize with NaNs (so they persist unless overwritten)
    residual_map = np.full((H, W), np.nan)

    for c in range(C):
        diff = np.abs(measurements[c] - reference)

        # Where residual_map is NaN and diff is finite → initialize with diff
        mask_init = np.isnan(residual_map) & np.isfinite(diff)
        residual_map[mask_init] = diff[mask_init]

        # Where both residual_map and diff are finite → accumulate
        mask_add = np.isfinite(residual_map) & np.isfinite(diff)
        residual_map[mask_add] += diff[mask_add]
    
    # Compile history
    history = {
        'references': reference_history,
        'weights': weight_history,
        'mac_sequential': np.array(mac_sequential),
        'mac_to_ground_truth': np.array(mac_to_ground_truth) 
                               if ground_truth is not None else None
    }
    
    return reference, weights, variance_map, residual_map, history

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
    return fig