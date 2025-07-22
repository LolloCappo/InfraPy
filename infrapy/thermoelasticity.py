import numpy as np

def LIA(data, fs, fl, method='band', df=0.0, df_res=0.1):
    """
    Lock-In Analyzer to extract magnitude and phase of a thermal response,
    using time-domain (single or narrowband) or FFT-based approach.

    Parameters:
    -----------
    data : ndarray [frames, height, width]
        Time sequence of thermal images.
    fs : float
        Sampling frequency [Hz].
    fl : float
        Target excitation frequency [Hz].
    method : str
        'band' for time-domain lock-in, or 'fft' for frequency-domain (FFT) lock-in.
    df : float
        Half-width of the frequency band around fl [Hz]. Set to 0 for single-frequency lock-in.
    df_res : float
        Frequency resolution within the band [Hz] (only for 'band' method).

    Returns:
    --------
    magnitude : ndarray [height, width]
        Magnitude of response at (or near) frequency fl.
    phase : ndarray [height, width]
        Phase of response at (or near) frequency fl [degrees].
    f_used : float or ndarray
        Frequency/frequencies actually used in the analysis.
    """

    N = data.shape[0]
    t = np.arange(N) / fs

    if method == 'band':
        if df == 0.0:
            freqs = np.array([fl])
        else:
            freqs = np.arange(fl - df, fl + df + df_res, df_res)

        Re_total = np.zeros(data.shape[1:], dtype=np.float64)
        Img_total = np.zeros(data.shape[1:], dtype=np.float64)

        for f in freqs:
            sine = np.sin(2 * np.pi * f * t)[:, None, None]
            cosine = np.cos(2 * np.pi * f * t)[:, None, None]

            L1 = sine * data
            L2 = cosine * data

            Re = 2 * np.trapz(L1, dx=1/fs, axis=0)
            Img = 2 * np.trapz(L2, dx=1/fs, axis=0)

            Re_total += Re
            Img_total += Img

        Re_avg = Re_total / len(freqs)
        Img_avg = Img_total / len(freqs)

        magnitude = np.sqrt(Re_avg**2 + Img_avg**2)
        phase = np.arctan2(Img_avg, Re_avg) * 180 / np.pi
        return magnitude, phase, freqs if len(freqs) > 1 else freqs[0]

    elif method == 'fft':
        freqs = np.fft.rfftfreq(N, d=1/fs)
        fft_data = np.fft.rfft(data, axis=0)

        idx = np.argmin(np.abs(freqs - fl))
        f_target = freqs[idx]
        F = fft_data[idx]

        magnitude = 2 * np.abs(F) / N
        phase = np.angle(F, deg=True)
        return magnitude, phase, f_target

    else:
        raise ValueError("Invalid method. Choose 'band' or 'fft'.")
