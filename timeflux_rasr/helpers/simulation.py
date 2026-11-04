"""Hybrid synthetic EEG: clean EEG with known artifacts.

Used by the documentation, the examples and the validation, so that every
figure can be reproduced without any recording. Only NumPy and SciPy are
needed.

The clean EEG is a random linear mixture of oscillatory sources (6-30 Hz,
slowly modulated) and 1/f sources, plus a little sensor noise, band-passed
at 1-45 Hz. Artifacts with a known time course are added on a fraction of
fixed-length windows:

* ``blink``: slow (1-3 Hz) burst, strongest on the first channels
  (think frontal electrodes);
* ``muscle``: broadband (20-60 Hz) burst on 1 to 3 neighbouring channels;
* ``pop``: a step on one channel (electrode pop).

Because the clean signal is known, the error of any cleaning method can be
measured exactly.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

ARTIFACTS = ("blink", "muscle", "pop")


def _pink(n, rng):
    f = np.fft.rfftfreq(n)
    spec = (rng.randn(len(f)) + 1j * rng.randn(len(f))) / np.maximum(f, 1 / n) ** 0.5
    x = np.fft.irfft(spec, n)
    return x / x.std()


def mixing_matrix(n_channels=16, n_sources=20, random_state=None):
    """Random source-to-channel mixing matrix, shape (n_channels, n_sources)."""
    rng = np.random.RandomState(random_state)
    return rng.randn(n_channels, n_sources)


def clean_eeg(n_times, sfreq=250, mixing=None, random_state=None):
    """Simulate clean EEG.

    Parameters
    ----------
    n_times : int
        Number of samples.
    sfreq : float (default: 250)
        Sampling rate, in Hz.
    mixing : ndarray, shape (n_channels, n_sources) | None
        Mixing matrix; use the same one for calibration and test data so
        that both come from the same "head". Default: 16 channels, 20
        sources.
    random_state : int | None
        Seed.

    Returns
    -------
    X : ndarray, shape (n_channels, n_times)
    """
    rng = np.random.RandomState(random_state)
    if mixing is None:
        mixing = mixing_matrix(random_state=rng.randint(2**31))
    n_channels, n_sources = mixing.shape
    t = np.arange(n_times) / sfreq
    S = np.empty((n_sources, n_times))
    for i in range(n_sources):
        if i % 2:
            S[i] = _pink(n_times, rng)
        else:
            f = rng.uniform(6, 30)
            env = 1 + 0.5 * np.sin(
                2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.rand() * 6
            )
            S[i] = env * np.sin(2 * np.pi * f * t + rng.rand() * 6)
            S[i] += 0.3 * _pink(n_times, rng)
    X = mixing @ S + 0.1 * rng.randn(n_channels, n_times)
    sos = butter(4, [1, 45], "bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=1)


def add_artifacts(
    X, sfreq=250, window=125, rate=0.2, amplitude=8.0, offset=0, random_state=None
):
    """Add artifacts on a fraction of fixed-length windows.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Clean EEG.
    sfreq : float (default: 250)
        Sampling rate, in Hz.
    window : int (default: 125)
        Window length, in samples; each artifact fills one window.
    rate : float (default: 0.2)
        Fraction of windows with an artifact.
    amplitude : float (default: 8.0)
        Artifact amplitude, in units of the EEG standard deviation.
    offset : int (default: 0)
        Shift of the artifact windows, in samples (to misalign them with the
        windows used by the cleaning method).
    random_state : int | None
        Seed.

    Returns
    -------
    X_contaminated : ndarray, shape (n_channels, n_times)
    mask : ndarray of bool, shape (n_times,)
        True on the samples that carry an artifact.
    kinds : list of str
        Kind of each artifact, in time order.
    """
    rng = np.random.RandomState(random_state)
    n_channels = X.shape[0]
    art = np.zeros_like(X)
    n_win = (X.shape[1] - offset) // window
    bad = rng.rand(n_win) < rate
    scale = X.std()
    tt = np.arange(window) / sfreq
    kinds = []
    for w in np.flatnonzero(bad):
        sl = slice(offset + w * window, offset + (w + 1) * window)
        kind = ARTIFACTS[rng.randint(len(ARTIFACTS))]
        kinds.append(kind)
        if kind == "blink":
            pattern = np.abs(rng.randn(n_channels)) * np.linspace(1, 0.1, n_channels)
            wave = np.sin(np.pi * tt / tt[-1]) ** 2
            wave = wave * np.sin(2 * np.pi * rng.uniform(1, 3) * tt)
            art[:, sl] += amplitude * scale * np.outer(pattern / pattern.max(), wave)
        elif kind == "muscle":
            chans = (
                rng.randint(n_channels) + np.arange(rng.randint(1, 4))
            ) % n_channels
            sos = butter(
                4, [20, min(60, 0.45 * sfreq)], "bandpass", fs=sfreq, output="sos"
            )
            burst = sosfiltfilt(sos, rng.randn(len(chans), window), axis=1)
            burst /= burst.std()
            art[chans, sl] += amplitude * scale * burst * np.hanning(window)
        else:
            ch = rng.randint(n_channels)
            step = np.arange(window) > rng.randint(window // 4, window // 2)
            art[ch, sl] += amplitude * scale * step
    mask = np.zeros(X.shape[1], bool)
    mask[offset : offset + n_win * window] = np.repeat(bad, window)
    return X + art, mask, kinds


def to_epochs(X, window):
    """Cut (n_channels, n_times) into non-overlapping (n_epochs, window, n_channels)."""
    n_channels = X.shape[0]
    n = X.shape[1] // window
    return X[:, : n * window].reshape(n_channels, n, window).transpose(1, 2, 0)


def from_epochs(E):
    """Inverse of `to_epochs`: (n_epochs, window, n_channels) to (n_channels, n_times)."""
    return E.transpose(2, 0, 1).reshape(E.shape[2], -1)
