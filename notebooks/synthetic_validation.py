"""Quantitative validation of RASR on hybrid synthetic EEG (issue #29).

Clean EEG is simulated as a random mixture of oscillatory and 1/f sources.
Artifacts with known time course are added on a fraction of the windows:
  * blink-like: slow (1-3 Hz) bursts with a frontal-like spatial pattern,
  * muscle-like: broadband (20-60 Hz) bursts on 1-3 neighbouring channels,
  * pops: a step on one channel.
Because the clean signal is known, we can measure for each method
  * artifact removal: relative error ||Xhat - clean|| / ||clean|| on the
    contaminated windows (raw data: the artifact-to-EEG ratio);
  * distortion: the same error on artifact-free windows (should stay ~0).

Methods: RASR (this package), and meegkit's ASR (euclid and riemann) as an
independent reference implementation of the MATLAB code.

Usage: python notebooks/synthetic_validation.py [n_seeds]
"""

import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from timeflux_rasr.estimators.rasr import RASR

SFREQ, N_CH, WIN = 250, 16, 125  # 0.5 s windows


def pink(n, rng):
    f = np.fft.rfftfreq(n)
    spec = (rng.randn(len(f)) + 1j * rng.randn(len(f))) / np.maximum(f, 1 / n) ** 0.5
    x = np.fft.irfft(spec, n)
    return x / x.std()


def clean_eeg(n_times, rng, A):
    n_src = A.shape[1]
    t = np.arange(n_times) / SFREQ
    S = np.empty((n_src, n_times))
    for i in range(n_src):
        if i % 2:
            S[i] = pink(n_times, rng)
        else:
            f = rng.uniform(6, 30)
            env = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.rand() * 6)
            S[i] = env * np.sin(2 * np.pi * f * t + rng.rand() * 6) + 0.3 * pink(n_times, rng)
    X = A @ S + 0.1 * rng.randn(N_CH, n_times)
    sos = butter(4, [1, 45], "bandpass", fs=SFREQ, output="sos")
    return sosfiltfilt(sos, X, axis=1)


def add_artifacts(X, rng, rate, amp, offset=0):
    """Add artifacts on a fraction ``rate`` of the WIN-sample windows, shifted by ``offset``."""
    art = np.zeros_like(X)
    n_win = (X.shape[1] - offset) // WIN
    bad = rng.rand(n_win) < rate
    scale = X.std()
    for w in np.flatnonzero(bad):
        sl = slice(offset + w * WIN, offset + (w + 1) * WIN)
        kind = rng.choice(["blink", "muscle", "pop"])
        tt = np.arange(WIN) / SFREQ
        if kind == "blink":
            pattern = np.abs(rng.randn(N_CH)) * np.linspace(1, 0.1, N_CH)
            wave = np.sin(np.pi * tt / tt[-1]) ** 2 * np.sin(2 * np.pi * rng.uniform(1, 3) * tt)
            art[:, sl] += amp * scale * np.outer(pattern / pattern.max(), wave)
        elif kind == "muscle":
            chans = (rng.randint(N_CH) + np.arange(rng.randint(1, 4))) % N_CH
            sos = butter(4, [20, 60], "bandpass", fs=SFREQ, output="sos")
            burst = sosfiltfilt(sos, rng.randn(len(chans), WIN), axis=1)
            burst /= burst.std()
            art[chans, sl] += amp * scale * burst * np.hanning(WIN)
        else:
            ch = rng.randint(N_CH)
            art[ch, sl] += amp * scale * (np.arange(WIN) > rng.randint(WIN // 4, WIN // 2))
    mask = np.zeros(X.shape[1], bool)
    mask[offset:offset + n_win * WIN] = np.repeat(bad, WIN)
    return X + art, mask


def epochs(X):
    n = X.shape[1] // WIN
    return X[:, : n * WIN].reshape(N_CH, n, WIN).transpose(1, 2, 0)  # (trials, samples, ch)


def unepoch(E):
    return E.transpose(2, 0, 1).reshape(N_CH, -1)


def run(seed, rate=0.2, amp=8.0, calib_rate=0.0, offset=0, cutoffs=(3, 5, 10)):
    rng = np.random.RandomState(seed)
    A = rng.randn(N_CH, 20)
    calib = clean_eeg(SFREQ * 120, rng, A)
    calib_c, _ = add_artifacts(calib, rng, calib_rate, amp)
    test = clean_eeg(SFREQ * 120, rng, A)
    test_c, bad = add_artifacts(test, rng, rate, amp, offset)
    n = (test.shape[1] // WIN) * WIN
    test, test_c, bad = test[:, :n], test_c[:, :n], bad[:n]

    out = {"raw": test_c}
    for metric in ["euclid", "riemann"]:
        for c in cutoffs:
            r = RASR(rejection_cutoff=c, metric=metric).fit(epochs(calib_c))
            out[f"RASR-{metric} c={c}"] = unepoch(r.transform(epochs(test_c)))
    try:
        from meegkit.asr import ASR
        for method in ["euclid", "riemann"]:
            for c in cutoffs:
                asr = ASR(sfreq=SFREQ, cutoff=c, method=method)
                asr.fit(calib_c)
                # stateful, one covariance per call: stream it window by window
                out[f"meegkit-{method} c={c}"] = np.concatenate(
                    [asr.transform(test_c[:, i:i + WIN]) for i in range(0, n, WIN)], axis=1)
    except ImportError:  # meegkit missing, or pymanopt for method='riemann'
        pass

    rows = []
    for m, Xh in out.items():
        err = Xh - test
        rows.append(dict(method=m, seed=seed, calib_rate=calib_rate, offset=offset,
                         artifact_err=float(np.linalg.norm(err[:, bad]) / np.linalg.norm(test[:, bad])),
                         clean_err=float(np.linalg.norm(err[:, ~bad]) / np.linalg.norm(test[:, ~bad]))))
    return rows


def main(n_seeds=5):
    rows = []
    for offset in [0, WIN // 2]:  # artifacts aligned with RASR's windows, or straddling two
        for calib_rate in [0.0, 0.2]:
            for seed in range(n_seeds):
                rows += run(seed, calib_rate=calib_rate, offset=offset)
    df = pd.DataFrame(rows)
    print(df.groupby(["offset", "calib_rate", "method"])[["artifact_err", "clean_err"]].median()
          .round(3).to_string())
    return df


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
