"""Quick start: clean simulated EEG with RASR, no Timeflux needed.

1. Simulate 2 minutes of clean EEG for calibration and 1 minute of test
   EEG with blinks, muscle bursts and electrode pops on 20 % of the windows.
2. Cut both into 0.5 s windows (epochs), fit RASR on the calibration
   windows, and clean the test windows.
3. Compare with the known clean EEG, and save a figure.

Usage: python examples/quickstart.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from timeflux_rasr.estimators.rasr import RASR
from timeflux_rasr.helpers.simulation import (
    add_artifacts,
    clean_eeg,
    from_epochs,
    mixing_matrix,
    to_epochs,
)

SFREQ, WIN = 250, 125  # 250 Hz, 0.5 s windows

# 1. Simulated data: the same "head" (mixing matrix) for both recordings
mixing = mixing_matrix(n_channels=16, random_state=0)
calibration = clean_eeg(120 * SFREQ, SFREQ, mixing, random_state=1)
truth = clean_eeg(60 * SFREQ, SFREQ, mixing, random_state=2)
raw, bad, kinds = add_artifacts(truth, SFREQ, WIN, rate=0.2, random_state=3)

# 2. Fit on calibration windows, clean the test windows
rasr = RASR(rejection_cutoff=5).fit(to_epochs(calibration, WIN))
cleaned = from_epochs(rasr.transform(to_epochs(raw, WIN)))


# 3. How good is it?
def error(X, mask):
    return np.linalg.norm(X[:, mask] - truth[:, mask]) / np.linalg.norm(truth[:, mask])


print(f"{len(kinds)} artifacts ({', '.join(sorted(set(kinds)))})")
print(
    f"windows with artifacts: error {error(raw, bad):.2f} raw -> {error(cleaned, bad):.2f}"
)
print(
    f"clean windows:          error {error(raw, ~bad):.2f} raw -> {error(cleaned, ~bad):.2f}"
)

# Figure: 5 seconds around the first blink, 6 channels
t = np.arange(raw.shape[1]) / SFREQ
starts = np.flatnonzero(bad[::WIN]) * WIN  # first sample of each artifact window
center = starts[kinds.index("blink")]
sl = slice(max(center - 2 * SFREQ, 0), center + 3 * SFREQ)
channels = [0, 1, 2, 7, 8, 15]
step = 6 * truth.std()
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, ch in enumerate(channels):
    y0 = -i * step
    ax.plot(
        t[sl],
        raw[ch, sl] + y0,
        color="#c2410c",
        lw=0.9,
        label="raw (with artifacts)" if i == 0 else None,
    )
    ax.plot(
        t[sl],
        truth[ch, sl] + y0,
        color="0.2",
        lw=2.4,
        alpha=0.3,
        label="true clean EEG" if i == 0 else None,
    )
    ax.plot(
        t[sl],
        cleaned[ch, sl] + y0,
        color="#1d4ed8",
        lw=0.9,
        label="after RASR" if i == 0 else None,
    )
for start, kind in zip(starts, kinds):
    if sl.start <= start < sl.stop:
        ax.axvspan(t[start], t[start + WIN - 1], color="#fde68a", alpha=0.5, lw=0)
        ax.text(t[start + WIN // 2], 1.3 * step, kind, ha="center", fontsize=9)
ax.set_yticks([-i * step for i in range(len(channels))])
ax.set_yticklabels([f"ch {c + 1}" for c in channels])
ax.set_ylim(-(len(channels) - 0.4) * step, 2.2 * step)
ax.set_xlabel("time (s)")
ax.legend(loc="upper right", ncol=3, frameon=False)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title("Simulated EEG: artifact windows in yellow", loc="left", fontsize=11)
fig.tight_layout()
out = Path(__file__).parents[1] / "doc" / "static" / "img" / "quickstart.png"
fig.savefig(out, dpi=130)
print(f"figure saved to {out}")
