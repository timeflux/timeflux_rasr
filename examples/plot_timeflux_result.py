"""Compare what the Timeflux graph saved with the known clean EEG.

Run after ``timeflux -d examples/simulated.yaml``.

Replay shifts all timestamps to the time of the run, so the saved signals
are matched to the simulation by position: the cleaned windows carry the
same timestamps as the raw samples they come from.

Usage: python examples/plot_timeflux_result.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SFREQ = 250

with pd.HDFStore(HERE / "data" / "simulated.hdf5", "r") as store:
    truth = store["/truth/eeg"].to_numpy().T
    bad = store["/truth/artifacts"].to_numpy()
with pd.HDFStore(HERE / "results" / "simulated_cleaned.hdf5", "r") as store:
    raw_saved = store["/raw"]
    cleaned_saved = pd.concat(
        [
            store[k]
            for k in store.keys()
            if k.startswith("/cleaned/") and not k.endswith("events")
        ]
    )

cleaned_saved = cleaned_saved[~cleaned_saved.index.duplicated()].sort_index()
position = raw_saved.index.get_indexer(cleaned_saved.index)
raw = raw_saved.to_numpy().T
cleaned = np.full_like(raw, np.nan)
cleaned[:, position] = cleaned_saved.to_numpy().T
done = ~np.isnan(cleaned[0])
test = np.arange(raw.shape[1]) >= raw.shape[1] // 2  # second minute


def error(X, mask):
    return np.linalg.norm(X[:, mask] - truth[:, mask]) / np.linalg.norm(truth[:, mask])


art, clean = done & test & bad, done & test & ~bad
print(
    f"cleaned {done[test].mean():.0%} of the test minute ({done[test].sum() / SFREQ:.1f} s)"
)
print(
    f"windows with artifacts: error {error(raw, art):.2f} raw -> {error(cleaned, art):.2f}"
)
print(
    f"clean windows:          error {error(raw, clean):.2f} raw -> {error(cleaned, clean):.2f}"
)

# Figure: 6 s of the test minute with the most artifacts, the 6 most affected channels
t = np.arange(raw.shape[1]) / SFREQ
starts = np.flatnonzero(done & test)[:: 6 * SFREQ]
best = max(starts, key=lambda s: bad[s : s + 6 * SFREQ].sum())
sl = slice(best, best + 6 * SFREQ)
channels = sorted(
    np.argsort(np.abs(raw[:, sl] - truth[:, sl]).sum(1))[-6:]
)  # most affected
step = 6 * truth.std()
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, ch in enumerate(channels):
    y0 = -i * step
    ax.plot(
        t[sl],
        raw[ch, sl] + y0,
        color="#c2410c",
        lw=0.9,
        label="raw, replayed by Timeflux" if i == 0 else None,
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
        label="cleaned by the graph" if i == 0 else None,
    )
edges = np.flatnonzero(np.diff(np.r_[0, bad[sl].astype(int), 0]))
for a, b in zip(edges[::2], edges[1::2]):
    ax.axvspan(t[sl][a], t[sl][b - 1], color="#fde68a", alpha=0.5, lw=0)
ax.set_yticks([-i * step for i in range(len(channels))])
ax.set_yticklabels([f"ch{c + 1}" for c in channels])
ax.set_ylim(-(len(channels) - 0.4) * step, 1.6 * step)
ax.set_xlabel("time in the recording (s)")
ax.legend(loc="upper right", ncol=3, frameon=False)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title(
    "Output of examples/simulated.yaml: artifact windows in yellow",
    loc="left",
    fontsize=11,
)
fig.tight_layout()
out = HERE.parent / "doc" / "static" / "img" / "timeflux_simulated.png"
fig.savefig(out, dpi=130)
print(f"figure saved to {out}")
