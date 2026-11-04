"""Write a simulated EEG recording that Timeflux can replay.

The file ``examples/data/simulated.hdf5`` holds what a real recording made
with Timeflux would hold:

* ``/eeg/raw``: 16 channels at 250 Hz, a DataFrame indexed by time:
  1 minute of clean calibration EEG, then 1 minute of EEG with blinks,
  muscle bursts and electrode pops on 20 % of the 0.5 s windows;
* ``/events``: the two markers that tell RASR when to learn,
  ``calibration_begins`` and ``calibration_ends``.

It also stores what a real recording cannot give, to measure the result:

* ``/truth/eeg``: the same 2 minutes without artifacts;
* ``/truth/artifacts``: True on the samples that carry an artifact.

Usage: python examples/simulate_recording.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from timeflux_rasr.helpers.simulation import add_artifacts, clean_eeg, mixing_matrix

SFREQ, WIN = 250, 125
CALIBRATION, TEST = 60, 60  # seconds

mixing = mixing_matrix(n_channels=16, random_state=0)
calibration = clean_eeg(CALIBRATION * SFREQ, SFREQ, mixing, random_state=1)
test = clean_eeg(TEST * SFREQ, SFREQ, mixing, random_state=2)
test_raw, test_bad, _ = add_artifacts(test, SFREQ, WIN, rate=0.2, random_state=3)

truth = np.hstack([calibration, test])
raw = np.hstack([calibration, test_raw])
bad = np.r_[np.zeros(calibration.shape[1], bool), test_bad]

start = pd.Timestamp("2026-01-01 12:00:00")
index = start + pd.to_timedelta(np.arange(raw.shape[1]) / SFREQ, unit="s")
channels = [f"ch{i + 1}" for i in range(raw.shape[0])]

# Timeflux events: a 'label' column and a 'data' column (JSON string)
events = pd.DataFrame(
    {"label": ["calibration_begins", "calibration_ends"], "data": ["{}", "{}"]},
    index=[index[0], index[calibration.shape[1] - 1]],
)

out = Path(__file__).parent / "data" / "simulated.hdf5"
out.parent.mkdir(exist_ok=True)
with pd.HDFStore(out, mode="w", complib="zlib", complevel=3) as store:
    store.put(
        "/eeg/raw", pd.DataFrame(raw.T, index=index, columns=channels), format="table"
    )
    store.put("/events", events, format="table")
    store.put(
        "/truth/eeg",
        pd.DataFrame(truth.T, index=index, columns=channels),
        format="table",
    )
    store.put("/truth/artifacts", pd.Series(bad, index=index), format="table")
print(
    f"wrote {out}: {raw.shape[1] / SFREQ:.0f} s, {raw.shape[0]} channels, {test_bad.mean():.0%} of the test with artifacts"
)
