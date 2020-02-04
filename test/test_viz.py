import mne
import numpy as np
import os
from utils.viz import (plot_all_mne_data, plot_time_dist)
rdn_seed = 42
home = os.path.expanduser("~")

def test_plot_all_mne_data_unitary():
    """test if output correctly figures"""
    output_folder = os.path.expanduser("~")
    title = "test"
    files = []
    files.append(os.path.join(output_folder,title + "_raw"))
    files.append(os.path.join(output_folder,title + "_psd"))
    files.append(os.path.join(output_folder,title + "_dist"))

    # delete file if existing
    any([os.path.exists(file) for file in files])
    np.random.rand(rdn_seed)
    data = np.random.randn(4, 1024) * 1e-6
    info = mne.create_info(["ch1", "ch2", "ch3", "ch4"], 256, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    plot_all_mne_data(raw, output_folder=output_folder, title=title)