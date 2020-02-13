import mne
import numpy as np
import os
from utils.viz import (plot_all_mne_data, plot_time_dist)
rdn_seed = 42
home = os.path.expanduser("~")
from timeit import default_timer as timer


def test_plot_all_mne_data_unitary():
    """test if outputs correctly figures"""
    output_folder = os.path.expanduser("~")
    title = "test"
    extension = ".png"
    files = []
    files.append(os.path.join(output_folder, f"{title}_raw{extension}"))
    files.append(os.path.join(output_folder, f"{title}_psd{extension}"))
    files.append(os.path.join(output_folder, f"{title}_dist{extension}"))

    # delete file if existing
    for file in files:
        if os.path.exists(file):
            os.remove(file)

    np.random.rand(rdn_seed)
    data = np.random.randn(4, 1024) * 1e-6
    info = mne.create_info(["ch1", "ch2", "ch3", "ch4"], 256, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    plot_all_mne_data(raw, output_folder=output_folder, title=title)

    assert all([os.path.exists(file) for file in files])

def test_plot_time_dist_unitary():
    output_folder = os.path.expanduser("~")
    title = "test_time"
    extension = ".png"
    files = []
    files.append(os.path.join(output_folder, f"{title}_dist{extension}"))
    # delete file if existing
    for file in files:
        if os.path.exists(file):
            os.remove(file)
    np.random.rand(rdn_seed)
    data = np.random.randn(100, 1024, 8)
    time_table = - np.ones((data.shape[0], 1))  # initialize time table
    for n_epoch in range(data.shape[0]):
        start_in = timer()
        data[n_epoch, :, :] = data[n_epoch, :, :] ** 2
        time_table[n_epoch] = timer() - start_in

    plot_time_dist(time_table, output_folder=output_folder, title=title)
    assert all([os.path.exists(file) for file in files])

