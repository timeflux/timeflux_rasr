"""Comparison between RASR Matlab and Python outputs.
It is a refactoring of the jupyter notebook and it is extended to include more advanced metrics and figure outputs.
"""

from utils.config import Config
from collections import OrderedDict
from glob import glob
import os
import pandas as pd
import numpy as np
import mne
from mne.io import read_raw_eeglab
import seaborn as sns
import matplotlib.pyplot as plt
from pyxdf import load_xdf
from utils.utils import (epoch, get_stream_names, extract_signal_stream, float_index_to_time_index, estimate_rate,
                pandas_to_mne)
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.channelselection import FlatChannelRemover
from timeflux_rasr.estimation import RASR
from utils.viz import plot_all_mne_data
sns.set(font_scale=1)

print("Config LOADED")

if __name__ == '__main__':
    # Kick-Off Notebook for rASR implementation
    """
    This is the script version for comparison of our implementation with Sarah Blum's matlab on rASR. 
    It consists in: 
    - loading the eeg signal (and events?) from the *xdf* and *set* files
    - visualizing the power spectral density (check dead channels and line noise) 
    - apply a IIR filter using either timeflux node or mne functions 
    - epoching and apply any method from the RASR sklearn estimator 
    - comparison both qualitative and quantitative of the RASR output
    - save the figure and results
    """

    # Load xdf and extract eeg stream
    print("Load EEG files")
    for k_file in range(len(Config.raw_files)):
        # LOAD DATA

        ## Load raw data
        raw_xdf_fname = Config.raw_files[k_file]
        streams, _ = load_xdf(raw_xdf_fname)
        stream_names = get_stream_names(streams)
        df_eeg_raw = extract_signal_stream(streams, 'Android_EEG_010026')
        df_presentation_events = extract_signal_stream(streams, 'Presentation_Markers')
        eeg_columns = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'FC1', 'FC2', 'Cz', 'C3', 'C4', 'T7',
                       'T8', 'CPz', 'CP1', 'CP2', 'CP5', 'CP6', 'Tp9', 'Tp10', 'Pz', 'P3',
                       'P4', 'O1', 'O2']
        df_eeg_raw = df_eeg_raw.loc[:, eeg_columns]
        bad_ch = []
        df_eeg_raw = float_index_to_time_index(df_eeg_raw)

        duration = (df_eeg_raw.index[-1] - df_eeg_raw.index[0]).total_seconds() / 60
        rate = estimate_rate(df_eeg_raw)
        mne_eeg_raw, mene_event_id, mne_picks = pandas_to_mne(df_eeg_raw, rate=rate, bad_ch=bad_ch)
        mne_eeg_filtered_from_raw = mne_eeg_raw.copy().filter(1, 30)

        ## Load filtered data
        mne_eeg_filtered = read_raw_eeglab(Config.filtered_files[k_file])
        df_eeg_filtered = mne_eeg_filtered.to_data_frame()

        ## Load calibration data
        mne_eeg_calibration = read_raw_eeglab(Config.calibration_files[k_file])
        df_eeg_calibration = mne_eeg_calibration.to_data_frame()

        ## Load rASR output

        mne_eeg_cleaned = read_raw_eeglab(Config.riemannian_asr_out_files[k_file])
        df_eeg_cleaned = mne_eeg_cleaned.to_data_frame()

        size = int(rate * 3)  # size of window in samples
        interval = size  # step interval in samples

        # convert filtered data into epochs
        np_eeg_filtered_epochs = epoch(df_eeg_filtered, size, interval, axis=0)  # (n_channels,  n_times, n_trials)
        print("shape test data")
        print(np_eeg_filtered_epochs.shape)
        # np_eeg_filtered_epochs = np.swapaxes(np_eeg_filtered_epochs, 0, 2 ) # (n_trials, n_channels, n_times)

        # convert calibration data into epochs
        np_eeg_calibration_epochs = epoch(df_eeg_calibration.values, size, interval,
                                          axis=0)  # (n_channels,  n_times, n_trials)
        # np_eeg_calibration_epochs = np.swapaxes(np_eeg_calibration_epochs, 0, 2 )# (n_trials, n_channels, n_times)
        print("shape training data")
        print(np_eeg_calibration_epochs.shape)

        # %% md

        ## RASR IMPLEMENTATION

        X_fit = np_eeg_calibration_epochs
        X_test = np_eeg_filtered_epochs

        rASR_pipeline = make_pipeline(RASR(rejection_cutoff=20, max_dimension=0.33))
        X_test_transformed = rASR_pipeline.fit(X_fit).transform(X_test)

        mne_eeg_rasr_info = mne_eeg_filtered.info
        data = X_test_transformed.reshape(X_test_transformed.shape[0] * X_test_transformed.shape[1], -1).transpose()
        mne_eeg_rasr = mne.io.RawArray(data * 1e-6, mne_eeg_rasr_info)

        #comparison
        title = "s" + str(k_file) + "_filtered"
        plot_all_mne_data(mne_eeg_filtered, Config.results_folder, title)

        plt.close()
        title = "s" + str(k_file) + "_RASR_matlab"
        plot_all_mne_data(mne_eeg_cleaned, Config.results_folder, title)

        title = "s" + str(k_file) + "_RASR_matlab_diff"
        eeg_rasr_diff = mne_eeg_filtered[:, 0:len(mne_eeg_cleaned)][0] - mne_eeg_cleaned.get_data()
        mne_eeg_rasr_diff = raw = mne.io.RawArray(data=eeg_rasr_diff, info=mne_eeg_cleaned.info, verbose=False)
        plot_all_mne_data(mne_eeg_rasr_diff, Config.results_folder, title)

        plt.close()
        title = "s" + str(k_file) + "_RASR_python"
        plot_all_mne_data(mne_eeg_rasr, Config.results_folder, title)

        title = "s" + str(k_file) + "_RASR_python_diff"
        eeg_rasr_diff = mne_eeg_filtered[:, 0:len(mne_eeg_rasr)][0] - mne_eeg_rasr.get_data()
        mne_eeg_rasr_diff = raw = mne.io.RawArray(data=eeg_rasr_diff, info=mne_eeg_filtered.info, verbose=False)
        plot_all_mne_data(mne_eeg_rasr_diff, Config.results_folder, title)
