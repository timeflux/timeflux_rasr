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

sns.set(font_scale=1)

print("Config LOADED")

if __name__ == '__main__':
    # Kick-Off Notebook for rASR implementation
    """
    This notebook is a very crude and straightforward dive into the data provided by Sarah Blum on rASR. It's main purpose is to share with you some useful functions I have already, so that you save some time. 
    It consists in: 
    - loading the eeg signal (and events?) from the *xdf* and *set* files
    - visualizing the power spectral density (check dead channels and line noise) 
    - apply a IIR filter using either timeflux node or mne functions 
    - apply a rolling window and apply any method from a sklearn estimator 
    - leave implementation of rASR as in a sklearn estimator as a TODO thing 
    - go back to pandas and plot the pre/post signal 
    """

    # Load xdf and extract eeg stream
    print("Load EEG files")
    k_file = 0

    raw_xdf_fname = Config.raw_files[k_file]
    streams, _ = load_xdf(raw_xdf_fname)
    stream_names = get_stream_names(streams)
    df_eeg_raw = extract_signal_stream(streams, 'Android_EEG_010026')
    df_presentation_events = extract_signal_stream(streams, 'Presentation_Markers')
    print(df_eeg_raw.head())

    print("RAW EEG LOADED")

    ## select EEG channels
    eeg_columns = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'FC1', 'FC2', 'Cz', 'C3', 'C4', 'T7',
                   'T8', 'CPz', 'CP1', 'CP2', 'CP5', 'CP6', 'Tp9', 'Tp10', 'Pz', 'P3',
                   'P4', 'O1', 'O2']
    df_eeg_raw = df_eeg_raw.loc[:, eeg_columns]

    print("EEG channels selected")

    ## Convert float index in datetime, estimate duration/srate
    """
    - convert float index in datetime
    - estimate duration/srate
    """
    # %%



    df_eeg_raw = float_index_to_time_index(df_eeg_raw)

    duration = (df_eeg_raw.index[-1] - df_eeg_raw.index[0]).total_seconds() / 60
    rate = estimate_rate(df_eeg_raw)

    print(f'Duration of session was {duration} min. \n ' +
          f'Rate is of {rate} Hz.')

    # %% md

    ### Convert from pandas to mne, and eventually filter using timeflux or mne

    # %%

    # df_eeg_raw = pd.read_hdf(fname, '/eeg/raw') # TODO: adapt hdf5 groups
    # df_events = pd.read_hdf(fname, '/events/speller') # TODO: adapt hdf5 groups
    # baseline_events = df_events[df_events.label.str.contains('baseline')] # TODO: adapt calib times
    # calib_times = baseline_events.index[0], baseline_events.index[-1] # TODO: adapt calib times

    bad_ch = []

    mne_eeg_raw, mene_event_id, mne_picks = pandas_to_mne(df_eeg_raw, rate=rate, bad_ch=bad_ch)

    # Either filter using mne offline tool (higher order)
    # ----------------------------------------------------
    mne_eeg_filtered = mne_eeg_raw.copy().filter(1, 30)
    df_eeg_filtered = pd.DataFrame(mne_eeg_filtered.to_data_frame().values,
                                   df_eeg_raw.index, df_eeg_raw.columns)

    # Or mimick online with timeflux
    # ------------------------------
    # notch + bandpass in realtime world
    # =======================================================================#
    # ======= UNCOMMENT THESE LINES TO DO SO ================================#
    # from timeflux_dsp.nodes.filters import IIRFilter, IIRLineFilter
    ## apply a line noise filter
    # line = IIRLineFilter(rate=rate, edges_center=(50, 100))
    # line.i.data = df_eeg_raw
    # line.update()
    ## apply a bandpass filter
    # bandpass = IIRFilter(rate=rate, order=3, frequencies=[1, 30], filter_type='bandpass')
    # bandpass.i.data = line.o.data
    # bandpass.update()
    # convert dataframe to mne
    # df_eeg_filtered, _, _ = pandas_to_mne(bandpass.o.data, sfreq=rate, bad_ch=bad_ch)
    # =======================================================================#

    # %% md

    ### Visualize power spectral density

    # %%

    mne_eeg_raw.plot_psd();
    plt.suptitle('Raw');

    mne_eeg_filtered.plot_psd();
    plt.suptitle('Filtered');

    ## Load filtered data


    mne_eeg_filtered = read_raw_eeglab(Config.filtered_files[k_file])
    df_eeg_filtered = mne_eeg_filtered.to_data_frame()
    mne_eeg_filtered.plot_psd();
    mne_eeg_filtered.plot();


    ## Load calibration data
    mne_eeg_calibration = read_raw_eeglab(Config.calibration_files[k_file])
    df_eeg_calibration = mne_eeg_calibration.to_data_frame()
    mne_eeg_calibration.plot();

    ## Load rASR output

    mne_eeg_cleaned = read_raw_eeglab(Config.riemannian_asr_out_files[k_file])
    mne_eeg_cleaned.plot();

    df_eeg_cleaned = mne_eeg_cleaned.to_data_frame()
    df_eeg_cleaned.head()

    # Epoch the data

    # %%

    size = int(rate * 3)  # size of window in samples
    interval = size  # step interval in samples

    # convert filtered data into epochs
    np_eeg_filtered_epochs = epoch(df_eeg_filtered.values, size, interval, axis=0)  # (n_channels,  n_times, n_trials)
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

    # %%

    # Let's say we calibrate on the 20 first epochs and test on the rest
    # TODO : refine this
    X_fit = np_eeg_calibration_epochs
    X_test = np_eeg_filtered_epochs
    print("copied epochs")

    # %% md

    # %%

    # %% md

    # RASR tests

    # %%

    rASR_pipeline = make_pipeline(RASR())
    print(X_fit.shape)
    print(X_test.shape)
    X_test_transformed = rASR_pipeline.fit(X_fit).transform(X_test)

    # %%

    k_ch = 8
    k_epoch = 10

    plt.figure()
    plt.plot(X_test[k_epoch, :, k_ch])
    plt.plot(X_test_transformed[k_epoch, :, k_ch], '--')
    plt.suptitle(f'Pre/post from pipeline on one chunk of EEG signal')
    plt.xlabel('time (sample)')
    plt.show()

    # %% md

    ## Back to signal

    # %%

    values = X_test_transformed.reshape(-1, X_test_transformed.shape[-1])

    # %%

    df_eeg_cleaned_v2 = pd.DataFrame(values, columns=df_eeg_raw.columns)
    df_eeg_cleaned_v2.head()

    # %% md
