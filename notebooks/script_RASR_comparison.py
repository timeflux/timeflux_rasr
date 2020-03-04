"""Comparison between RASR Matlab and Python outputs.
It is a refactoring of the jupyter notebook and it is extended to include more advanced metrics and figure outputs.
"""

from utils.config import Config as cfg
import mne
from mne.io import read_raw_eeglab
import seaborn as sns
import matplotlib.pyplot as plt
from pyxdf import load_xdf
from utils.utils import (epoch, get_stream_names, extract_signal_stream, float_index_to_time_index, estimate_rate,
                         pandas_to_mne, check_params)
from sklearn.pipeline import make_pipeline
from timeflux_rasr.estimation import RASR
from timeflux_blending.blending import Blending

from utils.viz import (plot_all_mne_data, plot_time_dist)
import logging
import os
from timeit import default_timer as timer
import numpy as np

sns.set(font_scale=1)
logging.info("Config LOADED")

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
    offset_test = 4     # in order to save more tests
    test_configuration = [{"window_len": 0.5, "window_overlap": 0.66, "rejection_cutoff": 3},
                          {"window_len": 0.5, "window_overlap": 0.66, "rejection_cutoff": 5},
                          {"window_len": 3, "window_overlap": 0.9, "rejection_cutoff": 3},
                          {"window_len": 3, "window_overlap": 0.9, "rejection_cutoff": 5}]

    for test_ind in range(len(test_configuration)):  # for looping though many test_configuration
        Config = cfg()  # initialize class
        Config.results_folder = os.path.join(Config.results_folder, f"test_{test_ind}")
        logging.basicConfig(filename=os.path.join(Config.results_folder, '_output.log'), level=logging.DEBUG)
        if not os.path.exists(Config.results_folder):
            os.mkdir(Config.results_folder)
        else:
            logging.debug("will overwrite previous test")
        # Load xdf and extract eeg stream
        logging.info("Load EEG files")
        for k_file in range(len(Config.raw_files)):
            # LOAD DATA
            ## Load training data
            mne_eeg_training = read_raw_eeglab(Config.calibration_files[k_file])
            df_eeg_training = mne_eeg_training.to_data_frame()

            ## Load test data
            mne_eeg_test = read_raw_eeglab(Config.filtered_files[k_file])
            df_eeg_test = mne_eeg_test.to_data_frame()

            ## test data cleaned with Matlab rASR

            mne_eeg_rasr_matlab = read_raw_eeglab(Config.riemannian_asr_out_files[k_file])
            df_eeg_rasr_matlab = mne_eeg_rasr_matlab.to_data_frame()

            # PREPARE TRAINING AND TEST EPOCHS
            window_size = int(test_configuration[test_ind]["srate"] * test_configuration[test_ind]["window_len"])
            window_overlap = int(window_size * test_configuration[test_ind]["window_overlap"])
            window_interval = window_size - window_overlap  # step interval in samples

            # convert training data into epochs
            X_training = epoch(df_eeg_training.values, window_size, window_interval, axis=0)

            # convert test data into epochs
            X_test = epoch(df_eeg_test, window_size, window_interval, axis=0)

            ## RASR IMPLEMENTATION

            rASR_pipeline = make_pipeline(RASR(**check_params(RASR, **test_configuration[test_ind])))
            blending_pipeline = make_pipeline(Blending(window_overlap=window_overlap, merge=True))
            logging.info("Pipeline initialized")
            start = timer()
            rASR_pipeline = rASR_pipeline.fit(X_training)
            end = timer()
            print(f"test_{test_ind}: Pipeline fitted in {end - start}s ({(end - start) / X_training.shape[0]}s/epoch)")

            X_test_transformed = np.zeros(X_test.shape)
            start = timer()
            time_table = - np.ones((X_test.shape[0], 1))  # initialize time table
            for n_epoch in range(X_test.shape[0]):
                start_in = timer()
                X_test_transformed[n_epoch, :, :] = rASR_pipeline.transform(X_test[[n_epoch], :, :])
                time_table[n_epoch] = timer() - start_in
            end = timer()
            print(f"test_{test_ind}: Pipeline transform in {end - start}s ({(end - start) / X_training.shape[0]}s/epoch)")
            title = f"s{k_file}_transform_computational_time"
            plot_time_dist(time_table, output_folder=Config.results_folder, title=title)

            mne_eeg_rasr_info = mne_eeg_test.info
            data = X_test_transformed.reshape(X_test_transformed.shape[0] * X_test_transformed.shape[1], -1).transpose()
            mne_eeg_rasr_python = mne.io.RawArray(data * 1e-6, mne_eeg_rasr_info)

            # comparison
            title = f"s{k_file}_filtered"
            plot_all_mne_data(mne_eeg_test, Config.results_folder, title)

            title = f"s{k_file}_RASR_matlab"
            plot_all_mne_data(mne_eeg_rasr_matlab, Config.results_folder, title)

            title = f"s{k_file}_RASR_matlab_diff"
            eeg_rasr_matlab_diff = mne_eeg_test[:, 0:len(mne_eeg_rasr_matlab)][0] - mne_eeg_rasr_matlab.get_data()
            mne_eeg_rasr_diff = mne.io.RawArray(data=eeg_rasr_matlab_diff, info=mne_eeg_rasr_matlab.info, verbose=False)
            plot_all_mne_data(mne_eeg_rasr_diff, Config.results_folder, title)

            title = f"s{k_file}_RASR_python"
            plot_all_mne_data(mne_eeg_rasr_python, Config.results_folder, title)

            title = f"s{k_file}_RASR_python_diff"
            eeg_rasr_diff = mne_eeg_test[:, 0:len(mne_eeg_rasr_python)][0] - mne_eeg_rasr_python.get_data()
            mne_eeg_rasr_diff = mne.io.RawArray(data=eeg_rasr_diff, info=mne_eeg_test.info, verbose=False)
            plot_all_mne_data(mne_eeg_rasr_diff, Config.results_folder, title)

            title = f"s{k_file}_RASR_matlab-python_diff"
            max_samples = min(len(mne_eeg_rasr_matlab), len(mne_eeg_rasr_python)) - 1
            eeg_rasr_matlab_python_diff = np.sqrt((mne_eeg_rasr_matlab[:, 0:max_samples][0] -
                                                   mne_eeg_rasr_python[:, 0:max_samples][0]) ** 2)
            mne_eeg_rasr_diff = mne.io.RawArray(data=eeg_rasr_matlab_python_diff, info=mne_eeg_rasr_matlab.info,
                                                verbose=False)
            plot_all_mne_data(mne_eeg_rasr_diff, Config.results_folder, title)

            # TODO: output more metrics for large scale analysis (e.g. parameters effect, etc.)
