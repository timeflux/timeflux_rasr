""" Convert XDF into Timeflux-readable/replayable HDF5
"""

import logging

import pandas as pd
from pyxdf import load_xdf
from tqdm import tqdm

from utils.config import Config
from utils.utils import get_stream_names, extract_signal_stream, float_index_to_time_index

logger = logging.getLogger()

if __name__ == '__main__':
    # Load xdf and extract eeg stream
    logging.info("Converting XDF to HDF5")
    for fname in tqdm(Config.raw_files):
        # Load xdf
        streams, _ = load_xdf(fname, dejitter_timestamps=True)
        # extract raw eeg and presentation marker streams
        stream_names = get_stream_names(streams)
        df_eeg_raw = extract_signal_stream(streams, 'Android_EEG_010026')
        df_presentation_events = extract_signal_stream(streams, 'Presentation_Markers')
        # standardize marker stream into timeflux events
        df_presentation_events = float_index_to_time_index(df_presentation_events)
        df_presentation_events.columns = ['label']
        # extract calibration events, between markers  ‘6 rest eeg’ and the immediately following one ‘6 walking’
        df_calibration_events = pd.DataFrame(index=df_presentation_events.loc[
                                                   df_presentation_events[
                                                       df_presentation_events.label == '6 rest eeg'].index[0]:].iloc[
                                                   :2].index, data=['calibration_begins', 'calibration_ends'],
                                             columns=['label'])
        # extract only eeg columns and convert kndex to datetime
        eeg_columns = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'FC1', 'FC2', 'Cz', 'C3', 'C4', 'T7',
                       'T8', 'CPz', 'CP1', 'CP2', 'CP5', 'CP6', 'Tp9', 'Tp10', 'Pz', 'P3',
                       'P4', 'O1', 'O2']
        df_eeg_raw = df_eeg_raw.loc[:, eeg_columns]
        df_eeg_raw = float_index_to_time_index(df_eeg_raw)

        # save in hdf5
        hdf_fname = fname.replace('raw', 'replay').replace('xdf', 'hdf5')
        with pd.HDFStore(hdf_fname, 'w') as store:
            df_eeg_raw.to_hdf(store, '/eeg')
            df_calibration_events.to_hdf(store, '/events', format='table')
