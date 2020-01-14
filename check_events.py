from utils.config import Config
from collections import OrderedDict
from glob import glob 
import os
import mne
import pandas as pd 
import numpy as np 
from pyxdf import load_xdf

import seaborn as sns 
sns.set(font_scale=1)
import matplotlib.pyplot as plt 

from utils.utils import get_stream_names, extract_signal_stream
from utils.utils import float_index_to_time_index, time_index_to_float_index
from utils.utils import estimate_rate
from utils.utils import pandas_to_mne

print("Config LOADED")

k_file = 0

raw_xdf_fname = Config.raw_files[k_file]
streams, _ = load_xdf(raw_xdf_fname)
stream_names = get_stream_names(streams)
df_eeg_raw = extract_signal_stream(streams , 'Android_EEG_010026')
df_presentation_events = extract_signal_stream(streams , 'Presentation_Markers')

print("RAW EEG LOADED")

eeg_columns = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'FC1', 'FC2', 'Cz', 'C3', 'C4', 'T7',
       'T8', 'CPz', 'CP1', 'CP2', 'CP5', 'CP6', 'TP9', 'TP10', 'Pz', 'P3',
       'P4', 'O1', 'O2']
df_eeg_raw = float_index_to_time_index(df_eeg_raw)

duration = (df_eeg_raw.index[-1] - df_eeg_raw.index[0]).total_seconds()/60
rate = estimate_rate(df_eeg_raw)
print('Duration of session was {duration} min. \nRate is of {rate} Hz.')

bad_ch = []
raw, event_id, picks = pandas_to_mne(data=df_eeg_raw, rate=rate, bad_ch=bad_ch, 
                                     montage_kind='standard_1005', unit_factor=1e-6, 
                                     events=df_presentation_events)
events = mne.find_events(raw)
event_id = {'select block condition': 1,  # 0 
            'instructions': 2,            # 1
            '11': 3,                      # entree de l experimentateur, num de block
            'question': 4,                # 2 + code de la question 
            'subject response': 5,        # 3 + mot en allemand
            'enough walking': 6,          # 4 OK, task could start anytime
            '44': 7,                      # entree de l'experimentateur, num de liste de mots
            'end of experiment': 8,       # 5 block + num, list + num (liste de mots), condition (outside/inside)
            'start task': 9,              # 55 ?
            'start walking/rest eeg': 10, # 6 walking / rest eeg
            '66': 11}                     # entree de l'experimentateur, outside

raw.plot(events=events, event_id=event_id, color={'eeg':'darkblue'},
         lowpass=45, highpass=0.5, start=125, n_channels=24)

# ep = mne.Epochs(raw, events=events, tmin=0.0, tmax=10.0, event_id={'start task': 9},
#                 baseline=None, preload=True)
# ep.copy().pick_channels(['Fp1', 'Fp2']).filter(l_freq=0.5, h_freq=45.).plot()
# ep.copy().filter(l_freq=1.0, h_freq=None).plot()
