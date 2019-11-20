"""Copyright: Raph B-L < raphaelle@timeflux.io >, 2019"""

from warnings import warn

import mne
import numpy as np
import pandas as pd
from numpy.lib import stride_tricks


def indices(list_, filtr=lambda x: bool(x)):
    # return indices of the element that met the condition defined in filtr
    return [i for i, x in enumerate(list_) if filtr(x)]


def get_channel_names(stream):
    """
        extract channel name from xdf stream
    :param stream: dictionnary
        xdf stream to parse
    :return: list
        list of channels names
    """

    try:
        return [channel_info['label'][0] for channel_info in
                stream['info']['desc'][0]['channels'][0]['channel']]
    except TypeError:
        print("Warning : Channel description is empty")
        return None
    except IndexError:
        print("Warning : No channels names found")
        return None


def get_stream_names(streams, type=None):
    """
        extract stream name from xdf stream
    :param streams: list of dictionnaries
        xdf streams to parse
    :param type: string
        type of stream, eg. 'Signal' or 'Markers'
    :return: list
        list of stream name contained in streams
    """
    if type is None:
        return [stream['info']['name'][0] for stream in streams]
    else:
        return [stream['info']['name'][0] for stream in streams if stream['info']['type'][0] == type]


def extract_signal_stream(streams, name='nexus_signal_raw', channels='all', n=0):
    """
        extract signal from given stream
    :param streams: list
        streams to be extracted
    :param name: string
        stream name as specified in xdf
    :param channels: string
        if 'all' extract all channels from stream
    :param n: int
        index of stream corresponding to specified stream name
    :return: dataframe
    """
    stream_index = indices(streams, lambda d: d['info']['name'][0] == name)
    if stream_index:
        stream_index = stream_index[n]
        stream = streams[stream_index]

        stream_times = stream['time_stamps']
        stream_values = stream['time_series']
        stream_channels = get_channel_names(stream)

        if (len(stream_times) > 0) & (len(stream_values) > 0):
            if channels == 'all':
                return pd.DataFrame(index=stream_times, data=stream_values, columns=stream_channels)
            else:
                if stream_values is not None:
                    return pd.DataFrame(index=stream_times,
                                        data=stream_values[:,
                                             [indices(stream_channels, filtr=lambda ch: (ch == channel_name))[0] for
                                              channel_name in channels]], columns=channels)
        else:
            warn("Stream {0} is empty".format(name))
            return pd.DataFrame()
    else:
        warn(name + 'is not in streams.')
        return pd.DataFrame()


def estimate_rate(data):
    """ Estimate nominal sampling rate of a DataFrame.
    This function checks if the index are correct, that is monotonic and regular
    (the jitter should not exceed twice the nominal timespan)
    Notes
    -----
    This function does not take care of jitters in the Index and consider that the rate as the 1/Ts
    where Ts is the average timespan between samples.
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame with index corresponding to timestamp (either DatetimeIndex or floats)
    Returns
    -------
    rate: nominal rate of the DataFrame
    """
    # check that the index is monotonic
    if not data.index.is_monotonic:
        raise Exception('Data index should be monotonic')
    if data.shape[0] < 2:
        raise Exception('Sampling rate requires at least 2 points')

    if isinstance(data.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
        delta = data.index - data.index[0]
        index_diff = np.diff(delta) / np.timedelta64(1, 's')
    elif np.issubdtype(data.index, np.number):
        index_diff = np.diff(data.index)
    else:
        raise Exception('Dataframe index is not numeric')

    average_timespan = np.median(index_diff)
    if np.any(index_diff >= average_timespan * 2):
        raise Exception('Effective sampling is greater than twice the nominal rate')

    return 1 / average_timespan


def pandas_to_mne(data, rate, events=None, montage_kind='standard_1005', unit_factor=1e-6, bad_ch=[]):
    ''' Convert a pandas Dataframe into mne raw object

    Parameters
    ----------
    data : Dataframe with index=timestamps, columns=eeg channels
    rate : Sampling rate
    events : array, shape = (n_events, 3) with labels on the third axis.
    unit_factor : unit factor to apply to get Voltage
    bad_ch : list of channels to reject
    montage_kind : str (default: 'standard_1005')
        EEG montage name

    Returns
    -------
    raw: raw object
    '''
    n_chan = len(data.columns)

    X = data.copy().values
    times = data.index

    ch_names = list(data.columns)
    ch_types = ['eeg'] * n_chan
    montage = mne.channels.read_montage(montage_kind) if montage_kind is not None else None
    # sfreq = estimate_rate(data)
    X *= unit_factor

    if events is not None:
        events_onsets = events.index
        events_labels = events.label.values
        event_id = {mk: (ii + 1) for ii, mk in enumerate(np.unique(events_labels))}
        ch_names += ['stim']
        ch_types += ['stim']

        trig = np.zeros((len(X), 1))
        for ii, m in enumerate(events_onsets):
            ix_tr = np.argmin(np.abs(times - m))
            trig[ix_tr] = event_id[events_labels[ii]]

        X = np.c_[X, trig]
    else:
        event_id = None

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=rate, montage=montage)
    info["bads"] = bad_ch
    raw = mne.io.RawArray(data=X.T, info=info, verbose=False)
    picks = mne.pick_channels(raw.ch_names, include=[], exclude=["stim"] + bad_ch)
    return raw, event_id, picks


from datetime import datetime


def time_index_to_float_index(df, inplace=False):
    """Convert a dataframe float indices to `datetime64['us']` indices."""
    if not inplace:
        df = df.copy()
    df.index = df.index.map(lambda d: d.timestamp())
    return df


def float_index_to_time_index(df, inplace=False):
    """Convert a dataframe float indices to `datetime64['us']` indices."""
    if not inplace:
        df = df.copy()
    df.index = df.index.map(datetime.utcfromtimestamp)
    df.index = pd.to_datetime(df.index, unit='us', utc=False)
    return df




def epoch(a, size, interval, axis=-1):
    """ Small proof of concept of an epoching function using NumPy strides
    License: BSD-3-Clause
    Copyright: David Ojeda <david.ojeda@gmail.com>, 2018

    Create a view of `a` as (possibly overlapping) epochs.
    The intended use-case for this function is to epoch an array representing
    a multi-channels signal with shape `(n_samples, n_channels)` in order
    to create several smaller views as arrays of size `(size, n_channels)`,
    without copying the input array.
    This function uses a new stride definition in order to produce a view of
    `a` that has shape `(num_epochs, ..., size, ...)`. Dimensions other than
    the one represented by `axis` do not change.
    Parameters
    ----------
    a: array_like
        Input array
    size: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    axis: int
        Axis of the samples on `a`. For example, if `a` has a shape of
        `(num_observation, num_samples, num_channels)`, then use `axis=1`.
    Returns
    -------
    ndarray
        Epoched view of `a`. Epochs are in the first dimension.
    """
    a = np.asarray(a)
    n_samples = a.shape[axis]
    n_epochs = (n_samples - size) // interval + 1

    new_shape = list(a.shape)
    new_shape[axis] = size
    new_shape = (n_epochs,) + tuple(new_shape)

    new_strides = (a.strides[axis] * interval,) + a.strides

    return stride_tricks.as_strided(a, new_shape, new_strides)
