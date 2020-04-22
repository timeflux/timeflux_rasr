import pytest
import numpy as np
import numpy.testing as npt
from utils.utils import epoch, geometric_median, check_params
import inspect
from timeflux_rasr.estimation import RASR, _fit_eeg_distribution

# TODO: pytest for timeflux specific utils (not used in the RASR module)
""" list
- [ ] indices(list_, filtr=lambda x: bool(x))
- [ ] get_channel_names(stream)
- [ ] get_stream_names(streams, type=None)
- [ ] extract_signal_stream(streams, name='nexus_signal_raw', channels='all', n=0)
- [ ] estimate_rate(data)
- [ ] pandas_to_mne(data, rate, events=None, montage_kind='standard_1005', unit_factor=1e-6, bad_ch=[])
- [ ] time_index_to_float_index(df, inplace=False)
- [ ] float_index_to_time_index(df, inplace=False)
"""


def compute_nb_epochs(N, T, I):
    """Return the exact number of expected windows based on the samples (N), window_length (T) and interval (I)"""
    return int(np.ceil((N-T+1) / I))

def test_epoch_unit1():
    np.random.seed(seed=42)
    N = 1000               # signal length
    T = 100                # window length
    I = 101                # interval
    Ne = 8                 # electrodes
    window_length = 1      # in seconds
    window_overlap = 0     # in seconds
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X, T, I)
    assert epochs.shape == (K, Ne, T)

def test_epoch_unit2():
    np.random.seed(seed=42)
    N = 100               # signal length
    T = 100                # window length
    I = 101                # interval
    Ne = 8                 # electrodes
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X, T, I)
    assert epochs.shape == (K, Ne, T)

def test_epoch_unit_with_axis():
    np.random.seed(seed=42)
    N = 1000               # signal length
    T = 100                # window length
    I = 10                 # interval
    Ne = 8                 # electrodes
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X.T, T, I, axis=0)
    assert epochs.shape == (K, T, Ne)

def test_epoch_unit_with_axis():
    epochs_target = np.array([[[1,  2,  3,  4]],
                              [[4,  5,  6,  7]],
                              [[7,  8,  9, 10]]])
    X = np.expand_dims(np.arange(1, 11), axis=0)
    T = 4                 # window length
    I = 3                  # interval
    epochs = epoch(X, T, I, axis=1)
    npt.assert_array_equal(epochs, epochs_target)

def test_epoch_fail_size():
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        np.random.seed(seed=42)
        N = 1000  # signal length
        T = 100  # window length
        I = 0  # interval
        Ne = 8  # electrodes
        X = np.random.randn(Ne, N)
        epochs = epoch(X.T, T, I, axis=0)
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        np.random.seed(seed=42)
        N = 1000  # signal length
        T = 0  # window length
        I = 1  # interval
        Ne = 8  # electrodes
        X = np.random.randn(Ne, N)
        epochs = epoch(X.T, T, I, axis=0)

def test_geometric_median_matlab1():
    """compure with exiting value got with matlab"""
    X = np.expand_dims(np.arange(1, 1001) ** 2 / 10000, axis=1)
    npt.assert_array_almost_equal(geometric_median(X), 25.0501, decimal=1)

def test_geometric_median_matlab2():
    """compure with exiting value got with matlab"""
    X = np.vstack([np.arange(1, 1001) ** 2 / 10000, np.arange(1,1001)])
    npt.assert_array_almost_equal(geometric_median(X.T), [27.0382, 500.3017], decimal=3)

def test_geometric_median_zero_distance():
    """compure with exiting value got with matlab"""
    X = np.array([[1, 1, 1], [2, 2, 2]])
    npt.assert_array_almost_equal(geometric_median(X.T), [1., 2.], decimal=3)

def test_geometric_median_semi_zero_distance():
    """compure with exiting value got with matlab"""
    X = np.array([[1, 1, 1], [1, 2, 3]])
    npt.assert_array_almost_equal(geometric_median(X.T), [1., 2.], decimal=3)

def test_geometric_median_non_converge(capsys):
    """compure with exiting value got with matlab"""
    np.random.seed(seed=42)
    X = np.random.rand(10000, 10)
    msg_cap = "Geometric median could converge in 3 iteration with eps=0.0000000001 \n"
    geometric_median(X.T, eps=1e-10, max_it=3)
    captured = capsys.readouterr()
    assert captured.out == msg_cap

def test_check_params_fun1():
    kwargs = dict(rejection_cutoff=4.0, max_dimension=0.33, unknown_param=10, beta_range=100)
    valid_kwargs = dict(beta_range=100)
    assert valid_kwargs == check_params(_fit_eeg_distribution, **kwargs)

def test_check_params_none():
    kwargs = dict(rejection_cutoff=4.0, max_dimension=0.33, unknown_param=10)
    valid_kwargs = dict()
    assert valid_kwargs == check_params(_fit_eeg_distribution, **kwargs)

def test_check_params_invalid():
    kwargs = dict(rejection_cutoff=4.0, max_dimension=0.33, unknown_param=10)
    valid_kwargs = dict()
    assert (valid_kwargs, kwargs) == check_params(_fit_eeg_distribution, return_invalids=True, **kwargs)

def test_check_params_class():
    kwargs = dict(rejection_cutoff=4.0, max_dimension=0.33, unknown_param=10, beta_range=100)
    valid_kwargs = dict(rejection_cutoff=4.0, max_dimension=0.33)
    assert valid_kwargs == check_params(RASR, **kwargs)