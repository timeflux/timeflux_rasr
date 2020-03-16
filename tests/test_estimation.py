import pytest
from timeflux_rasr.estimation import RASR, _fit_eeg_distribution, _rms
from utils.utils import epoch
import numpy as np
import logging
from sklearn.pipeline import Pipeline
import numpy.testing as npt


def test_fit_eeg_distribution_values():
    X = np.arange(1, 1001) ** 2 / 10000
    mu, sig, alpha, beta = _fit_eeg_distribution(X)
    # Comparaison to matlab output
    npt.assert_almost_equal([mu, sig, alpha, beta], [6.4810, 2.6627, 4.4935, 3.5000], decimal=4)

def test_fit_eeg_distribution_invalid_params():
    with pytest.raises(ValueError, match="X needs to be a 1D ndarray."):
        _fit_eeg_distribution(np.random.randn(2,1))  # the user should squeeze the array

    with pytest.raises(ValueError, match='quantile_range needs to be a 2-elements vector.'):
        _fit_eeg_distribution(np.random.randn(100), quantile_range=[0.1])

    with pytest.raises(ValueError, match='Unreasonable quantile_range.'):
        _fit_eeg_distribution(np.random.randn(100), quantile_range=[0.1, 10])
    with pytest.raises(ValueError, match='Unreasonable quantile_range.'):
        _fit_eeg_distribution(np.random.randn(100), quantile_range=[-0.1, 0.9])

    with pytest.raises(ValueError, match='Unreasonable step sizes.'):
        _fit_eeg_distribution(np.random.randn(100), step_sizes=[0.2, 0.1])
    with pytest.raises(ValueError, match='Unreasonable step sizes.'):
        _fit_eeg_distribution(np.random.randn(100), step_sizes=[0.1, 0.00001])

    with pytest.raises(ValueError):
        _fit_eeg_distribution(np.random.randn(50))  # too small value
    _fit_eeg_distribution(np.random.randn(50), step_sizes=[0.1, 0.1])  # it should work

    with pytest.raises(ValueError, match='Unreasonable shape range.'):
        _fit_eeg_distribution(np.random.randn(100), beta_range=[0.2, 2])
    with pytest.raises(ValueError, match='Unreasonable shape range.'):
        _fit_eeg_distribution(np.random.randn(100), beta_range=[1, 10])

def test_rms():
    X = np.array([
        [[1, 2, 3],
         [1, 2, 3]],
        [[1, 3, 3],
         [2, 4, 10]]
    ])
    rms_values = _rms(X)
    npt.assert_almost_equal(rms_values, np.array([[1., 2., 3.], [1.58113883, 3.53553391, 7.38241153]]), decimal=4)

def test_tensordot():
    """toy to test tensordot along different dimensions (used in RASR)"""
    A = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    B = np.array([[1, 0], [0, 1], [1, 1]])
    expecteddot = np.array([[[4, 5], [10, 11]], [[16, 17], [22, 23]]])

    # right-hand side
    Ap = np.zeros((2, 2, 2))
    for k in range(A.shape[0]):
        Ap[k, :, :] = A[k, :, :].dot(B)
    npt.assert_array_almost_equal(Ap, expecteddot)

    Ap2 = np.tensordot(A, B, axes=(2, 0))
    npt.assert_array_almost_equal(Ap2, expecteddot)

    # left-hand side
    Ap3 = np.zeros((2, 2, 2))
    for k in range(A.shape[0]):
        Ap3[k, :, :] = B.T.dot(np.transpose(A, axes=[0, 2, 1])[k, :, :]).T
    npt.assert_array_almost_equal(Ap3, expecteddot)

    Ap4 = np.tensordot(B, np.transpose(A, axes=[0, 2, 1]), axes=(0, 1))
    B.T.dot(np.transpose(A, axes=[0, 2, 1]))


def test_rasr_rand_fit_transform():
    """test initialization, fit and transform of RASR"""
    np.random.seed(seed=42)
    X = np.random.randn(100, 32, 8)
    pipeline = RASR()
    pipeline.fit_transform(X)

def test_rasr_nan_fit_transform():
    """test initialization, fit and transform of RASR"""
    np.random.seed(seed=42)
    X = np.random.randn(100, 32, 8)
    X2 = X.copy()
    X2[10, 10, 0] = np.nan
    pipeline = RASR()
    with pytest.raises(ValueError, match=f"Input contains NaN, infinity or a value too large for dtype\(\'float64\'\)."):
        pipeline.fit(X2)
    with pytest.raises(ValueError, match=f"Input contains NaN, infinity or a value too large for dtype\(\'float64\'\)."):
        pipeline.fit(X)
        pipeline.transform(X2)

def test_rasr_rand_fit_transform_training_test():
    """test initialization, fit and transform of RASR"""
    np.random.seed(seed=42)
    X = np.random.randn(150, 250, 8)
    Xtrain = X[0:100, :, :]
    Xtest  = X[100:, :, :]

    # fit test
    logging.info("Test RASR: random pipeline...")

    pipeline = Pipeline([
        ("RASR", RASR())
    ])

    pipeline.fit(Xtrain)
    logging.info("Test RASR: fitted random pipeline")
    Xclean = pipeline.transform(Xtest)
    logging.info("Test RASR: transformed random pipeline")

def test_rasr_invalid_params():
    """test that exception is raised for invalid params"""
    with pytest.raises(ValueError, match="Training requires at least 100 of trials to fit."):
        np.random.seed(seed=42)
        X = np.random.randn(10, 100, 8)
        pipeline = RASR()
        pipeline.fit_transform(X)

    with pytest.raises(ValueError, match="X.shape should be \(n_trials, n_samples, n_electrodes\)."):
        np.random.seed(seed=42)
        X = np.random.randn(100, 8)
        pipeline = RASR()
        pipeline.fit_transform(X)

    with pytest.raises(ValueError, match="X.shape should be \(n_trials, n_samples, n_electrodes\)."):
        np.random.seed(seed=42)
        X = np.random.randn(100, 100, 8)
        pipeline = RASR()
        pipeline.fit(X)
        pipeline.transform(X[-1,: , :])

def test_rasr_unknown_params():
    dict_of_params = dict(rejection_cutoff=4.0, max_dimension=0.33, unknown_param=10)

    invalid_params = dict(unknown_param=10)
    with pytest.raises(ValueError, match=f"got an unexpected keyword arguments \'{invalid_params}\'"):
        # params are passed to _fit_eeg_distribution()
        np.random.seed(seed=42)
        X = np.random.randn(100, 100, 8)
        pipeline = RASR(**dict_of_params)
        pipeline.fit(X)

def test_rasr_fit_to_randn_distribution_cutoff3():
    """Check fit against approximate estimation of the mixing and threshold matrices for given cutoff"""
    np.random.seed(seed=10)
    srate = 250
    cutoff = 3
    window_len = int(0.5 * srate)
    window_interval = int(window_len * 0.34)
    X = np.diag(np.arange(1, 5)).dot(np.random.randn(4, srate * 60))
    epochs = np.swapaxes(epoch(X, window_len, window_interval), 2, 1)
    dict_of_params = dict(rejection_cutoff=cutoff, max_dimension=0.33)
    pipeline = RASR(**dict_of_params)
    pipeline.fit(epochs)
    target_mixing = np.diag(np.arange(1, 5))
    target_threshold = target_mixing * (1 + cutoff * 0.05)  # based on gaussian
    npt.assert_allclose(np.abs(pipeline.mixing_), target_mixing, rtol=0.05, atol=0.01)
    npt.assert_allclose(np.abs(pipeline.threshold_), target_threshold, rtol=0.05, atol=0.05)

def test_rasr_fit_to_randn_distribution_cutoff5():
    """Check fit against approximate estimation of the mixing and threshold matrices for given cutoff"""
    np.random.seed(seed=10)
    srate = 250
    cutoff = 5
    window_len = int(0.5 * srate)
    window_interval = int(window_len * 0.34)
    X = np.diag(np.arange(1, 5)).dot(np.random.randn(4, srate * 60))
    epochs = np.swapaxes(epoch(X, window_len, window_interval), 2, 1)
    dict_of_params = dict(rejection_cutoff=cutoff, max_dimension=0.33)
    pipeline = RASR(**dict_of_params)
    pipeline.fit(epochs)
    target_mixing = np.diag(np.arange(1, 5))
    target_threshold = target_mixing * (1 + cutoff * 0.05)  # based on gaussian
    npt.assert_allclose(np.abs(pipeline.mixing_), target_mixing, rtol=0.05, atol=0.01)
    npt.assert_allclose(np.abs(pipeline.threshold_), target_threshold, rtol=0.05, atol=0.05)

def test_rasr_fit_to_randn_distribution_cutoff20():
    """Check fit against approximate estimation of the mixing and threshold matrices for given cutoff"""
    np.random.seed(seed=10)
    srate = 250
    cutoff = 20
    window_len = int(0.5 * srate)
    window_interval = int(window_len * 0.34)
    X = np.diag(np.arange(1, 5)).dot(np.random.randn(4, srate * 60))
    epochs = np.swapaxes(epoch(X, window_len, window_interval), 2, 1)
    dict_of_params = dict(rejection_cutoff=cutoff, max_dimension=0.33)
    pipeline = RASR(**dict_of_params)
    pipeline.fit(epochs)
    target_mixing = np.diag(np.arange(1, 5))
    target_threshold = target_mixing * (1 + cutoff * 0.05)  # based on gaussian
    npt.assert_allclose(np.abs(pipeline.mixing_), target_mixing, rtol=0.05, atol=0.01)
    npt.assert_allclose(np.abs(pipeline.threshold_), target_threshold, rtol=0.1, atol=0.1)

# TODO: test_rasr_error_1              # test wrong size input
# TODO: test_rasr_error_3              # test when singular matrix as input
# TODO: test_rms                       # test output from a given matrix
# TODO: test_rasr_output1              # test with fixed seed and parameters output (see below)
# npt.assert_almost_equal(Xclean[0, 0], saved_array1)    # test first sample to given output
# npt.assert_almost_equal(Xclean[49, 0], saved_array2)   # test first sample to given output
# TODO: test_rasr_singular             # test using duplicate column for singular matrix
