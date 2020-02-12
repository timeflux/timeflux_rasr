import pytest
from timeflux_rasr.estimation import RASR
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from timeflux_rasr.estimation import _fit_eeg_distribution
import numpy.testing as npt
from sklearn.utils.estimator_checks import check_estimator

def test_fit_eeg_distribution():
    X = np.arange(1, 1001) ** 2 / 10000

    mu, sig, alpha, beta = _fit_eeg_distribution(X)

    # Comparaison to matlab output
    npt.assert_almost_equal([mu, sig, alpha, beta], [6.4810, 2.6627, 4.4935, 3.5000], decimal=4)

# def test_rasr_parameters():
#     pipeline = RASR(srate = 2, step_sizes=[0.1, 0.1])
#     check_estimator(pipeline)
#
# def test_rasr_sklearn():
#     check_estimator(RASR(srate=2, window_overlap=0.99, step_sizes=[0.1, 0.1]))

def test_rasr_rand_fit_transform():
    """test initialization, fit and transform of RASR"""
    np.random.seed(seed=42)
    srate = 250
    X = np.random.randn(100, 1 * srate, 8)
    Xtrain = X[0:50, :, :]
    Xtest  = X[50:, :, :]
    shapetest = Xtest.shape

    # fit test
    logging.info("Test RASR: random pipeline...")

    pipeline = Pipeline([
        ("RASR", RASR(srate=250))
    ])

    pipeline.fit(Xtrain)
    logging.info("Test RASR: fitted random pipeline")
    Xclean = pipeline.transform(Xtest)
    logging.info("Test RASR: transformed random pipeline")

    assert Xclean.shape == shapetest

# TODO: test_rasr_error_1              # test wrong size input
# TODO: test_rasr_error_2              # test wrong parameters (blocksize too low, window_len too low, etc.)
# TODO: test_rasr_error_3              # test when singular matrix as input
# TODO: test_rms                       # test output from a given matrix
# TODO: test_rasr_output1              # test with fixed seed and parameters output (see below)
# npt.assert_almost_equal(Xclean[0, 0], saved_array1)    # test first sample to given output
# npt.assert_almost_equal(Xclean[49, 0], saved_array2)   # test first sample to given output
# TODO: test_rasr_nan                  # test with NaN in array as input
# TODO: test_rasr_singular             # test using duplicate column for singular matrix
