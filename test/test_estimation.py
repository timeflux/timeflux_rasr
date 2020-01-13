import pytest
from timeflux_rasr.estimation import RASR
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from timeflux_rasr.estimation import _fit_eeg_distribution

tolerance = 0.05   # 5% tolance

def test_fit_eeg_distribution():
    X = np.arange(1, 1001) ** 2 / 10000

    mu, sig, alpha, beta = _fit_eeg_distribution(X)

    # Comparaison to matlab output
    assert abs(mu - 6.4810) < (6.4810 * tolerance)
    assert abs(sig - 2.6627) < (2.6627 * tolerance)
    assert abs(alpha - 4.4935) < (4.4935 * tolerance)
    assert abs(beta - 3.5000) < (3.5000 * tolerance)

def test_rasr_rand_fit_transform():
    """test initialization, fit and transform of RASR"""

    X = np.random.randn(100, 250, 8)
    Xtrain = X[0:49, :, :]
    Xtest  = X[50:-1, :, :]
    shapetest = Xtest.shape

    # fit test
    logging.info("Test RASR: random pipeline...")

    pipeline = Pipeline([
        ("RASR", RASR())
    ])

    pipeline.fit(Xtrain)
    logging.info("Test RASR: fitted random pipeline")
    Xclean = pipeline.transform(Xtest)
    logging.info("Test RASR: transformed random pipeline")

    assert Xclean.shape == shapetest

# TODO: test_rasr_error_1              # test wrong size input
# TODO: test_rasr_error_2              # test wrong parameters (blocksize too low, window_len too low, etc.)
# TODO; test_rasr_error_3              # test when singular matrix as input
# TODO: test_eeg_fit_distribution      # test output from given vector
# TODO: test_rms                       # test output from a given matrix

