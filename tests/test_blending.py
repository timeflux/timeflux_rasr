import pytest
from timeflux_blending.blending import Blending
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import numpy.testing as npt

def test_blending_not_init():
    with pytest.raises(ValueError, match="window_overlap parameter is not initialized."):
        Blending(window_overlap=None)

def test_blending_fit():
    X = np.ones((1, 10, 2))
    Blend = Blending(window_overlap=10)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(0, 10)/9)))/2)[:,None], 2, axis=1)
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform[0,:], expectedmat)

