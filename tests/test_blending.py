import pytest
from timeflux_blending.blending import Blending, _merge_overlap
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import numpy.testing as npt

def test_blending_not_init():
    with pytest.raises(ValueError, match="window_overlap parameter is not initialized."):
        Blending(window_overlap=None)

def test_blending_fit():
    """Check if returns correct blending on a single mat"""
    X = np.ones((1, 10, 2))
    Blend = Blending(window_overlap=10)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(0, 10)/9)))/2)[:, None], 2, axis=1)
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform[0, :], expectedmat)

def test_blending_wrong_transform():
    """Check if returns error when number of spatial dimension changed"""
    X = np.ones((1, 10, 2))
    Blend = Blending(window_overlap=10)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(0, 10)/9)))/2)[:,None], 2, axis=1)
    Blend.fit(X)
    X = np.ones((1, 10, 3))
    with pytest.raises(ValueError, match='Shape of input is different from what was seen in `fit`'):
        Xtransform = Blend.transform(X)

def test_blending_fit_wrong_dim():
    """Check if returns error when number of spatial dimension changed"""
    X = np.ones((1, 10))
    Blend = Blending(window_overlap=10)
    with pytest.raises(ValueError, match='X.shape should be \(n_trials, n_samples, n_electrodes\).'):
        Blend.fit(X)

def test_blending_transform_wrong_dim():
    """Check if returns error when number of spatial dimension changed"""
    X = np.ones((1, 10, 2))
    Blend = Blending(window_overlap=10)
    Blend.fit(X)
    with pytest.raises(ValueError, match='X.shape should be \(n_trials, n_samples, n_electrodes\).'):
        Blend.transform(np.ones((10, 2)))

def test_blending_fit_repeat():
    """Check if last values are saved when successive calls of Blending.
    """
    X = np.ones((1, 10, 2))
    Blend = Blending(window_overlap=10)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(0, 10)/9)))/2)[:, None], 2, axis=1)
    Xtransform = Blend.fit_transform(X)
    X = np.zeros((1, 10, 2))
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform[0, :], (1 - expectedmat) * expectedmat )

def test_blending_fit_multi():
    """Check if handle properly multiple windows
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    Blend = Blending(window_overlap=10)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(0, 10)/9)))/2)[:, None], 2, axis=1)
    expectedmat = np.concatenate([expectedmat[None, :], (1 - expectedmat[None, :]) * expectedmat[None, :]],)
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform, expectedmat)

def test_blending_fit_multi_no_overlap():
    """Check if handle properly multiple windows with no overlap
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    Blend = Blending(window_overlap=0)
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform, X)

def test_merge_zero():
    """Merge multiple windows with no overlap
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    Xtransform = _merge_overlap(X, window_overlap=0)
    npt.assert_array_almost_equal(Xtransform, np.concatenate(X))

def test_merge_simple():
    """Merge multiple windows with overlap
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    Xtransform = _merge_overlap(X, window_overlap=10)
    npt.assert_array_almost_equal(Xtransform, X[-1, :, :])

def test_merge_simple2():
    """Merge multiple windows with overlap
    """
    X = np.reshape(np.arange(1., 21.), (5, 2, 2))
    Xtransform = _merge_overlap(X, window_overlap=1)
    npt.assert_array_almost_equal(Xtransform, np.concatenate([X[0:-1, 0, :], X[-1, :, :]]))

def test_merge_too_big_overlap():
    """Try to merge multiple windows with too big overlap
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    with pytest.raises(ValueError, match="window_overlap cannot be higher than n_samples."):
        Xtransform = _merge_overlap(X, window_overlap=11)

def test_merge_negativ_overlap():
    """Try to merge multiple windows with negativ overlap
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    with pytest.raises(ValueError, match="window_overlap should be non-negative."):
        Xtransform = _merge_overlap(X, window_overlap=-1)

def test_merge_wrong_dim():
    X = np.concatenate([np.ones((10, 2)), np.zeros((10, 2))])
    with pytest.raises(ValueError, match='X.shape should be \(n_trials, n_samples, n_electrodes\).'):
        Xtransform = _merge_overlap(X, window_overlap=1)

