import pytest
from timeflux_blending.blending import Blending, _merge_overlap
import numpy as np
import numpy.testing as npt

def test_blending_not_init():
    with pytest.raises(ValueError, match="window_overlap parameter is not initialized."):
        Blending(window_overlap=None)

def test_blending_params_type():
    with pytest.raises(TypeError, match='window_overlap must int'):
        Blend = Blending(window_overlap=10.)
    with pytest.raises(TypeError, match='merge must bool'):
        Blend = Blending(window_overlap=10, merge=-10)
    with pytest.raises(TypeError, match='windowing must be None, or bool'):
        Blend = Blending(window_overlap=10, merge=True, windowing='Nope')

def test_blending_fit():
    """Check if returns correct blending on a single mat"""
    X = np.ones((1, 10, 2))
    Blend = Blending(window_overlap=10, windowing=True)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(1, 11)/11)))/2)[:, None], 2, axis=1)
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform[0, :], expectedmat)

def test_blending_wrong_transform():
    """Check if returns error when number of spatial dimension changed"""
    X = np.ones((1, 10, 2))
    Blend = Blending(window_overlap=10)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(1, 11)/11)))/2)[:,None], 2, axis=1)
    Blend.fit(X)
    X = np.ones((1, 10, 3))
    with pytest.raises(ValueError, match='Shape of input is different from what was seen in `fit`'):
        Xtransform = Blend.transform(X)

def test_blending_NaN():
    """Check if returns error when NaN"""
    X = np.ones((1, 10, 2))
    X[0, 5, 1] = np.NaN
    Blend = Blending(window_overlap=10)
    with pytest.raises(ValueError, match='Input contains NaN, infinity or a value too large for dtype\(\'float64\'\).'):
        Blend.fit(X)
    X2 = np.ones((1, 10, 3))
    Blend.fit(X2)
    with pytest.raises(ValueError, match='Input contains NaN, infinity or a value too large for dtype\(\'float64\'\).'):
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
    Blend = Blending(window_overlap=10, windowing=True)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(1, 11)/11)))/2)[:, None], 2, axis=1)
    Xtransform = Blend.fit_transform(X)
    X = np.zeros((1, 10, 2))
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform[0, :], (1 - expectedmat) * expectedmat )

def test_blending_fit_multi():
    """Check if handle properly multiple windows
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    Blend = Blending(window_overlap=10, windowing=True)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(1, 11)/11)))/2)[:, None], 2, axis=1)
    expectedmat = np.concatenate([expectedmat[None, :], (1 - expectedmat[None, :]) * expectedmat[None, :]],)
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform, expectedmat)

def test_blending_fit_multi_no_overlap():
    """Check if handle properly multiple windows with no overlap
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    Blend = Blending(window_overlap=0, windowing=True)
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

def test_blending_with_merge():
    """Check if handle properly multiple windows for blending with merge
    """
    X = np.concatenate([np.ones((1, 10, 2)), np.zeros((1, 10, 2))])
    Blend = Blending(window_overlap=10, merge=True, windowing=True)
    expectedmat = np.repeat(np.array((1 - np.cos(np.pi * (np.arange(1, 11)/11)))/2)[:, None], 2, axis=1)
    expectedmat = np.concatenate([expectedmat[None, :], (1 - expectedmat[None, :]) * expectedmat[None, :]],)
    Xtransform = Blend.fit_transform(X)
    npt.assert_array_almost_equal(Xtransform, expectedmat[-1, :, :])

def test_blending_with_merge2():
    """Check if handle properly multiple windows for blending with with merge
    """
    X = np.reshape(np.arange(1., 21.), (5, 2, 2))
    Blend = Blending(window_overlap=1, merge=True, windowing=True)
    Xtransform = Blend.fit_transform(X)
    expectedmat = np.array([[0.5, 1.],
                           [4., 5.],
                           [8., 9.],
                           [12., 13.],
                           [16., 17.],
                           [19., 20.]])
    npt.assert_array_almost_equal(Xtransform, expectedmat)

def test_blending_with_merge3():
    """Check if handle properly multiple windows for blending with with merge
    """
    X = np.reshape(np.arange(1., 21.), (5, 2, 2))
    Blend = Blending(window_overlap=1, merge=True, windowing=False)
    Xtransform = Blend.fit_transform(X)
    expectedmat = np.array([[ 2.,  3.],
                           [ 4.,  5.],
                           [ 8.,  9.],
                           [12., 13.],
                           [16., 17.],
                           [19., 20.]])
    npt.assert_array_almost_equal(Xtransform, expectedmat)

def test_blending_with_merge4():
    """Check if handle properly multiple windows for blending with with merge with first window interpolation to zero
    """
    X = np.reshape(np.arange(1., 31.), (5, 3, 2))
    Blend = Blending(window_overlap=2, merge=True, windowing=True)
    Xtransform = Blend.fit_transform(X)
    expectedmat = np.array([[ 0.25  ,  0.5   ],
                           [ 3.4375,  4.25  ],
                           [ 9.25  , 10.25  ],
                           [15.25  , 16.25  ],
                           [21.25  , 22.25  ],
                           [26.    , 27.    ],
                           [29.    , 30.    ]])
    npt.assert_array_almost_equal(Xtransform, expectedmat)

def test_blending_with_merge5():
    """Check if handle properly multiple windows for blending with with merge without first interpolation
    """
    X = np.reshape(np.arange(1., 31.), (5, 3, 2))
    Blend = Blending(window_overlap=2, merge=True)
    Xtransform = Blend.fit_transform(X)
    expectedmat = np.array([[ 2.5  ,  3.5  ],
                           [ 4.375,  5.375],
                           [ 9.25 , 10.25 ],
                           [15.25 , 16.25 ],
                           [21.25 , 22.25 ],
                           [26.   , 27.   ],
                           [29.   , 30.   ]])
    npt.assert_array_almost_equal(Xtransform, expectedmat)