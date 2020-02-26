from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np


class Blending(BaseEstimator, TransformerMixin):
    """Smooths features temporally to avoid discontinuities by sine-cosine blending interpolation.

    This estimator smooths features to avoid discontinuities by blending the redundant samples that has been
    transformed by different operations. Only samples with the exact same timestamp will be blended. Features with an
    unique timestamp will be left untouched. Therefore this function is useful only if the two following criteria are
    met:
    - there is an overlapping between the features of each consecutive observation
    - the features have been transformed by different operators or by an adaptive operator.

    Parameters
    ----------
    window_overlap : int (default=1)
        Number of overlapping features that will be used for blending.
    merge : bool (default=False)
        If set to True, the output will be a unique 2D matrix when all windows will be collapsed and overlapping samples
        merged. By default, each window will be blended with the previous window.

    Attributes
    ----------
    n_channels_ : int
        The dimension managed by the fitted Blending, e.g. number of electrodes.`
    last_window_ : ndarray, shape (Nt, Ne)
        The last window observed during the last call of transform. It will be blended with the first window.
    """

    def __init__(self, window_overlap=1, merge=False):
        if window_overlap is None:
            raise ValueError("window_overlap parameter is not initialized.")
        self.window_overlap = window_overlap

        self.last_window_ = None  # init last_window_

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : ndarray, shape (n_trials,  n_samples, n_channels)
            Training data.
        y : ndarray, shape (n_trials,) | None, optional
            labels corresponding to each trial, not used (mentioned for sklearn comp)

        Returns
        -------
        self : Blending instance.
            the fitted Blending estimator.
        """
        X = check_array(X, allow_nd=True)
        self.n_channels_ = X.shape[2]
        return self

    def transform(self, X):
        """Blend signals
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_samples, n_channels)
            Data to clean, already filtered
        Returns
        -------
        Xblended : ndarray, shape (n_trials, n_samples, n_channels)
            Cleaned data
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_channels_')

        X = check_array(X, allow_nd=True)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[2] != self.n_channels_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        if self.last_window_ is None:
            # generate flat last_window_ for first blending
            self.last_window_ = np.zeros((X.shape[1], X.shape[2]))

        # estimate the blending coefficients
        blend_coeff = (1 - np.cos(np.pi * (np.arange(0, self.window_overlap) / (self.window_overlap - 1)))) / 2
        blend_coeff = blend_coeff[:, None]
        for k in range(X.shape[0]):
            last_values = self.last_window_[-self.window_overlap:, :]  # sampled that will be blended
            new_values = X[k, 0:self.window_overlap, :]
            X[k, 0:self.window_overlap, :] = ((1 - blend_coeff) * last_values) + (blend_coeff * new_values)

        # # apply the reconstruction to intermediate samples(using raised - cosine blending)
        #     blend = (1 - np.cos(np.pi * np.arrange(0,self.window_overlap) / (self.window_overlap))) / 2
        #     data(:, subrange) = bsxfun( @ times, blend, R * data(:, subrange)) + bsxfun( @ times, 1 - blend, state.last_R * data(:, subrange));
        #     end
        #     [last_n, state.last_R, state.last_trivial] = deal(n, R, trivial);

        return X

    def fit_transform(self, X, y=None):
        """
                Parameters
                ----------
                X : ndarray, shape (n_trials,  n_samples, n_channels)
                    Training data.
                y : ndarray, shape (n_trials,) | None, optional
                    labels corresponding to each trial, not used (mentioned for sklearn comp)
                Returns
                -------
                X : ndarray, shape (n_trials, n_samples, n_channels)
                    blended data
                """
        self.fit(X, y)

        return self.transform(X)
