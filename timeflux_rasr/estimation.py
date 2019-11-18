from sklearn.base import BaseEstimator, TransformerMixin


class RASR(BaseEstimator, TransformerMixin):
    """ RASR
    Implements this (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6499032/) paper.
    Matlab code from the author here:  https://github.com/s4rify/rASRMatlab

    Attributes
    ----------
    foo : float
        xxxx
    """

    def __init__(self, metric='riemann', n_jobs=1):
        """Init."""
        # TODO:

    def partial_fit(self, X, y=None):
        """
        """
        # TODO if relevent

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Training data.
        y : ndarray, shape (n_trials, n_dims) | None, optional
            The regressor(s). Defaults to None.
        Returns
        -------
        X : ndarray, shape (n_trials, n_good_channels, n_times)
            The data without flat channels.
        """
        # TODO; implement that
        return self

    def transform(self, X):
        """Clean signal
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Training data.
        Returns
        -------
        X : ndarray, shape (n_trials, n_good_channels, n_times)
            The data without flat channels.
        """
        # TODO; implement that
        return X

    def fit_transform(self, X, y=None):
        """Estimate rASR and clean signal
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Training data.
        y : ndarray, shape (n_trials, n_dims) | None, optional
            The regressor(s). Defaults to None.
        Returns
        -------
        X : ndarray, shape (n_trials, n_good_channels, n_times)
            The data without flat channels.
        """
        self.fit(X, y)
        return self.transform(X)