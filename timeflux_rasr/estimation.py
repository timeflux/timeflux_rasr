from sklearn.base import BaseEstimator, TransformerMixin


class RASR(BaseEstimator, TransformerMixin):
    """ RASR
    Implements this (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6499032/) paper.
    Matlab code from the author here:  https://github.com/s4rify/rASRMatlab

    Parameters
    ----------
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `pyriemann.utils.covariance`
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.

    Attributes
    ----------
    mixing_ : array, shape(n_chan, n_chan)
        Mixing matrix computed from geometric median covariance matrix U such as
        .. math:: mixing_ = M: M*M = U
    threshold_ : array, shape(n_chan,)
        Threshold operator used to find the subspace dimension such as:
        .. math:: threshold_ = T: X_{clean} = m ( V^T_{clean} M )^+ V^T X

    """

    def __init__(self, estimator = 'scm', metric='riemann', n_jobs=1):
        """Init."""
        # TODO:

        self.estimator = estimator
        self.metric = metric


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