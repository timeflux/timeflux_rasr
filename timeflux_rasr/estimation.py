from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.covariance import _check_est
from pyriemann.utils.mean import (mean_covariance, _check_mean_method)
from pyriemann.utils.covariance import covariances
from scipy.linalg import (sqrtm, eigh)
import numpy as np

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
    mixing_ : ndarray, shape(n_chan, n_chan)
        Mixing matrix computed from geometric median covariance matrix U such as
        .. math:: mixing_ = M: M*M = U
    threshold_ : ndarray, shape(n_chan,)
        Threshold operator used to find the subspace dimension such as:
        .. math:: threshold_ = T: X_{clean} = m ( V^T_{clean} M )^+ V^T X

    """

    def __init__(self, srate=None, estimator = 'scm', metric='riemann', n_jobs=1, window_len=0.5,):
        """Init."""
        # TODO:

        self.estimator = _check_est(estimator)
        self.window_len=window_len
        self.window_overlap

        if srate is None:


        # initialize metrics
        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric # unused for now

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def partial_fit(self, X, y=None):
        """
        """
        # TODO if relevent

    def fit(self, X, y=None, sample_weight=None):
        """
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            Training data.
        y : ndarray, shape (n_trials,) | None, optional
            labels corresponding to each trial, not used
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each trial. if None, each trial is treated with
            equal weights. Useful if too many training epochs are corrupted
            to avoid biased distribution estimation.

        Returns
        -------
        self : RASR instance.
            the fitted RASR estimator.
        """
        # TODO; implement that

        # estimate covariances matrices
        covmats = covariances(X, estimator=self.estimator)

        if sample_weight is None:
            sample_weight = numpy.ones(X.shape[0])

        # TODO: implement geometric median instead of geometric mean (robust to bad epochs)
        covmean = mean_covariance(X, metric=self.metric_mean,
                                sample_weight=sample_weight)

        self.mixing_ = sqrtm(covmean)

        evals, evecs = eigh(self.mixing_)

        # TODO: implement matrix RMS
        rms_vals = rms(np.dot(np.evecs


        return self

    def transform(self, X):
        """Clean signal
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_trials)
            Training data.
        Returns
        -------
        X : ndarray, shape (n_trials, n_good_channels, n_trials)
            The data without flat channels.
        """
        # TODO; implement that
        return X

    def fit_transform(self, X, y=None):
        """Estimate rASR and clean signal
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_trials)
            Training data.
        y : ndarray, shape (n_trials,) | None, optional
            labels corresponding to each trial, not used
        Returns
        -------
        X : ndarray, shape (n_trials, n_good_channels, n_times)
            Cleaned data.
        """
        self.fit(X, y)
        return self.transform(X)