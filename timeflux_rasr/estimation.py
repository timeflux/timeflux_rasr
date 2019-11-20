from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.covariance import _check_est
from pyriemann.utils.mean import (mean_covariance, _check_mean_method)
from pyriemann.utils.covariance import covariances
from scipy.linalg import (sqrtm, eigh)
import numpy as np
from utils.utils import epoch

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
    srate : float or int (default: 128)
        Sample rate of the data, in Hz.

   rejection_cutoff : float (default: 5)
        Standard deviation cutoff for rejection. Data portions whose variance is larger
        than this threshold relative to the calibration data are considered missing
        data and will be removed. The most aggressive value that can be used without
        losing too much EEG is 2.5. A quite conservative value would be 5.
    blocksize : int (default: 10)
        Block size for calculating the robust data covariance and thresholds, in samples;
        allows to reduce the memory and time requirements of the robust estimators by this
        factor (down to Channels x Channels x Samples x 16 / blocksize bytes). Default: 10
    window_len : float (default: 0.5)
        Window length in second that is used to check the data for artifact content. This is
        ideally as long as the expected time scale of the artifacts but short enough to
        allow for several 1000 windows to compute statistics over. Default: 0.5.
    window_overlap : float (default: 0.66)
        Window overlap fraction. The fraction of two successive windows that overlaps.
        Higher overlap ensures that fewer artifact portions are going to be missed (but
        is slower). Default: 0.66
    max_dropout_fraction :
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation. Default: 0.1
    min_clean_fraction :
        Minimum fraction of windows that need to be clean, used for threshold
        estimation. Default: 0.25

    Attributes
    ----------
    mixing_ : ndarray, shape(n_chan, n_chan)
        Mixing matrix computed from geometric median covariance matrix U such as
        .. math:: mixing_ = M: M*M = U
    threshold_ : ndarray, shape(n_chan,)
        Threshold operator used to find the subspace dimension such as:
        .. math:: threshold_ = T: X_{clean} = m ( V^T_{clean} M )^+ V^T X

    """

    def __init__(self, srate=128, estimator = 'scm', metric='riemann', n_jobs=1, window_len=0.5,
                 window_overlap=0.66, blocksize=None, rejection_cutoff=5):
        """Init."""
        # TODO:

        self.estimator = _check_est(estimator)
        self.window_len = window_len
        self.window_overlap = window_overlap
        self.srate = srate

        if blocksize is None:
            self.blocksize = self.srate * 0.5 # 500 ms of signal
        else:
            self.blocksize = blocksize

        self.rejection_cutoff = rejection_cutoff

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
        Nt, Ns, Ne = X.shape

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        # epoching
        epochs = epoch(X, self.blocksize, self.blocksize)

        # estimate covariances matrices
        covmats = covariances(epochs, estimator=self.estimator)

        # TODO: implement geometric median instead of geometric mean (robust to bad epochs) and euclidian median (ASR standard)
        covmean = mean_covariance(covmats, metric=self.metric_mean,
                                sample_weight=sample_weight)

        self.mixing_ = sqrtm(covmean)  # estimate matrix matrix

        # TODO: implement manifold-aware PCA (rASR) and standard PCA (ASR)
        evals, evecs = eigh(self.mixing_)  # compute PCA
        indx = np.argsort(evals)    # sort in ascending
        evecs = evecs[:, indx]
        filtered_x = X.dot(evecs)   # apply PCA

        # RMS on sliding window
        window_samples = int(round(window_len * srate))
        epochs_sliding = epoch(filtered_x, window_samples, int(window_samples * window_overlap), axis=0)

        rms_sliding = _rms(epochs_sliding)

        # TODO: implement distribution fitting

        dist_params = np.zeros((Ne,4)) # mu, sig, alpha, beta parameters of estimated distribution

        for c in range(Ne):
            dist_params[c,:] = _fit_eeg_distribution(rms_sliding[:,c])
        self.threshold_ = np.diag(dist_params[:, 0] + self.rejection_cutoff * dist_params[:, 1]).dot(np.transpose(evecs))

        print("rASR calibrated")

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


def _rms(epochs):
    """ Estimate Root Mean Square Amplitude for each epoch and each electrode
    Parameters
    ----------
    epochs : ndarray, shape (n_trials, n_samples, n_electrodes)
        the epochs for the estimation

    Returns
    -------
    RMS : ndarray, shape (n_trials, n_electrodes)
        Root Mean Square Amplitude
    """

    Nt, Ns, Ne = epochs.shape

    RMS = np.zeros((Nt, Ne))

    for i in range(Nt):
        RMS[i, :] = np.sqrt(np.mean(epochs[i, :, :]**2,axis=0))

    return RMS

def _fit_eeg_distribution(X, min_clean_fraction=0.25, max_dropout_fraction=0.1,
                          quants=np.array([0.022, 0.6]) ,step_sizes=np.array([0.01, 0.01]), beta=np.arange(1.7,3.51,0.15)):
    # TODO: implement function
    print("DO FUNCTION")

    return np.zeros((4,))

if __name__ == '__main__':
    print("TEST rASR estimation and checking computation time and each step")
    import time
    C = 4
    S = 100000
    srate = 512
    window_len = int(round(srate * 0.5))
    window_overlap = 0.66

    mean = np.zeros(C)
    cov = [np.random.uniform(0.1, 5, C) for i in range(C)]
    cov = np.dot(np.array(cov),np.transpose(np.array(cov)))
    X = np.random.multivariate_normal(mean, cov, (S,))

    blocksize = int(round(srate * 0.5))

    t = time.time()
    epochs = epoch(X, blocksize, blocksize, axis=0)
    epochs = np.swapaxes(epochs, 1, 2)  # (n_trials, n_channels, n_times)
    print('Elapsed for epoching: %.6f ms' % ((time.time()-t)*1000))

    covmats = covariances(epochs)
    print('Elapsed for epoching+covmats: %.6f ms' % ((time.time()-t)*1000))

    meancovs = mean_covariance(covmats,metric='euclid')
    print('Elapsed for epoching+covmats+mean: %.6f ms' % ((time.time()-t)*1000))

    mixing = sqrtm(meancovs)
    print('Elapsed for epoching+covmats+mean+sqrtm: %.6f ms' % ((time.time()-t)*1000))

    evals, evecs  = eigh(mixing)
    indx = np.argsort(evals) # sort in ascending
    evecs = evecs[:, indx]
    Xf = X.dot(evecs)
    print('Elapsed for epoching+covmats+mean+sqrtm+PCA: %.6f ms' % ((time.time()-t)*1000))


    epochs_sliding = epoch(Xf, window_len, int(window_len*window_overlap), axis=0)

    RMS = _rms(epochs_sliding)
    print('Elapsed for ...+RMS: %.6f ms' % ((time.time()-t)*1000))
