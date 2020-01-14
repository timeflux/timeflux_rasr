from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.covariance import _check_est
from pyriemann.utils.covariance import covariances
from scipy.linalg import (sqrtm, eigh)
import numpy as np
from utils.utils import epoch, get_length, geometric_median
from scipy.special import gammaincinv
from scipy.special import gamma
import logging
import warnings


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
    max_dimension : Maximum dimensionality of artifacts to remove. Up to this many dimensions (or up
        to this fraction of dimensions) can be removed for a given data segment. If the
        algorithm needs to tolerate extreme artifacts a higher value than the default
        may be used (the maximum fraction is 1.0). Default 0.66

    Attributes
    ----------
    mixing_ : ndarray, shape(n_chan, n_chan)
        Mixing matrix computed from geometric median covariance matrix U such as
        .. math:: mixing_ = M: M*M = U
    threshold_ : ndarray, shape(n_chan,)
        Threshold operator used to find the subspace dimension such as:
        .. math:: threshold_ = T: X_{clean} = m ( V^T_{clean} M )^+ V^T X
    """

    def __init__(self, srate=128, estimator='scm', metric='euclid', window_len=0.5,
                 window_overlap=0.66, blocksize=None, rejection_cutoff=5, max_dimension=0.66):
        """Init."""
        # TODO:

        self.estimator = _check_est(estimator)
        self.window_len = window_len
        self.window_overlap = window_overlap
        self.srate = srate
        self.max_dimension = max_dimension

        if blocksize is None:
            self.blocksize = int(round(self.srate * 0.5))  # 500 ms of signal
        else:
            self.blocksize = int(round(blocksize))

        self.rejection_cutoff = rejection_cutoff

        # initialize metrics
        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric  # unused for now

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

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : ndarray, shape (n_trials,  n_samples, n_channels)
            Training data, already filtered.
        y : ndarray, shape (n_trials,) | None, optional
            labels corresponding to each trial, not used (mentioned for sklearn comp)

        Returns
        -------
        self : RASR instance.
            the fitted RASR estimator.
        """

        # TODO: 2D array for sklearn compatibility? (see below)
        if (X.shape[0] > 1) and (len(X.shape) < 3):
            warnings.warn("RASR.fit(): support only ONE large chunk of data as input, \n "
                          "            X.shape should be (1, Ns, Ne), assuming X.shape (Ns, Ne)")
            Nt = 1
            Ns, Ne = X.shape  # 2D array (but loosing first dim for trials, not sklearn-friendly)

        elif len(X.shape) == 3:
            # concatenate all epochs
            Nt, Ns, Ne = X.shape  # 3D array (not fully sklearn-compatible). First dim should always be trials.
            X = X.reshape((X.shape[1] * X.shape[0], X.shape[2]))
            if Nt > 1:
                warnings.warn("RASR.fit(): concatenating all epochs. \n"
                                "            it may cause issues if overlapping")

        else:
            # TODO: add condition where data X.shape is (Nt, Ns * Ne) but will require additional Ne parameter
            raise ValueError("X.shape should be (1, Ns, Ne) or (Ns, Ne)")

        assert Ne < Ns, "number of samples should be higher than number of electrodes, check than \n" \
                        + "X.shape is (n_trials,  n_samples, n_channels) or (n_samples, n_channels) "
        # epoching
        logging.info("epoching")
        epochs = epoch(X, self.blocksize, self.blocksize, axis=0)

        # estimate covariances matrices
        covmats = covariances(np.swapaxes(epochs, 1, 2), estimator=self.estimator)  # (n_trials, n_channels, n_times)

        # geometric median (maybe not best with double reshape) but implement as is in matlab
        # NOTE: while the term geometric median is used, it is NOT riemannian median but euclidian median, i.e.
        # it might be suboptimal for Symmetric Positive Definite matrices.
        logging.info("geometric median")
        # covmean = mean_covariance(covmats, metric=self.metric_mean)
        covmean = np.reshape(geometric_median(
            np.reshape(covmats,
                       (covmats.shape[0], covmats.shape[1] * covmats.shape[2])
                       )
        ), (covmats.shape[1], covmats.shape[2])
        )

        self.mixing_ = sqrtm(covmean)  # estimate matrix matrix

        # TODO: implement both manifold-aware PCA (rASR) and standard PCA (ASR)
        evals, evecs = eigh(self.mixing_)  # compute PCA
        indx = np.argsort(evals)  # sort in ascending
        evecs = evecs[:, indx]
        filtered_x = X.dot(evecs)  # apply PCA

        # RMS on sliding window
        window_samples = int(round(self.window_len * self.srate))
        epochs_sliding = epoch(filtered_x, window_samples, int(window_samples * self.window_overlap), axis=0)
        rms_sliding = _rms(epochs_sliding)

        dist_params = np.zeros((Ne, 4))  # mu, sig, alpha, beta parameters of estimated distribution

        for c in range(Ne):
            dist_params[c, :] = _fit_eeg_distribution(rms_sliding[:, c])
        self.threshold_ = np.diag(dist_params[:, 0] + self.rejection_cutoff * dist_params[:, 1]).dot(
            np.transpose(evecs))

        logging.info("rASR calibrated")

        return self

    def transform(self, X):
        """Clean signal
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_samples, n_channels)
            Data to clean, already filtered
        Returns
        -------
        Xclean : ndarray, shape (n_trials, n_samples, n_channels)
            Cleaned data
        """
        logging.info("RASR.transform(): check input")
        # TODO: 2D array for sklearn compatibility? (see below)
        if (X.shape[0] > 1) and (len(X.shape) < 3):
            warnings.warn("RASR.transform(): assuming X.shape (Ns, Ne)")
            Nt = 1
            Ns, Ne = X.shape  # 2D array (but loosing first dim for trials, not sklearn-friendly)
            X = np.expand_dims(X, 0)

        elif len(X.shape) == 3:
            Nt, Ns, Ne = X.shape  # 3D array (not fully sklearn-compatible). First dim should always be trials.

        else:
            # TODO: add condition where data X.shape is (Nt, Ns * Ne) but will require additional Ne parameter
            raise ValueError("X.shape should be (1, Ns, Ne) or (Ns, Ne)")

        Xclean = np.zeros((Nt, Ns, Ne))

        assert Ne < Ns, "number of samples should be higher than number of electrodes, check than \n" \
                        + "X.shape is (n_trials,  n_samples, n_channels) or (n_samples, n_channels) "

        logging.info("RASR.transform(): compute covariances")

        covmats = covariances(np.swapaxes(X, 1, 2), estimator=self.estimator)  # (n_trials, n_channels, n_times)

        # TODO: update the mean covariance (required for online update) only in partial_fit ?

        logging.info("RASR.transform(): clean each epoch")

        for k in range(Nt):
            # TODO: HAVE BOTH euclidian PCA and Riemannian PCA (PGA) using pymanopt
            evals, evecs = eigh(covmats[k, :])  # compute PCA
            # TODO: comment in matlab "use eigenvalues in descending order" but actually is doing in ascending
            indx = np.argsort(evals)  # sort in ascending
            evecs = evecs[:, indx]

            keep = (evals[indx] < sum((self.threshold_ * evecs) ** 2)) | \
                   (np.arange(Ne) < (Ne * (1 - self.max_dimension)))

            keep = np.expand_dims(keep, 0)  # for element wise multiplication that follows

            spatialfilter = np.linalg.pinv(keep.transpose() * evecs.transpose().dot(self.mixing_))

            R = self.mixing_.dot(spatialfilter).dot(evecs.transpose())

            Xclean[k, :] = X[k, :].dot(R.transpose())  # suboptimal in term of memory but great for debug

        return Xclean

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
            Cleaned data
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

    return np.sqrt(np.mean(epochs ** 2, axis=1))


def _fit_eeg_distribution(X, min_clean_fraction=0.25, max_dropout_fraction=0.1,
                          quantile_range=np.array([0.022, 0.6]), step_sizes=np.array([0.01, 0.01]),
                          beta_range=np.arange(1.7, 3.51, 0.15)):
    """ Estimate the mean and standard deviation of clean EEG from contaminated data

    This function estimates the mean and standard deviation of clean EEG from a sample of amplitude
    values (that have preferably been computed over short windows) that may include a large fraction
    of contaminated samples. The clean EEG is assumed to represent a generalized Gaussian component in
    a mixture with near-arbitrary artifact components. By default, at least 25% (min_clean_fraction) of
    the data must be clean EEG, and the rest can be contaminated. No more than 10%
    (max_dropout_fraction) of the data is allowed to come from contamination that cause lower-than-EEG
    amplitudes (e.g., sensor unplugged). There are no restrictions on artifacts causing
    larger-than-EEG amplitudes, i.e., virtually anything is handled (with the exception of a very
    unlikely type of distribution that combines with the clean EEG samples into a larger symmetric
    generalized Gaussian peak and thereby "fools" the estimator). The default parameters should be
    fine for a wide range of settings but may be adapted to accommodate special circumstances.

    The method works by fitting a truncated generalized Gaussian whose parameters are constrained by
    min_clean_fraction, max_dropout_fraction, quantile_range, and beta_range. The alpha and beta parameters
    of the gen. Gaussian are also returned. The fit is performed by a grid search that always finds a
    close-to-optimal solution if the above assumptions are fulfilled.

    Parameters
    ----------
    X : ndarray, shape (n_samples,)
        vector of amplitude values of EEG, possible containing artifacts
        (coming from single samples or windowed averages)

    min_clean_fraction : float (default: 0.25)
        Minimum fraction of values in X that needs to be clean

    max_dropout_fraction : float (default: 0.1)
        Maximum fraction of values in X that can be subject to
        signal dropouts (e.g., sensor unplugged)

    quantile_range : ndarray, shape (2,) (default: [0.022 0.6])
        Quantile range [lower,upper] of the truncated generalized Gaussian distribution
        that shall be fit to the EEG contents

    step_sizes : ndarray, shape (2,) (default: [0.01 0.01])
        Step size of the grid search; the first value is the stepping of the lower bound
        (which essentially steps over any dropout samples), and the second value
        is the stepping over possible scales (i.e., clean-data quantiles)

    beta_range : ndarray, shape (n_points,) (default: np.arange(1.70, 3.51, 0.15))
        Range that the clean EEG distribution's shape parameter beta may take

    Returns
    -------
    Mu : float
        estimated mean of the clean EEG distribution

    Sigma : float
        estimated standard deviation of the clean EEG distribution

    Alpha : float
        estimated scale parameter of the generalized Gaussian clean EEG distribution (optional)

    Beta : float
        estimated shape parameter of the generalized Gaussian clean EEG distribution (optional)

    """

    # sanity checks
    if len(X.shape) > 1:
        raise ValueError('X needs to be a 1D ndarray.')

    n = len(X)

    if get_length(quantile_range) > 2:
        raise ValueError('quantile_range needs to be a 2-elements vector.')
    if any(quantile_range > 1) | any(quantile_range < 0):
        raise ValueError('Unreasonable quantile_range.')
    if any(step_sizes < 0.0001) | any(step_sizes > 0.1):
        raise ValueError('Unreasonable step sizes.')
    if any(step_sizes * n < 1):
        raise ValueError(f"Step sizes compared to actual number of samples available, step_sizes * n should be "
                         f"greater than 1 (current value={step_sizes * n}")
    if any(beta_range >= 7) | any(beta_range <= 1):
        raise ValueError('Unreasonable shape range.')

    # sort data for quantiles
    X = np.sort(X)

    if any(beta_range <= 1):
        raise ValueError('Unreasonable shape range.')

    # compute z bounds for the truncated standard generalized Gaussian pdf and pdf rescaler for each beta
    zbounds = []
    rescale = []
    for k, b in enumerate(beta_range):
        zbounds.append(np.sign(quantile_range - 0.5) *
                       gammaincinv(
                           (1 / b),
                           np.sign(quantile_range - 0.5) * (2 * quantile_range - 1)
                       ) ** (1 / b)
                       )
        rescale.append(b / (2 * gamma(1 / b)))

    # determine the quantile-dependent limits for the grid search and convert everything in samples

    # we can generally skip the tail below the lower quantile
    lower_min = int(round(min(quantile_range) * n))
    # maximum width in samples is the fit interval if all data is clean
    max_width = int(round(n * np.diff(quantile_range)[0]))
    # minimum width in samples of the fit interval, as fraction of data
    min_width = int(round(min_clean_fraction * n * np.diff(quantile_range)[0]))  #
    max_dropout_fraction_n = int(round(max_dropout_fraction * n))
    step_sizes_n = np.round(step_sizes * n).astype(int)
    assert any(step_sizes_n >= 1)   # should be catched earlier but double-checking

    # get matrix of shifted data ranges
    indx = np.arange(lower_min, lower_min + max_dropout_fraction_n + 0.5, step_sizes_n[0]).astype(int)  # epochs start

    range_ind = np.arange(0, max_width)  # interval indices
    Xs = np.zeros((max_width, get_length(indx)))  # preload entire quantile interval matrix
    for k, i in enumerate(indx):
        Xs[:, k] = X[i + range_ind]  # build each quantile interval

    X1 = Xs[0, :]
    Xs = Xs - X1  # substract baseline value for each interval (starting at 0)

    # gridsearch to find optimal fitting coefficient based on given parameters
    opt_val = float("inf")
    opt_lu = float("inf")
    opt_bounds = float("inf")
    opt_beta = float("inf")
    gridsearch_val = np.arange(max_width - 1, min_width, -step_sizes_n[0]).astype(int)

    for m in gridsearch_val:  # gridsearch for different quantile interval
        # scale and bin the data in the intervals
        nbins = int(round(3 * np.log2(1 + m / 2))) + 1  # scale interval
        H = Xs[range(m), :] * nbins / Xs[m - 1, :]  # scale data bins
        binscounts = np.zeros((nbins, H.shape[1]))  # init bincounts
        for k in range(H.shape[1]):
            binscounts[:, k], _ = np.histogram(H[:, k], nbins)

        logq = np.log(binscounts + 0.01)  # return log(bincounts) in intervals

        # for each shape value...
        for k, beta in enumerate(beta_range):
            bounds = zbounds[k]

            # evaluate truncated generalized Gaussian pdf at bin centers
            x = bounds[0] + np.linspace(0.5, (nbins - 0.5), num=nbins) / nbins * np.diff(bounds)[0]
            p = np.exp(-np.abs(x) ** beta) * rescale[k]
            p = p / np.sum(p)

            # calc KL divergences for the specific interval
            kl = np.sum(p * (np.log(p) - np.transpose(logq)), axis=1) + np.log(m)

            # update optimal parameters
            idx = np.argmin(kl)
            if kl[idx] < opt_val:
                opt_val = kl[idx]
                opt_beta = beta
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx] + Xs[m, idx]]

    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)[0]
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha ** 2) * gamma(3 / beta) / gamma(1 / beta))

    return mu, sig, alpha, beta


if __name__ == '__main__':
    # TODO: remove all following section and put it into pytest
    # NOTE(Louis): I don't know exatly how to do the sequential testing with pytest without saving in attributes
    import time

    doSequential = True
    logging.info("TEST rASR: prepare data")

    C = 4
    S = int(1e5)
    srate = 128
    window_len = int(round(srate * 0.5))
    window_overlap = 0.66

    mean = np.zeros(C)
    cov = [np.random.uniform(0.1, 5, C) for i in range(C)]
    cov = np.dot(np.array(cov), np.transpose(np.array(cov)))
    X = np.random.multivariate_normal(mean, cov, (S,))

    X[:, -1] += (np.random.randint(0, 1000, (S,)) > 995) * 1000  # artefact with 0.5 % chance

    if doSequential:
        logging.info("TEST rASR estimation and checking computation time and each step")

        blocksize = int(round(srate * 0.5))

        t = time.time()

        epochs = epoch(X, blocksize, blocksize, axis=0)

        logging.info('Elapsed for epoching: %.6f ms' % ((time.time() - t) * 1000))

        covmats = covariances(np.swapaxes(epochs, 1, 2))  # (n_trials, n_channels, n_times)
        logging.info('Elapsed for epoching+covmats: %.6f ms' % ((time.time() - t) * 1000))

        # meancovs = mean_covariance(covmats, metric='euclid')
        meancovs = np.reshape(geometric_median(
            np.reshape(covmats,
                       (covmats.shape[0], covmats.shape[1] * covmats.shape[2])
                       )
        ), (covmats.shape[1], covmats.shape[2])
        )
        logging.info('Elapsed for epoching+covmats+mean: %.6f ms' % ((time.time() - t) * 1000))

        mixing = sqrtm(meancovs)
        logging.info('Elapsed for epoching+covmats+mean+sqrtm: %.6f ms' % ((time.time() - t) * 1000))

        evals, evecs = eigh(mixing)
        indx = np.argsort(evals)  # sort in ascending
        evecs = evecs[:, indx]
        Xf = X.dot(evecs)
        logging.info('Elapsed for epoching+covmats+mean+sqrtm+PCA: %.6f ms' % ((time.time() - t) * 1000))

        epochs_sliding = epoch(Xf, window_len, int(window_len * window_overlap), axis=0)

        rms_sliding = _rms(epochs_sliding)
        logging.info('Elapsed for ...+RMS: %.6f ms' % ((time.time() - t) * 1000))

        dist_params = np.zeros((C, 4))  # mu, sig, alpha, beta parameters of estimated distribution

        for c in range(C):
            dist_params[c, :] = _fit_eeg_distribution(rms_sliding[:, c])
        logging.info('Elapsed for ...+eeg_fit: %.6f ms' % ((time.time() - t) * 1000))

        # test median methods

        points = np.random.uniform(1e-15, 1, (300, 16))

        points_med = geometric_median(points)
