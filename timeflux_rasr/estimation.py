from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.covariance import _check_est
from pyriemann.utils.mean import (mean_covariance, _check_mean_method)
from pyriemann.utils.covariance import covariances
from scipy.linalg import (sqrtm, eigh)
import numpy as np
from utils.utils import epoch
from utils.utils import get_length
from scipy.special import gammaincinv
from scipy.special import gamma


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

    def __init__(self, srate=128, estimator='scm', metric='euclid', n_jobs=1, window_len=0.5,
                 window_overlap=0.66, blocksize=None, rejection_cutoff=5):
        """Init."""
        # TODO:

        self.estimator = _check_est(estimator)
        self.window_len = window_len
        self.window_overlap = window_overlap
        self.srate = srate

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
        X : ndarray, shape (n_trials, n_channels, n_samples)
            Training data.
        y : ndarray, shape (n_trials,) | None, optional
            labels corresponding to each trial, not used (mentioned for sklearn comp)

        Returns
        -------
        self : RASR instance.
            the fitted RASR estimator.
        """
        # TODO: 2D array for sklearn compatibility?

        if (X.shape[0] > 1) and (len(X.shape) < 3):
            print("WARNING: RASR.fit(): support only ONE large chunk of data as input")
            print("WARNING: RASR.fit(): X.shape should be (1, Ns, Ne)")
            print("WARNING: RASR.fit(): assuming X.shape (Ns, Ne)")
            Nt = 1
            Ns, Ne = X.shape  # 2D array (but loosing first dim for trials, not sklearn-friendly)

        elif (X.shape[0] == 1) and (len(X.shape) == 3):
            Nt, Ns, Ne = X.shape  # 3D array (not fully sklearn-compatible). First dim should always be trials.
            X = X[0, :]
        else:
            # TODO: add condition where data X.shape is (Nt, Ns * Ne) but will require additional Ne parameter
            raise ValueError("X.shape should be (1, Ns, Ne) or (Ns, Ne)")

        # epoching
        print("epoching")
        print(self.blocksize)
        epochs = epoch(X, self.blocksize, self.blocksize, axis=0)
        epochs = np.swapaxes(epochs, 1, 2)  # (n_trials, n_channels, n_times)

        print(X.shape)
        print(epochs.shape)

        # estimate covariances matrices
        covmats = covariances(epochs, estimator=self.estimator)
        print(covmats.shape)

        # TODO: implement geometric median instead of geometric mean (robust to bad epochs) and euclidian median (ASR standard)
        print("covmean")
        covmean = mean_covariance(covmats, metric=self.metric_mean)

        self.mixing_ = sqrtm(covmean)  # estimate matrix matrix

        # TODO: implement manifold-aware PCA (rASR) and standard PCA (ASR)
        evals, evecs = eigh(self.mixing_)  # compute PCA
        indx = np.argsort(evals)  # sort in ascending
        evecs = evecs[:, indx]
        filtered_x = X.dot(evecs)  # apply PCA
        print("filtered_x")
        print(filtered_x.shape)

        # RMS on sliding window
        window_samples = int(round(self.window_len * self.srate))
        print(window_samples)
        epochs_sliding = epoch(filtered_x, window_samples, int(window_samples * window_overlap), axis=0)
        epochs = np.swapaxes(epochs, 1, 2)  # (n_trials, n_channels, n_times)
        print("epochs_sliding")
        print(epochs_sliding.shape)
        rms_sliding = _rms(epochs_sliding)
        print("rms_sliding")
        print(rms_sliding.shape)
        # TODO: implement distribution fitting

        dist_params = np.zeros((Ne, 4))  # mu, sig, alpha, beta parameters of estimated distribution

        for c in range(Ne):
            dist_params[c, :] = _fit_eeg_distribution(rms_sliding[:, c])
        self.threshold_ = np.diag(dist_params[:, 0] + self.rejection_cutoff * dist_params[:, 1]).dot(
            np.transpose(evecs))

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
        RMS[i, :] = np.sqrt(np.mean(epochs[i, :, :] ** 2, axis=0))

    return RMS


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
    # TODO: implement function
    print("DO FUNCTION")

    # sanity checks
    if len(X.shape) > 1:
        raise ValueError('X needs to be a 1D ndarray.')
    if get_length(quantile_range) > 2:
        raise ValueError('quantile_range needs to be a 2-elements vector.')
    if any(quantile_range > 7) | any(quantile_range < 0):
        raise ValueError('Unreasonable quantile_range.')
    if any(step_sizes < 0.0001) | any(step_sizes > 0.1):
        raise ValueError('Unreasonable step sizes.')
    if any(beta_range >= 7) | any(beta_range <= 1):
        raise ValueError('Unreasonable shape range.')

    # sort data for quantiles
    print(X.shape)
    X = np.sort(X)
    n = get_length(X)

    # compute z bounds for the truncated standard generalized Gaussian pdf and pdf rescaler for each beta
    zbounds = []
    rescale = []
    for k, b in enumerate(beta_range):
        zbounds.append(np.sign(quantile_range - 1 / 2) *
                       gammaincinv(
                           np.sign(quantile_range - 1 / 2) * (2 * quantile_range - 1),
                           (1 / b) ** (1 / b)
                       )
                       )
        rescale.append(b / (2 * gamma(1 / b)))

    # determine the quantile-dependent limits for the grid search and convert everything in samples

    # we can generally skip the tail below the lower quantile
    lower_min = int(round(min(quantile_range) * n))
    # maximum width in samples is the fit interval if all data is clean
    max_width = int(round(n * np.diff(quantile_range)[0]))
    # minimum width in samples of the fit interval, as fraction of data
    min_width = int(round(min_clean_fraction * n * np.diff(quantile_range)[0]))  #
    print(n)
    print(max_dropout_fraction)
    max_dropout_fraction_n = int(round(max_dropout_fraction * n))
    step_sizes_n = np.round(step_sizes * n).astype(int)

    # get matrix of shifted data ranges
    print(lower_min)
    print(max_dropout_fraction_n)
    print(step_sizes_n)
    indx = np.arange(lower_min, lower_min + max_dropout_fraction_n + 1e-15, step_sizes_n[0]).astype(int)  # epochs start
    print(indx)
    range_ind = np.arange(0, max_width)  # interval indices
    print(range_ind)
    Xs = np.zeros((max_width, get_length(indx)))  # preload entire quantile interval matrix
    print(Xs.shape)
    for k, i in enumerate(indx):
        Xs[:, k] = X[i + range_ind]  # build each quantile interval

    X1 = Xs[0, :]
    Xs = Xs - X1  # substract baseline value for each interval (starting at 0)

    # gridsearch to find optimal fitting coefficient based on given parameters

    opt_val = float("inf")
    gridsearch_val = np.arange(min_width, max_width + 1e-15, step_sizes_n[0]).astype(int)
    print(gridsearch_val)
    for m in gridsearch_val:  # gridsearch for different quantile interval
        # scale and bin the data in the intervals
        print(m)
        print(m.shape)

        nbins = int(round(3 * np.log2(1 + m / 2)))  # scale interval
        print(nbins)
        H = Xs[range(m), :] * nbins / Xs[m - 1, :]  # scale data bins
        print("H shape")
        print(H.shape)
        binscounts = np.zeros((nbins, H.shape[1]))   # init bincounts
        for k in range(H.shape[1]):
            binscounts[:, k], _ = np.histogram(H[:,k], nbins)

        print(binscounts.shape)
        print(binscounts)
        logq = np.log(binscounts + 0.01)  # return log(bincounts) in intervals

        # for each shape value...
        for k, beta in enumerate(beta_range):
            bounds = zbounds[k];

            # evaluate truncated generalized Gaussian pdf at bin centers
            x = bounds[0] + np.linspace(0.5, (nbins-0.5), num=nbins) / nbins * np.diff(bounds)[0];
            p = np.exp(-np.abs(x) ** beta) * rescale[k];
            p = p / np.sum(p);

            # calc KL divergences for the specific interval
            kl = np.sum(p * (np.log(p) - logq)) + np.log(m)
            # TODO: check matlab behaviour of KLdiv and comapre

            # update optimal parameters
            [min_val, idx] = min(kl)
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = beta
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx] + X[m, idx]]


    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)[0]
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha ^ 2) * gamma(3 / beta) / gamma(1 / beta))

    return mu, sig, alpha, beta


if __name__ == '__main__':
    doSequential = False
    doTest = True

    print("TEST rASR: prepare data")
    import time

    C = 4
    S = int(1e5)
    srate = 128
    window_len = int(round(srate * 0.5))
    window_overlap = 0.66

    mean = np.zeros(C)
    cov = [np.random.uniform(0.1, 5, C) for i in range(C)]
    cov = np.dot(np.array(cov), np.transpose(np.array(cov)))
    X = np.random.multivariate_normal(mean, cov, (S,))

    if doSequential:
        print("TEST rASR estimation and checking computation time and each step")

        blocksize = int(round(srate * 0.5))

        t = time.time()
        epochs = epoch(X, blocksize, blocksize, axis=0)
        epochs = np.swapaxes(epochs, 1, 2)  # (n_trials, n_channels, n_times)
        print('Elapsed for epoching: %.6f ms' % ((time.time() - t) * 1000))

        covmats = covariances(epochs)
        print('Elapsed for epoching+covmats: %.6f ms' % ((time.time() - t) * 1000))

        meancovs = mean_covariance(covmats, metric='euclid')
        print('Elapsed for epoching+covmats+mean: %.6f ms' % ((time.time() - t) * 1000))

        mixing = sqrtm(meancovs)
        print('Elapsed for epoching+covmats+mean+sqrtm: %.6f ms' % ((time.time() - t) * 1000))

        evals, evecs = eigh(mixing)
        indx = np.argsort(evals)  # sort in ascending
        evecs = evecs[:, indx]
        Xf = X.dot(evecs)
        print('Elapsed for epoching+covmats+mean+sqrtm+PCA: %.6f ms' % ((time.time() - t) * 1000))

        epochs_sliding = epoch(Xf, window_len, int(window_len * window_overlap), axis=0)

        RMS = _rms(epochs_sliding)
        print('Elapsed for ...+RMS: %.6f ms' % ((time.time() - t) * 1000))

    if doTest:
        # fit test
        print("Test RASR: pipeline...")
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("RASR", RASR())
        ])

        pipeline.fit(np.expand_dims(X, axis=0))
        print("Test RASR: fitted pipeline")
