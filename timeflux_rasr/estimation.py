from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.covariance import _check_est
from pyriemann.utils.covariance import covariances
from scipy.linalg import (sqrtm, eigh)
import numpy as np
from utils.utils import epoch, geometric_median
from scipy.special import gammaincinv
from scipy.special import gamma
import logging
import warnings
from sklearn.utils.validation import check_array, check_is_fitted


class RASR(BaseEstimator, TransformerMixin):
    """ RASR
    Implements this (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6499032/) paper.
    Matlab code from the author here:  https://github.com/s4rify/rASRMatlab

    Parameters
    ----------
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `pyriemann.utils.covariance`
    rejection_cutoff : float (default: 3.0)
        Standard deviation cutoff for rejection. Data portions whose variance is larger
        than this threshold relative to the calibration data are considered missing
        data and will be removed. The most aggressive value that can be used without
        losing too much EEG is 2.5. A quite conservative value would be 5.
    max_dimension : float (default: 0.66)
        Maximum dimensionality of artifacts to remove. Up to this many dimensions (or up
        to this fraction of dimensions) can be removed for a given data segment. If the
        algorithm needs to tolerate extreme artifacts a higher value than the default
        may be used (the maximum fraction is 1.0).
    max_dropout_fraction : float (default: 0.1)
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation in _fit_eeg_distribution.
    min_clean_fraction : float (default: 0.25)
        Minimum fraction of windows that need to be clean, used for threshold
        estimation in _fit_eeg_distribution.
    quantile_range, step_sizes, beta_range :
        additional parameters passed to _fit_eeg_distribution (should be kept as default in general).


    Attributes
    ----------
    Ne_ : int
        The dimension managed by the fitted RASR, e.g. number of electrodes.
    mixing_ : ndarray, shape(n_chan, n_chan)
        Mixing matrix computed from geometric median covariance matrix U such as
        .. math:: mixing_ = M: M*M = U
    threshold_ : ndarray, shape(n_chan,)
        Threshold operator used to find the subspace dimension such as:
        .. math:: threshold_ = T: X_{clean} = m ( V^T_{clean} M )^+ V^T X
    """

    def __init__(self, estimator='scm', rejection_cutoff=3.0, max_dimension=0.66, **kwargs):
        """Init."""

        self.estimator = _check_est(estimator)

        self.max_dimension = max_dimension

        self.rejection_cutoff = rejection_cutoff

        self.args_eeg_distribution = kwargs
        self.Ne_ = None  # will be initialized during training

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
        X = check_array(X, allow_nd=True)
        shapeX = X.shape
        if len(shapeX) == 3:
            # concatenate all epochs
            Nt, Ns, Ne = shapeX  # 3D array (not fully sklearn-compatible). First dim should always be trials.
        else:
            raise ValueError("X.shape should be (n_trials, n_samples, n_electrodes).")

        assert Ne < Ns, "number of samples should be higher than number of electrodes, check than \n" \
                        + "X.shape is (n_trials,  n_samples, n_channels)."

        if shapeX[0] < 100:
            raise ValueError("Training requires at least 100 of trials to fit.")

        self.Ne_ = Ne   # save attribute for testing

        epochs = X.copy()
        epochs = check_array(epochs, allow_nd=True)

        # estimate covariances matrices
        covmats = covariances(np.swapaxes(epochs, 1, 2), estimator=self.estimator)  # (n_trials, n_channels, n_times)
        covmats = check_array(covmats, allow_nd=True)

        # geometric median
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
        epochs = np.tensordot(epochs, evecs, axes=(2, 0))  # apply PCA to epochs

        # RMS on sliding window
        rms_sliding = _rms(epochs)

        dist_params = np.zeros((Ne, 4))  # mu, sig, alpha, beta parameters of estimated distribution

        #TODO: use joblib to parrallelize this loop (code bottleneck)
        for c in range(Ne):
            dist_params[c, :] = _fit_eeg_distribution(rms_sliding[:, c], **self.args_eeg_distribution)
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
        check_is_fitted(self, ['Ne_', 'mixing_', 'threshold_'])
        X = check_array(X, allow_nd=True)
        logging.info("RASR.transform(): check input")
        shapeX = X.shape

        if len(shapeX) == 3:
            Nt, Ns, Ne = shapeX
        else:
            raise ValueError("X.shape should be (n_trials, n_samples, n_electrodes).")

        Xclean = np.zeros((Nt, Ns, Ne))

        assert Ne < Ns, "number of samples should be higher than number of electrodes, check than \n" \
                        + "X.shape is (n_trials,  n_samples, n_channels)."

        logging.info("RASR.transform(): compute covariances")

        covmats = covariances(np.swapaxes(X, 1, 2), estimator=self.estimator)  # (n_trials, n_channels, n_times)

        logging.info("RASR.transform(): clean each epoch")

        # TODO: parallelizing the loop for efficiency
        for k in range(Nt):
            # TODO: HAVE BOTH euclidian PCA and Riemannian PCA (PGA) using pymanopt
            evals, evecs = eigh(covmats[k, :])  # compute PCA
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
    """ Estimate Root Mean Square Amplitude for each epoch and each electrode.

    .. math::

        rms = \sqrt{(\frac{1}{n})\sum_{i=1}^{n}(x_{i})^{2}}

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


def _fit_eeg_distribution(X, min_clean_fraction=None, max_dropout_fraction=None,
                          quantile_range=None, step_sizes=None,
                          beta_range=None):
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

    # default parameters
    if min_clean_fraction is None:
        min_clean_fraction = 0.25
    if max_dropout_fraction is None:
        max_dropout_fraction = 0.1
    if quantile_range is None:
        quantile_range = np.array([0.022, 0.6])
    if step_sizes is None:
        step_sizes = np.array([0.01, 0.01])
    if beta_range is None:
        beta_range = np.arange(1.7, 3.51, 0.15)

    # check valid parameters
    n = len(X)
    quantile_range = np.array(quantile_range)
    step_sizes = np.array(step_sizes)
    beta_range = np.array(beta_range)

    if not len(quantile_range) == 2:
        raise ValueError('quantile_range needs to be a 2-elements vector.')
    if any(quantile_range > 1) | any(quantile_range < 0):
        raise ValueError('Unreasonable quantile_range.')
    if any(step_sizes < 0.0001) | any(step_sizes > 0.1):
        raise ValueError('Unreasonable step sizes.')
    if any(step_sizes * n < 1):
        raise ValueError(f"Step sizes compared to actual number of samples available, step_sizes * n should be "
                         f"greater than 1 (current value={step_sizes * n}. More training data required.")
    if any(beta_range >= 7) | any(beta_range <= 1):
        raise ValueError('Unreasonable shape range.')

    # sort data for quantiles
    X = np.sort(X)

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
    assert indx.dtype == "int"

    range_ind = np.arange(0, max_width)  # interval indices
    Xs = np.zeros((max_width, len(indx)))  # preload entire quantile interval matrix
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