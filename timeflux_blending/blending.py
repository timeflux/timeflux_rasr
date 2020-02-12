from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

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
    blending_coverage : float (default 1.0)
        Fraction of the overlapping features that will be used for blending. By default the blending will start at the
        first overlapping feature and end at the last overlapping feature (maximum coverage of 1.0). If the overlapping
        is very long, one can consider to diminish the coverage to reduce the transitory period.
    """
    def __init__(self, blending_coverage=1.0):
        self.blending_coverage = blending_coverage

    def fit(self, X, y=None):
        X = check_array(X, allow_nd=True)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        X = check_array(X, allow_nd=True)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return X

    def fit_transform(self,X, y=None):
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
