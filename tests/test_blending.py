from timeflux_blending.blending import Blending
from sklearn.utils.estimator_checks import check_estimator

def test_blending_check():
    check_estimator(Blending)