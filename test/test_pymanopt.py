from utils.pymanopt import nonlinear_eigh
import numpy as np

def test_nonlinear_eigh_unitary():
    n = 10
    X = np.eye(n)
    Xfilt = nonlinear_eigh(X, int(n-1))
    # TODO: compare with eigh (should be relativement closed for i

def test_inv_versus_regression():
    # do the test to see if it fits to nonlinear_eigh