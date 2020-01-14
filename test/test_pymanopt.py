from utils.pymanopt import nonlinear_eigh
import numpy as np

def test_nonlinear_eigh_unitary():
    n = 5
    X = np.eye(n)
    Xfilt = nonlinear_eigh(X, n-1)
    # TODO: assert when it should be permuted eye
    print(Xfilt)

def test_inv_versus_regression():
    # do the test to see if it fits to nonlinear_eigh
    X = np.eye
    np.linalg
    assert False
