from utils.pymanopt import nonlinear_eigh
import numpy as np

def test_nonlinear_eigh_unitary_n_minus_1():
    n = 5
    X = np.eye(n)
    Xfilt = nonlinear_eigh(X, n-1)
    # TODO: assert Xfilt should be a permuted identity matrix of rank n-1 (it failed rightnow)
    print(Xfilt)

def test_nonlinear_eigh_unitary_n():
    n = 5
    X = np.eye(n)
    Xfilt = nonlinear_eigh(X, n)
    # TODO: assert Xfilt should be a permuted identity matrix of rank n (it failed rightnow)
    print(Xfilt)
