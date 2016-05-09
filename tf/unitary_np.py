#!/usr/bin/env ipython
#
# Functions pertaining to the unitary group and its associated Lie algebra.
# (RETURN NP OBJECTS)
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         12/4/16
# ------------------------------------------

import numpy as np
import pdb

from scipy.linalg import expm, polar
from scipy.fftpack import fft2, ifft2

def lie_algebra_element(n, lambdas, check_skew_hermitian=False):
    # TODO: np-ify
    """
    Explicitly construct an element of a Lie algebra, assuming 'default' basis,
    given a set of coefficients (lambdas).

    That is,
        L = sum lambda_i T^i
    where T^i are the basis elements and lambda_i are the coefficients

    The Lie algebra is u(n), associated to the Lie group U(n) of n x n unitary matrices.
    The Lie algebra u(n) is n x n skew-Hermitian matrices. 

    Args:
        n:              see above
        lambdas:        a ndarray of length n x n
    Returns:
        a rank-2 numpy ndarray dtype complex, the element of the Lie algebra
        a 2D Tensor of shape [n, n], dtype tf.complex64, the element of the Lie algebra

    POSSIBLE TODO: combine this with function to generate elements of the basis,
        or create a basis-element generator
    """
    lie_algebra_dim = n*n
    assert len(lambdas) = lie_algebra_dim

    L = np.zeros(shape=(n, n), dtype=complex)

    for e in xrange(0, lie_algebra_dim):
        T_re, T_im = lie_algebra_basis_element(n, e)
        L += lambdas[e]*T_re + 1j*lambdas[e]*T_im
    
    if check_skew_hermitian:
        assert np.array_equal(np.transpose(np.conjugate(L)), -L)

    return L

def lie_algebra_basis_element(n, e, check_skew_hermitian=False):
    """
    Return a *single* element (the e-th one) of a basis of u(n).
    See lie_algebra_basis for more details.

    Args:
        n:  as in u(n)
        e:  which element?

    Returns:
        Two numpy arrays: the real part and imaginary parts of the element.
    """
    lie_algebra_dim = n*n
    T_re = np.zeros(shape=(n, n), dtype=np.float32)
    T_im = np.zeros(shape=(n, n), dtype=np.float32)

    # three cases: imaginary diagonal, imaginary off-diagonal, real off-diagonal
    # (reasonably sure this is how you convert these, it makes sense to me...)
    i = e / n
    j = e % n
    if i > j:
        # arbitrarily elect these to be the real ones
        T_re[i, j] = 1
        T_re[j, i] = -1
    else:
        # (includes i == j, the diagonal part)
        T_im[i, j] = 1
        T_im[j, i] = 1
    if check_skew_hermitian:
        basis_matrix = T_re + 1j*T_im
        assert np.array_equal(np.transpose(np.conjugate(basis_matrix)), -basis_matrix)
    return T_re, T_im

def lie_algebra_basis(n):
    """
    Generate a basis of the Lie algebra u(n), associated to the Lie group U(n) of n x n unitary matrices.
    The Lie algebra u(n) is n x n skew-Hermitian matrices. 
    A skew-Hermitian matrix is a square complex matrix A such that A^{dagger} = -A
    This means that conj(A_ij) = -A_ji
    The Lie algebra has real dimension n^2.

    One construction for the basis is as follows:
    - the first n elements are:
        diag(..., i, ...) (i in the nth position)
        ... Since diag(...)^T = diag(...) and conj(i) = -i
    - the next n(n-1)/2 elements are:
        a = 1..n, b = a..n, A_ab = 1, A_ba = -1
        ... Since real, so we just need A^T = -A
    - the next n(n-1)/2 elements are:
        a = 1..n, b = a..n, A_ab = i, A_ba = i
        ... Since imaginary, so we just need A^T = A
    
    Note that I will construct the real and imaginary parts separatey, and join them at the end as a tf.tensor.

    Args:
        n: see above
    Returns:
        rank-3 numpy ndarray, complex, list of the n^2 basis matrices of the Lie algebra  
    
    TODO: double-check all the maths here
    """
    lie_algebra_dim = n*n
    tensor_re = np.zeros(shape=(lie_algebra_dim, n, n), dtype=np.float32)
    tensor_im = np.zeros(shape=(lie_algebra_dim, n, n), dtype=np.float32)

    for e in xrange(0, lie_algebra_dim):
        T_re, T_im = lie_algebra_basis_element(n, e)
        tensor_re[e, :, :] = T_re
        tensor_im[e, :, :] = T_im

    # ensure they are indeed skew-Hermitian
    basis = tensor_re + 1j*tensor_im 
    for basis_matrix in basis:
        assert np.array_equal(np.transpose(np.conjugate(basis_matrix)), -basis_matrix)
       
    return basis

def project_to_unitary(parameters, check_unitary=True):
    """
    Use polar decomposition to find the closest unitary matrix.
    """
    # parameters must be a square number (TODO: FIRE)
    n = int(np.sqrt(len(parameters)))

    A = parameters.reshape(n, n)
    U, p = polar(A, side='left')

    if check_unitary:
        assert np.allclose(np.dot(U, np.conj(U.T)), np.eye(n))

    parameters = U.reshape(n*n)
    return parameters

def unitary_matrix(n, method='lie_algebra', lambdas=None, check_unitary=True):
    """
    Returns a random unitary matrix of dimension n x n.
    I give no guarantees about the distribution we draw this from.
    To do it 'properly' probably requires a Haar measure.

    Options:
        - Lie algebra representation (optionally provide lambdas)
        - Using qr decomposition of a random square complex matrix
        - Something which could have been generated by the complex_RNN parametrisation
    """
    if method == 'lie_algebra':
        if lambdas is None:
            # create the lambdas
            lambdas = np.random.normal(size=n*n)
        L = lie_algebra_element(n, lambdas)
        U = expm(L)
    elif method == 'qr':
        if not lambdas is None:
            print 'WARNING: Method qr selected, but lambdas provided (uselessly)!'
        A = np.random.random(size=(n, n)) + 1j*np.random.random(size=(n, n))
        U, r = np.linalg.qr(A)
    elif method == 'composition':
        # skipping reflection because tired TODO fix
        # diag
        thetas1 = np.random.uniform(low=-np.pi, high=np.pi, size=n)
        diag1 = np.diag(np.cos(thetas1) + 1j*np.sin(thetas1))
        # fft
        step2 = fft2(diag1)
        # skipping reflection
        step3 = step2
        # permutation
        permutation = np.random.permutation(np.eye(n))
        step4 = np.dot(step3, permutation)
        # diag
        thetas2 = np.random.uniform(low=-np.pi, high=np.pi, size=n)
        diag2 = np.diag(np.cos(thetas2) + 1j*np.sin(thetas2))
        step5 = np.dot(step4, diag2)
        # ifft
        step6 = ifft2(step5)
        # skipping reflection
        step7 = step6
        # final diag
        thetas3 = np.random.uniform(low=-np.pi, high=np.pi, size=n)
        diag3 = np.diag(np.cos(thetas3) + 1j*np.sin(thetas3))
        U = np.dot(step7, diag3)
    else:
        print method
        raise NotImplementedError
    if check_unitary:
        assert np.allclose(np.dot(U, np.conj(U.T)), np.eye(n))
    return U

# === some grad hacks === #
def U_from_grads(U_grad, lambdas):
    """
    TODO: all
    TODO: update name
    """
    # YOLO FOR NOW
    # --- get new lambdas
    n = U_grad.shape[0]
    # --- todo: finish
    return np.random.normal(size=U_grad.shape)
