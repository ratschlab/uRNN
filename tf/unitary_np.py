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

from functools import partial

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
    assert len(lambdas) == lie_algebra_dim

    L = np.zeros(shape=(n, n), dtype=complex)

    for e in xrange(0, lie_algebra_dim):
        T = lie_algebra_basis_element(n, e, complex_out=True)
        L += lambdas[e]*T
    
    if check_skew_hermitian:
        assert np.array_equal(np.transpose(np.conjugate(L)), -L)

    return L

def lie_algebra_basis_element(n, e, check_skew_hermitian=False, complex_out=False):
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
    if complex_out:
        return T_re + 1j*T_im
    else:
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

# ===
def new_basis_lambdas(lambdas, basis_change):
    """
    Updates lambdas according to change of basis.
    """
    new_lambdas = np.dot(lambdas, basis_change)
    return new_lambdas

def random_unitary_composition(n):
    """
    Returns a random matrix generated from the unitary composition procedure
    of Arjovsky et al [https://arxiv.org/abs/1511.06464]
    
    ... skipping reflection because tired TODO fix
    """
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
    return U

def unitary_matrix(n, method='lie_algebra', lambdas=None, check_unitary=True,
                   basis_change=None):
    """
    Returns a random (or not) unitary matrix of dimension n x n.
    I give no guarantees about the distribution we draw this from.
    To do it 'properly' probably requires a Haar measure.

    Note on 'method':
        - Lie algebra representation (optionally provide lambdas)
        - Using qr decomposition of a random square complex matrix
        - Something which could have been generated by the complex_RNN 
            parametrisation

    Note on 'basis_change':
        Consider the basis defined by lie_algebra_basis(_element) to be the
        'canonical'/standard basis, e.g. the coordinates [0 ... 1 ... 0] etc.
        A new basis element can be represented simply by a vector in this basis
        [alpha_1, ..., alpha_N], which is really
            sum_i alpha_i T_i
        where T_i is the basis element with coordinates [0... 1_i ... 0]
            (to abuse notation slightly)
        So in general, the jth new basis element will be given by
            V_j = sum_i alpha_{ji} T_i
        This matrix alpha_ji defines the change of basis, and is what should be
        provided for this argument.

        Now, in this code I'm using a trick of sorts. Since ultimately we output
            sum_i lambda_i T_i
        we can avoid having to _explicitly_ calculate the new basis elements by
        modifying lambda_i

        If lambda_i are assumed given _with respect to_ the basis V, then we
        wish to calculate
            L = sum_i lambda_i V_i = sum_i lambda_i sum_j alpha_{ij} T_j
              = sum_{i, j} lambda_i alpha_{ij} T_j

        If we define ~lambda_j = sum_i lambda_i alpha_{ij}, then we have
            L = sum_j ~lambda_j T_j

        So calculating L with respect to a new basis is the same as calculating
        L with respect to the old basis, but using a different set of lambdas.
        
        So, in this script when we 'change basis', we're really just changing
        lambda.

    Args:
        n               unitary matrix is size n x n
        method          way of generating the unitary matrix (see note above)
        lambdas         (if method == lie algebra) the set of coefficients with 
                            respect to the basis of the algebra
        check_unitary   check if the result is unitary (mostly for testing)
                            before returning
        basis_change    (if method == lie algebra)  a matrix giving 
                            coefficients of 'new' basis wrt old basis (see above)

    Returns:
        U               the unitary matrix
    """
    # give warnings if unnecessary options provided
    if not method == 'lie_algebra':
        if not lambdas is None:
            print 'WARNING: Method', method, 'selected, but lambdas provided (uselessly)!'
        if not basis_change is None:
            print 'WARNING: Method', method, 'selected, but basis change provided (uselessly)!'
    # now on to generating the Us
    
    if method == 'lie_algebra':
        if lambdas is None:
            # create the lambdas
            lambdas = np.random.normal(size=n*n)
        if not basis_change is None:
            lambdas = new_basis_lambdas(lambdas, basis_change)
        L = lie_algebra_element(n, lambdas)
        U = expm(L)
    elif method == 'qr':
        A = np.random.random(size=(n, n)) + 1j*np.random.random(size=(n, n))
        U, r = np.linalg.qr(A)
    elif method == 'composition':
        U = random_unitary_composition(n)
    else:
        raise ValueError(method)
    
    if check_unitary:
        assert np.allclose(np.dot(U, np.conj(U.T)), np.eye(n))
    return U

# === some grad hacks === #

def numerical_partial_gradient(e, L, n, U, dcost_dU_re, dcost_dU_im, EPSILON):
    """ For giving to the pool.
    """
    dU_dlambda = (expm(L + EPSILON*lie_algebra_basis_element(n, e, complex_out=True)) - U)/EPSILON
    delta = np.trace(np.dot(dcost_dU_re, np.real(dU_dlambda)) + np.dot(dcost_dU_im, np.imag(dU_dlambda)))
    return delta

def numgrad_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, 
                          EPSILON=1e-5, learning_rate=0.01):
    """
    Given dcost/dU, get dcost/dlambdas
    Using... numerical differentiation. :(
    """
    # first term (d cost / d U)
    assert dcost_dU_re.shape == dcost_dU_im.shape
    assert len(lambdas) == n*n

    # second term (d U / d lambdas)
    # meanwhile update lambdas
    L = lie_algebra_element(n, lambdas)
    U = expm(L)

    # parallel version
    numerical_parallel = partial(numerical_partial_gradient, L=L, U=U, n=n, 
                                 dcost_dU_re=dcost_dU_re, dcost_dU_im=dcost_dU_im, 
                                 EPSILON=EPSILON)
    dlambdas = np.array(map(numerical_parallel, xrange(n*n)))
    lambdas += learning_rate*dlambdas
    
    # having updated the lambdas, get new U
    U_new = expm(lie_algebra_element(n, lambdas))

    return np.real(U_new), np.imag(U_new), lambdas, dlambdas
