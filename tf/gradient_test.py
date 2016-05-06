#!/usr/bin/env ipython
#
# Script to test when/if I can truncate the infinite sum for calculating the gradient.
# aka maths is hard, code is easy
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         6/5/16
# ------------------------------------------
#

import numpy as np
from math import factorial
from scipy.linalg import expm
import unitary
import pdb

# === CONSTANTS === #
J_MAX = 100
N_REPS = 20

# === SETUP === #
def setup(n):
    # (I have irritatingly written a random set of the unitary functions for tf)
    # get the the basis set (so inefficient!)
    Ts = np.zeros(shape=(n*n, n, n), dtype=complex)
    for e in xrange(n*n):
        T_re, T_im = unitary.lie_algebra_basis_element(n, e)
        Ts[e, :, :] = T_re + 1j*T_im

    # create L by choosing random lambdas
    lambdas = np.random.normal(size=(n*n))
    L = np.einsum('i,ijk', lambdas, Ts)

    # select a random element of Ts
    i = np.random.choice(n*n)
    T = Ts[i, :, :]
    return L, T

# === GET TRACE === #
def get_trace(L, T, J_MAX):
    # (so inefficient)
    n = L.shape[0]
    # trace will store np.mean(L^{j-a} T L^a)/j!
    trace = [0]*J_MAX
    # generate powers of L as we go along
    L_powers = np.zeros(shape=(J_MAX, n, n), dtype=complex)
    L_powers[0] = np.eye(n)
    trace[0] = 1
    #trace[1] = np.linalg.norm(np.mean(np.dot(L, T)))
    trace[1] = trace[0] + np.linalg.norm(np.mean(np.dot(L, L)))
    L_powers[1] = L
    for j in xrange(2, J_MAX):
        L_powers[j] = np.dot(L, L_powers[j-1])
        accumulator = np.zeros_like(L)
        for a in xrange(j):
            #accumulator += np.dot(np.dot(L_powers[j-a], T), L_powers[a])
            accumulator += np.dot(np.dot(L_powers[j-a], L), L_powers[a])
        accumulator /= factorial(j)
#        pdb.set_trace()
        trace_complex = np.mean(accumulator)
        trace[j] = trace[j-1] + np.linalg.norm(trace_complex)
        if trace[j] > 100:
            pdb.set_trace()
    return trace

# === GET COMBOTRACE === #
def get_combotrace(N, N_REPS, f=None):
    combo_trace = np.zeros(shape=(J_MAX, N_REPS))
    for l in xrange(N_REPS):
        L, T = setup(N)
        trace = get_trace(L, T, J_MAX)
        assert len(trace) == J_MAX
        if f is not None:
            for (j, t) in enumerate(trace):
                f.write(str(j) + ' ' + str(N)+ ' ' + str(l) + ' ' + str(t) + '\n')
        combo_trace[:, l] = trace
    if f is None:
        means = np.mean(combo_trace, axis=1)
        stds = np.std(combo_trace, axis=1)
        results = zip(list(means), list(stds))
        for m, s in results:
            print m, s
    return combo_trace

# === another outer loop === #
fout = open('gradient_test_traces.txt', 'w')
fout.write('j d rep val\n')
for n in [1, 2, 5, 10, 15, 25, 50]:
    print n
    ct = get_combotrace(n, N_REPS, fout)
fout.close()
