#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
#
# A stripped down experiment to test the learning of unitary matrices.
# aka stuck on gradients for RNN edition
# aka back 2 basics
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         27/4/16
# ------------------------------------------
#
#

import numpy as np
import pdb

from data import generate_unitary_learning
from unitary import unitary_matrix
from scipy.fftpack import fft2, ifft2
from functools import partial

# === loss functions === #
def trivial_loss(parameters, batch):
    """
    For testing.
    Parameters is just a vector, which we add to x to get y-hat. Very simple.
    """
    x, y = batch

    y_hat = x + parameters
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

def less_trivial_loss(parameters, batch):
    """
    For testing.
    Parameters is now a matrix!
    """
    x, y = batch
    d = x.shape[1]

    M = parameters.reshape(d, d)
    y_hat = np.dot(x, M)

    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

# === parametrisation-specific functions === #

def do_reflection(x, v_re, v_im):
    """
    Hey, it's this function again! Woo!
    (mostly copypasta from theano, with T replaced by np all over)
    """
    # FOR NOW OK
    input_re = np.real(x)
    # alpha
    input_im = np.imag(x)
    # beta
    reflect_re = v_re
    # mu
    reflect_im = v_im
    # nu

    vstarv = (reflect_re**2 + reflect_im**2).sum()

    # (the following things are roughly scalars)
    # (they actually are as long as the batch size, e.g. input[0])
    input_re_reflect_re = np.dot(input_re, reflect_re)
    # αμ
    input_re_reflect_im = np.dot(input_re, reflect_im)
    # αν
    input_im_reflect_re = np.dot(input_im, reflect_re)
    # βμ
    input_im_reflect_im = np.dot(input_im, reflect_im)
    # βν

    a = np.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
    # outer(αμ - βν, mu)
    b = np.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
    # outer(αν + βμ, nu)
    c = np.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
    # outer(αμ - βν, nu)
    d = np.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)
    # outer(αν + βμ, mu)

    output_re = input_re - 2. / vstarv * (d - c)
    output_im = input_im - 2. / vstarv * (a + b)
    
    output = output_re + 1j*output_im

    return output

def complex_RNN_loss(parameters, batch, permutation):
    """
    Transform data according to the complex_RNN transformations.
    (requires importing a bunch of things and weird tensorflow hax)
    NOTE: no longer folding in any input data...

    Parameters, once again, numpy array of values.
    """
    x, y = batch
    d = x.shape[1]

    # === expand the parameters === #

    # diag1 
    thetas1 = parameters[0:d]
    diag1 = np.diag(np.cos(thetas1) + 1j*np.sin(thetas1))
    # reflection 1
    reflection1_re = parameters[d:2*d]
    reflection1_im = parameters[2*d:3*d]
    # fixed permutation (get from inputs)
    # diag 2
    thetas2 = parameters[3*d:4*d]
    diag2 = np.diag(np.cos(thetas2) + 1j*np.sin(thetas2))
    # reflection 2
    reflection2_re = parameters[4*d:5*d]
    reflection2_im = parameters[5*d:6*d]
    # diag 3
    thetas3 = parameters[6*d:7*d]
    diag3 = np.diag(np.cos(thetas3) + 1j*np.sin(thetas3))

    # === do the transformation === #
    step1 = np.dot(x, diag1)
    step2 = fft2(step1)
    step3 = do_reflection(step2, reflection1_re, reflection1_im)
    step4 = np.dot(step3, permutation)
    step5 = np.dot(step4, diag2)
    step6 = ifft2(step5)
    step7 = do_reflection(step6, reflection2_re, reflection2_im)
    step8 = np.dot(step6, diag3)
    
    # POSSIBLY do relu_mod...

    # === now calculate the loss ... === #
    y_hat = step8
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

def general_unitary_loss(parameters, batch):
    """
    Hey, it's my one! Rendered very simple by existence of helper functions. :)
    """
    x, y = batch
    d = x.shape[1]

    lambdas = parameters
    U = unitary_matrix(d, lambdas)

    y_hat = np.dot(x, U.T)
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))

    return loss

# === utility functions === #
def numerical_gradient(loss_function, parameters, batch, EPSILON=10e-6):
    """
    Calculate the numerical gradient of a given loss function with respect to a np.array of parameters.
    """
    original_loss = loss_function(parameters, batch)

    assert len(parameters.shape) == 1
    parameters_gradient = np.zeros_like(parameters)

    n_parameters = len(parameters)
    for i in xrange(n_parameters):
        parameters_epsilon = np.zeros(n_parameters)
        parameters_epsilon[i] = EPSILON
       
        new_loss = loss_function(parameters + parameters_epsilon, batch)
        
        difference = (new_loss - original_loss)/EPSILON
        parameters_gradient[i] = difference
    
    return original_loss, parameters_gradient

def train_loop(batches, loss_function, initial_parameters, LEARNING_RATE=0.001, vali_data=None):
    """
    Arguments:
        batches:            list of training batches
        loss_function:      function for calculating loss, surprisingly enough
        initial_parameters  numpy vector of initial parameter values, surprisingly enough
                            should be consistent with loss_function...

    Returns:
        parameters          trained parameters

    Side-effects:
        prints out loss on test and validation data (if not None) during training
    """
    parameters = initial_parameters
    for (i, batch) in enumerate(batches):
        loss, parameters_gradient = numerical_gradient(loss_function, parameters, batch)
        print i, 'TRAIN:', loss
        if not vali_data is None and i % 20 == 0:
            vali_loss = loss_function(parameters, vali_data)
            print i, '\t\tVALI:', vali_loss
        # *now* update parameters
        parameters = parameters + LEARNING_RATE*parameters_gradient
    print 'Training complete!'
    return parameters

# === main loop === #
def main(experiment='trivial', TEST=True):
    """
    For testing, right now.
    """
    d = 5
    n_examples = 100
    n_batches = 1000
    
    # set up test data
    U = unitary_matrix(d)
    batches = generate_unitary_learning(U, n_examples, n_batches)
    vali_batch = batches[0]
    test_batch = batches[1]
    train_batches = batches[2:]

    if experiment == 'trivial':
        initial_parameters = np.random.normal(size=d)
        loss_fn = trivial_loss
    elif experiment == 'less_trivial':
        initial_parameters = np.random.normal(size=d*d)
        loss_fn = less_trivial_loss
    elif experiment == 'complex_RNN':
        initial_parameters = np.random.normal(size=7*d)
        permutation = np.random.permutation(np.eye(d))
        loss_fn = partial(complex_RNN_loss, permutation=permutation)
    elif experiment == 'general_unitary':
        initial_parameters = np.random.normal(size=d*d)
        loss_fn = general_unitary_loss
    else:
        print experiment
        raise NotImplementedError
  
    # train loop!
    trained_parameters = train_loop(train_batches, loss_fn, initial_parameters, vali_data=vali_batch)

    if TEST:
        test_loss = loss_fn(trained_parameters, test_batch)
        print 'TEST:', test_loss

    return trained_parameters
