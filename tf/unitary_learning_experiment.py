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
import cPickle

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
        train_trace         list of training losses
        vali_trace          list of vali losses
        parameters          trained parameters

    Side-effects:
        prints out loss on test and validation data (if not None) during training
    """
    parameters = initial_parameters
    train_trace = []
    vali_trace = []
    for (i, batch) in enumerate(batches):
        loss, parameters_gradient = numerical_gradient(loss_function, parameters, batch)
        #print i, 'TRAIN:', loss
        train_trace.append(loss)
        if not vali_data is None and i % 100 == 0:
            vali_loss = loss_function(parameters, vali_data)
            print i, '\t\tVALI:', vali_loss
            vali_trace.append(vali_loss)
        # *now* update parameters
        parameters = parameters - LEARNING_RATE*parameters_gradient
    print 'Training complete!'
    return train_trace, vali_trace, parameters

def run_experiment(loss_fn, batches, initial_parameters, TEST=True):
    """
    Such laziness.
    """
    vali_batch = batches[0]
    test_batch = batches[1]
    train_batches = batches[2:]

    train_trace, vali_trace, trained_parameters = train_loop(train_batches, loss_fn, initial_parameters, vali_data=vali_batch)
    
    if TEST:
        test_loss = loss_fn(trained_parameters, test_batch)
        print 'TEST:', test_loss
    else:
        test_loss = -1
    return train_trace, vali_trace, test_loss

def random_baseline(batches):
    """
    OK
    """
    test_batch = batches[1]
    x, y = test_batch
    d = x.shape[1]

    M = np.random.normal(shape=(d, d))
    y_hat = np.dot(M, x)
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

# === main loop === #
def main(experiments=['trivial', 'less_trivial', 'complex_RNN', 'general_unitary']):
    """
    For testing, right now.
    """
    d = 5
    batch_size = 100
    n_batches = 10000
    
    # set up test data
    U = unitary_matrix(d)
    batches = generate_unitary_learning(U, batch_size, n_batches)

    # prepare trace dicts
    train_traces = dict()
    vali_traces = dict()
    test_losses = dict()

    if 'trivial' in experiments:
        print 'Running "trivial" experiment!'
        loss_fn = trivial_loss
        initial_parameters = np.random.normal(size=d)
        # actually run
        train_trace, vali_trace, test_loss = run_experiment(loss_fn, batches, initial_parameters, TEST=True)
        # record
        train_traces['trivial'] = train_trace
        vali_traces['trivial'] = vali_trace
        test_losses['trivial'] = test_loss
    if 'less_trivial' in experiments:
        print 'Running "less_trivial" experiment!'
        loss_fn = less_trivial_loss
        initial_parameters = np.random.normal(size=d*d)
        # actually run
        train_trace, vali_trace, test_loss = run_experiment(loss_fn, batches, initial_parameters, TEST=True)
        # record
        train_traces['less_trivial'] = train_trace
        vali_traces['less_trivial'] = vali_trace
        test_losses['less_trivial'] = test_loss
    if 'complex_RNN' in experiments:
        print 'Running "complex_RNN" experiment!'
        permutation = np.random.permutation(np.eye(d))
        loss_fn = partial(complex_RNN_loss, permutation=permutation)
        initial_parameters = np.random.normal(size=7*d)
        # actually run
        train_trace, vali_trace, test_loss = run_experiment(loss_fn, batches, initial_parameters, TEST=True)
        # record
        train_traces['complex_RNN'] = train_trace
        vali_traces['complex_RNN'] = vali_trace
        test_losses['complex_RNN'] = test_loss
    if 'general_unitary' in experiments:
        print 'Running "general_unitary" experiment!'
        loss_fn = general_unitary_loss
        initial_parameters = np.random.normal(size=d*d)
        # actually run
        train_trace, vali_trace, test_loss = run_experiment(loss_fn, batches, initial_parameters, TEST=True)
        # record
        train_traces['general_unitary'] = train_trace
        vali_traces['general_unitary'] = vali_trace
        test_losses['general_unitary'] = test_loss

    # get a random baseline
    random_test_loss = random_baseline(batches)
    test_losses['random'] = random_test_loss
    
    experiment_settings = 'd'+str(d) + '_bn'+str(batch_size) + '_nb' + str(n_batches)
    cPickle.dump(train_traces, open(experiment_settings+'_train.pk', 'wb'))
    cPickle.dump(vali_traces, open(experiment_settings+'_train.pk', 'wb'))
    cPickle.dump(test_losses, open(experiment_settings+'_test.pk', 'wb'))
    return train_traces, vali_traces, test_losses
