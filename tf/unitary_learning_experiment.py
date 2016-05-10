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
import sys

from data import generate_unitary_learning, create_batches
from unitary_np import unitary_matrix, project_to_unitary
from scipy.fftpack import fft, ifft
from functools import partial
from multiprocessing import Pool
from random import sample

# === some globals === #
MEASURE_SKIP = 100
NUM_WORKERS = 32

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

def free_matrix_loss(parameters, batch):
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

def do_reflection(x, v_re, v_im, theano_reflection=False):
    """
    Hey, it's this function again! Woo!
    NOTE/WARNING: theano_reflection gives a DIFFERENT RESULT to the other one...
    see this unresolved issue:
    https://github.com/amarshah/complex_RNN/issues/2
    """
    if theano_reflection:
        # (mostly copypasta from theano, with T replaced by np all over)
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
    else:
        # do it the 'old fashioned' way
        v = v_re + 1j*v_im
        # aka https://en.wikipedia.org/wiki/Reflection_%28mathematics%29#Reflection_through_a_hyperplane_in_n_dimensions
        # but with conj v dot with x
        output = x - (2.0/np.dot(v, np.conj(v))) * np.outer(np.dot(x, np.conj(v)), v)

    return output

def complex_RNN_loss(parameters, batch, permutation, theano_reflection=True):
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
    step2 = fft(step1)
    step3 = do_reflection(step2, reflection1_re, reflection1_im, theano_reflection)
    #step3 = step2
    step4 = np.dot(step3, permutation)
    step5 = np.dot(step4, diag2)
    step6 = ifft(step5)
    step7 = do_reflection(step6, reflection2_re, reflection2_im, theano_reflection)
    #step7 = step6
    step8 = np.dot(step7, diag3)
    
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
    U = unitary_matrix(d, lambdas=lambdas)

    y_hat = np.dot(x, U.T)
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))

    return loss

# === utility functions === #
def numerical_partial_gradient(i, loss_function=None, parameters=None, batch=None, EPSILON=10e-6):
    """
    For giving to the pool.
    """
    n_parameters = len(parameters)
    parameters_epsilon = np.zeros(n_parameters)
    parameters_epsilon[i] = EPSILON
    new_loss = loss_function(parameters + parameters_epsilon, batch)
    return new_loss
    
def numerical_gradient(loss_function, parameters, batch, pool, EPSILON=10e-6):
    """
    Calculate the numerical gradient of a given loss function with respect to a np.array of parameters.
    """
    original_loss = loss_function(parameters, batch)

    assert len(parameters.shape) == 1
    parameters_gradient = np.zeros_like(parameters)

    numerical_parallel = partial(numerical_partial_gradient, 
                                 loss_function=loss_function,
                                 parameters=parameters,
                                 batch=batch)
    new_losses = np.array(pool.map(numerical_parallel, xrange(len(parameters))))
    parameters_gradient = (new_losses - original_loss)/EPSILON

    return original_loss, parameters_gradient

def train_loop(batches, loss_function, initial_parameters, pool, loginfo, LEARNING_RATE=0.001, vali_data=None, PROJECT_TO_UNITARY=False):
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
        loss, parameters_gradient = numerical_gradient(loss_function, parameters, batch, pool)
        batch_size = batch[0].shape[0]
        if i % MEASURE_SKIP == 0:
            # only record some of the points, for memory efficiency
            loginfo['train_file'].write(loginfo['exp_name']+' '+str((i + 1)*batch_size)+' '+str(loss)+' ' + str(loginfo['rep'])+' ' + loginfo['method']+'\n')
            if not vali_data is None:
                vali_loss = loss_function(parameters, vali_data)
                if i % (MEASURE_SKIP*4) == 0:
                    print (i + 1)*batch_size, '\t\tVALI:', vali_loss
                loginfo['vali_file'].write(loginfo['exp_name']+' '+str((i + 1)*batch_size)+' '+str(vali_loss)+' ' + str(loginfo['rep'])+' ' + loginfo['method']+'\n')
                # flush both files now and again
                loginfo['vali_file'].flush()
                loginfo['train_file'].flush()
        # *now* update parameters
        parameters = parameters - LEARNING_RATE*parameters_gradient
        if PROJECT_TO_UNITARY:
            # use the polar decomposition to re-unitarise the matrix
            parameters = project_to_unitary(parameters, check_unitary=False)
    print 'Training complete!'
    return parameters

def run_experiment(loss_fn, batches, initial_parameters, pool, loginfo, TEST=True, project=False):
    """
    Such laziness.
    """
    vali_batch = batches[0]
    test_batch = batches[1]
    train_batches = batches[2:]

    trained_parameters = train_loop(train_batches, 
                                    loss_fn, 
                                    initial_parameters, 
                                    pool,
                                    loginfo,
                                    vali_data=vali_batch,
                                    PROJECT_TO_UNITARY=project)
    
    if TEST:
        test_loss = loss_fn(trained_parameters, test_batch)
        print 'TEST:', test_loss
    else:
        test_loss = -1
    return test_loss

def random_baseline(test_batch, method):
    """
    Test using a random, UNITARY matrix.
    """
    x, y = test_batch
    d = x.shape[1]

    M = unitary_matrix(d, method=method)
    y_hat = np.dot(x, M)
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

def true_baseline(U, test_batch):
    """
    Use the actual generating unitary matrix.
    """
    x, y = test_batch
    
    y_hat = np.dot(x, U.T)
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

# === main loop === #
def main(d=5, experiments=['projection', 'complex_RNN', 'general_unitary'], method=None, n_reps=3, n_epochs=1, noise=0.01, start_from_rep=0):
    """
    For testing, right now.
    """
    # OPTIONS
    batch_size = 20
    n_batches = 50000
    if n_epochs is None:
        n_epochs = d
        print 'WARNING: No n_epochs provided, using', n_epochs
    
    experiment_settings = 'output/simple/fft1_d'+str(d) + '_noise'+str(noise) + '_bn'+str(batch_size) + '_nb' + str(n_batches)

    # save to an R-plottable file because I am so very lazy
    R_vali = open(experiment_settings+'_vali.txt', 'a')
    R_train = open(experiment_settings+'_train.txt', 'a')
    R_test = open(experiment_settings+'_test.txt', 'a')
    if start_from_rep == 0:
        # headers
        header = 'experiment training_examples loss rep method'
        R_vali.write(header+'\n')
        R_train.write(header+'\n')
        R_test.write('experiment loss rep method\n')
        # flush
        R_vali.flush()
        R_train.flush()
        R_test.flush()
    # put together
    loginfo = {'vali_file': R_vali, 'train_file': R_train, 'exp_name': None, 'rep': None, 'method': None}

    # some parallelism
    pool = Pool(NUM_WORKERS)

    for rep in xrange(start_from_rep, start_from_rep + n_reps):
        # randomly select the method
        method = sample(['lie_algebra', 'qr', 'composition'], 1)[0]
        print rep, ': generating U using:', method

        loginfo['method'] = method
        loginfo['rep'] = rep

        # set up test data
        U = unitary_matrix(d, method=method)
        batches = generate_unitary_learning(U, batch_size, n_batches, n_epochs, noise=noise)

        # prepare trace dicts
        test_losses = dict()

        # get 'baselines'
        test_batch = batches[1]
        random_test_loss = random_baseline(test_batch, method=method)
        test_losses['random_unitary'] = random_test_loss
        R_test.write('random_unitary ' + str(random_test_loss) + ' ' + str(rep) + ' ' + method +'\n')
        true_test_loss = true_baseline(U, test_batch)
        test_losses['true'] = true_test_loss
        R_test.write('true ' + str(true_test_loss) + ' ' + str(rep) + ' ' + method +'\n')
        R_test.flush()

        if 'trivial' in experiments:
            print 'Running "trivial" experiment!'
            loginfo['exp_name'] = 'trivial'
            loss_fn = trivial_loss
            initial_parameters = np.random.normal(size=d) + 1j*np.random.normal(size=d)
            # actually run
            test_loss = run_experiment(loss_fn, batches, initial_parameters, pool, loginfo, TEST=True)
            # record
            test_losses['trivial'] = test_loss
            # save now
            R_test.write('trivial ' + str(test_loss) + ' ' + str(rep) + ' ' + method +'\n')
            R_test.flush()
        if 'free_matrix' in experiments:
            print 'Running "free_matrix" experiment!'
            loginfo['exp_name'] = 'free_matrix'
            loss_fn = free_matrix_loss
            initial_parameters = np.random.normal(size=d*d) + 1j*np.random.normal(size=d*d)
            # actually run
            test_loss = run_experiment(loss_fn, batches, initial_parameters, pool, loginfo, TEST=True)
            # record
            test_losses['free_matrix'] = test_loss
            R_test.write('free_matrix ' + str(test_loss) + ' ' + str(rep) + ' ' + method +'\n')
            R_test.flush()
        if 'projection' in experiments:
            print 'Running "projection" experiment!'
            loginfo['exp_name'] = 'projection'
            # (this is just free_matrix with reprojecting to unitary...)
            loss_fn = free_matrix_loss
            initial_parameters = np.random.normal(size=d*d) + 1j*np.random.normal(size=d*d)
            # actually run
            test_loss = run_experiment(loss_fn, batches, initial_parameters, pool, loginfo, TEST=True, project=True)
            # record
            test_losses['projection'] = test_loss
            R_test.write('projection ' + str(test_loss) + ' ' + str(rep) + ' ' + method +'\n')
            R_test.flush()
        if 'complex_RNN_vanilla' in experiments:
            print 'Running "complex_RNN_vanilla" experiment!'
            loginfo['exp_name'] = 'complex_RNN_vanilla'
            permutation = np.random.permutation(np.eye(d))
            loss_fn = partial(complex_RNN_loss, 
                              permutation=permutation,
                              theano_reflection=True)
            # all of these parameters are real
            initial_parameters = np.random.normal(size=7*d)
            # actually run
            test_loss = run_experiment(loss_fn, batches, initial_parameters, pool, loginfo, TEST=True)
            # record
            test_losses['complex_RNN_vanilla'] = test_loss
            R_test.write('complex_RNN_vanilla ' + str(test_loss) + ' ' + str(rep) + ' ' + method +'\n')
            R_test.flush()
        if 'complex_RNN' in experiments:
            print 'Running "complex_RNN" experiment!'
            loginfo['exp_name'] = 'complex_RNN'
            permutation = np.random.permutation(np.eye(d))
            loss_fn = partial(complex_RNN_loss, 
                              permutation=permutation,
                              theano_reflection=False)
            # all of these parameters are real
            initial_parameters = np.random.normal(size=7*d)
            # actually run
            test_loss = run_experiment(loss_fn, batches, initial_parameters, pool, loginfo, TEST=True)
            # record
            test_losses['complex_RNN'] = test_loss
            R_test.write('complex_RNN ' + str(test_loss) + ' ' + str(rep) + ' ' + method +'\n')
            R_test.flush()
        if 'general_unitary' in experiments:
            print 'Running "general_unitary" experiment!'
            loginfo['exp_name'] = 'general_unitary'
            loss_fn = general_unitary_loss
            initial_parameters = np.random.normal(size=d*d)
            # actually run
            test_loss = run_experiment(loss_fn, batches, initial_parameters, pool, loginfo, TEST=True)
            # record
            test_losses['general_unitary'] = test_loss
            R_test.write('general_unitary ' + str(test_loss) + ' ' + str(rep) + ' ' + method +'\n')
            R_test.flush()
 
        print test_losses

        # save test things...
        for (exp_name, loss) in test_losses.iteritems():
            R_test.write(exp_name + ' ' + str(loss) + ' ' + str(rep) + ' ' + method +'\n')
    
        R_test.flush()

    R_vali.close()
    R_train.close()
    R_test.close()

    return True

## actually run !!! ###
if len(sys.argv) == 3:
    print 'Getting system arguments!'
    d=int(sys.argv[1])
    noise=float(sys.argv[2])
    print 'Dimension:', d
    print 'Noise:', noise
    main(d=d, noise=noise)
