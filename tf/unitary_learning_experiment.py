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

# === utility functions === #
def numerical_partial_gradient(i, n, loss_function, old_loss, parameters, 
                               batch, EPSILON=10e-6):
    """
    Gradient in a single coordinate direction.
    (returns a float)
    """
    parameters_epsilon = np.zeros(n)
    parameters_epsilon[i] = EPSILON
    new_loss = loss_function(parameters + parameters_epsilon, batch)
    gradient = (new_loss - old_loss)/EPSILON
    return gradient

def numerical_random_gradient(i, learnable_parameters, n, loss_function, 
                              old_loss, parameters, batch, EPSILON=10e-6):
    """
    Gradient in a random direction.
    (returns a vector)
    """
    # get a random direction in the learnable subspace
    random_direction = np.random.normal(size=len(learnable_parameters))
    random_direction /= np.linalg.norm(random_direction)
    # perturb the parameters
    parameters_epsilon = np.zeros(n)
    parameters_epsilon[learnable_parameters] = random_direction
    # calculate finite difference
    new_loss = loss_function(parameters + EPSILON*random_direction, batch)
    difference = (new_loss - old_loss)/EPSILON
    # each component gets a gradient in proportion to the random direction
    gradient_vector = np.zeros(n)
    gradient_vector[learnable_parameters] = difference*random_direction
    return gradient_vector

def numerical_gradient(loss_function, parameters, batch, pool, 
                       random_projection=False, update_indices=None):
    """
    Calculate the numerical gradient of a given loss function with respect to 
    a np.array of parameters.

    Args:
        loss_function
        parameters
        batch
        pool
        random_projection   bool indicating if we use random projection method
        update_indices      an array/iterable of which indices to calculate 
                                gradients for
    """
    original_loss = loss_function(parameters, batch)

    assert len(parameters.shape) == 1
    d_params = np.zeros_like(parameters)
    n = len(parameters)

    if update_indices is None:
        update_indices = xrange(len(parameters))

    if random_projection:
        # how many random projections to use? hmm...
        N_RANDOM = 5
        numerical_parallel = partial(numerical_random_gradient, 
                                     learnable_parameters=update_indices,
                                     n=n,
                                     loss_function=loss_function,
                                     old_loss=original_loss,
                                     parameters=parameters,
                                     batch=batch)
        gradients_list = pool.map(numerical_parallel, xrange(N_RANDOM))
        # seemingly numpy will convert this to an array or something
        d_params = np.sum(gradients_list, axis=0)
    else:
        numerical_parallel = partial(numerical_partial_gradient, 
                                     n=n,
                                     loss_function=loss_function,
                                     old_loss=original_loss,
                                     parameters=parameters,
                                     batch=batch)
        gradients = np.array(pool.map(numerical_parallel, update_indices))
        d_params[update_indices] = gradients

    return original_loss, d_params

def train_loop(experiment, train_batches, vali_batch, pool, loginfo):
    """
    The main training loop...

    Arguments:
        experiment          Experiment object, contains useful things
        train_batches   
        vali_batch
        pool
        loginfo

    Returns
        parameters          trained parameters

    Side-effects:
        prints out loss on train and vali data during training
    """
    parameters = experiment.initial_parameters()
    loss_function = experiment.loss_function
    exp_name = experiment.name

    for (i, batch) in enumerate(train_batches):
        loss, d_params = numerical_gradient(loss_function, parameters, batch, pool,
                                            random_projection=experiment.random_projection,
                                            update_indices=experiment.learnable_parameters)


        # === record
        batch_size = batch[0].shape[0]
        # only record some of the points, for memory efficiency
        if i % MEASURE_SKIP == 0:
            vali_loss = loss_function(parameters, vali_data)
            if i % (MEASURE_SKIP*4) == 0:
                print (i + 1)*batch_size, '\t\tVALI:', vali_loss

            loginfo['train_file'].write(exp_name+' '+str((i + 1)*batch_size)+' '+str(loss)+' ' + str(loginfo['rep'])+' ' + loginfo['method']+'\n')
            loginfo['vali_file'].write(exp_name+' '+str((i + 1)*batch_size)+' '+str(vali_loss)+' ' + str(loginfo['rep'])+' ' + loginfo['method']+'\n')

            # flush both files now and again
            loginfo['vali_file'].flush()
            loginfo['train_file'].flush()

        # === update parameters
        parameters = parameters - LEARNING_RATE*d_params
        if experiment.project:
            # use the polar decomposition to re-unitarise the matrix
            parameters = project_to_unitary(parameters, check_unitary=False)

    print 'Training complete!'
    return parameters

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

def logging(d, noise, batch_size, n_batches, start_from_rep):
    logging_path = 'output/simple/fft1_d'+str(d) + '_noise'+str(noise) + '_bn'+str(batch_size) + '_nb' + str(n_batches)
    # save to an R-plottable file because I am so very lazy
    R_vali = open(logging_path+'_vali.txt', 'a')
    R_train = open(logging_path+'_train.txt', 'a')
    R_test = open(logging_path+'_test.txt', 'a')
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
    return R_vali, R_train, R_test

# === main loop === #
def main(d, experiments, n_reps=3, n_epochs=1, noise=0.01, start_from_rep=0)
    """
    Args:
        d
        experiments
        n_reps
        n_epochs
        noise
        start_from_rep          int         initialise rep counter to this
    """
    # OPTIONS
    batch_size = 20
    n_batches = 50000
    if n_epochs is None:
        n_epochs = d
        print 'WARNING: No n_epochs provided, using', n_epochs
  
    # === logging === #
    R_vali, R_train, R_test = logging(d, noise, batch_size, n_batches, start_from_rep)
    loginfo = {'vali_file': R_vali, 'train_file': R_train, 'rep': None, 'method': None}

    # === parallelism === #
    pool = Pool(NUM_WORKERS)

    # === outer rep loop! === #
    for rep in xrange(start_from_rep, start_from_rep + n_reps):
        # randomly select the method to generate test data
        method = sample(['lie_algebra', 'qr', 'composition'], 1)[0]
        if method == 'sparse':
            raise NotImplementedError
            nonzero_index = sample(xrange(d*d), 1)[0]
            method = 'sparse_'+str(nonzero_index)
            sparse_lambdas = np.zeros(shape=(d*d))
            sparse_lambdas[nonzero_index] = 1
            U = unitary_matrix(d, method='lie_algebra', lambdas=sparse_lambdas)
            sparse_test = open(logging_path + '_sparse.txt', 'a')
            sparse_test.write('truth ' + map(lambda x: 'lambda_' + str(x), xrange(d*d)) + '\n')
        else:
            U = unitary_matrix(d, method=method)

        print rep, ': generated U using:', method

        loginfo['method'] = method
        loginfo['rep'] = rep

        # === the data === #
        batches = generate_unitary_learning(U, batch_size, n_batches, n_epochs, noise=noise)

        vali_batch = batches[0]
        test_batch = batches[1]
        train_batches = batches[2:]

        # === baselines === #
        random_test_loss = random_baseline(test_batch, method=method)
        true_test_loss = true_baseline(U, test_batch)
        baselines = {'random_unitary': random_test_loss, 'true': true_test_loss}
        for (name, loss) in baselines:
            R_test.write(name + ' ' + str(loss) + ' ' + str(rep) + ' ' + method +'\n')
        R_test.flush()

        # === run the experiments === #
        for experiment in experiments:
            exp_name = experiment.name
            print 'Running', exp_name, 'experiment!'
            trained_parameters = train_loop(experiment, train, vali, pool, loginfo)
            test_loss = experiment.loss_function(trained_parameters, test_batch)

            # record this experimental result
            experiment.test_loss = test_loss
            R_test.write(exp_name + ' ' + str(test_loss) + ' ' + str(rep) + ' ' + method + '\n')
            R_test.flush()

        # === report at the end of the rep === #
        print '\t\tRep', rep, 'completed. Test losses:'
        for (name, test_loss) in baselines:
            print name, ':', loss
        for experiment in experiments:
            print experiment.name, ':', experiment.test_loss

        # end of rep

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
