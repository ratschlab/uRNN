#!/usr/bin/env ipython
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

from data import generate_unitary_learning

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

# === utility functions === #
def numerical_gradient(loss_function, parameters, batch, EPSILON=10e-5):
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
    # set up test data
    d = 20
    n_examples = 100
    n_batches = 1000
    U = np.random.normal(size=(20, 20)) + 1j*np.random.normal(size=(20, 20))
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
    else:
        print experiment
        raise NotImplementedError
  
    # train loop!
    trained_parameters = train_loop(train_batches, loss_fn, initial_parameters, vali_data=vali_batch)

    if TEST:
        test_loss = loss_fn(trained_parameters, test_batch)
        print 'TEST:', test_loss

    return trained_parameters
