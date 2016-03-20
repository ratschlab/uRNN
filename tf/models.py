#!/usr/bin/env ipython
#
# Utility functions.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         20/3/16
# ------------------------------------------

import tensorflow as tf

def trivial(inputs, n_input, n_hidden, n_output, experiment):
    """
    This is just a test...
    TODO: fix the 'trivial' cost function, turns out to be somewhat non-trivial
    """
    x, y = inputs
    # parameters
    weights = tf.Variable(tf.random_normal(shape=(n_input, n_output)), name='weights')
    bias = tf.Variable(tf.random_normal(shape=(n_input, n_output)), name='bias')
    # create a nonsensical but valid (hopefully) cost function
    if experiment == 'adding':
        # this means x is rank 3, christ almighty
        hidden = tf.matmul(x[:, :n_input, 0], weights) + bias
    elif experiment == 'memory':
        # x is rank 2, but also it is integer-valued
        # its values are INDICES
        hidden = weights[x[:, n_input]]
    else:
        raise NotImplementedError
    cost = tf.reduce_max(hidden)
    parameters = [weights, bias]             # list of Tensors
    return cost, parameters
