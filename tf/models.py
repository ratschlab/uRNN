#!/usr/bin/env ipython
#
# Utility functions.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         20/3/16
# ------------------------------------------

import tensorflow as tf
from tensorflow.models.rnn import rnn
# for testing and learning
from tensorflow.models.rnn.rnn_cell import BasicRNNCell

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
    # THIS IS ALL TRASH
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
    accuracy = tf.reduce_min(hidden)
    parameters = [weights, bias]             # list of Tensors
    return cost, accuracy, parameters

def simple_RNN(x, n_hidden=20, batch_size=10, seq_length=15):
    cell = BasicRNNCell(n_hidden)
    state_0 = cell.zero_state(batch_size, tf.float32)
    # split up the input so the RNN can accept it...
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, seq_length, x)]
    outputs, final_state = rnn.rnn(cell, inputs, initial_state=state_0)
    return outputs
