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
from tensorflow.models.rnn.rnn_cell import RNNCell, linear, BasicRNNCell
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs

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

def simple_RNN(x, n_hidden, batch_size, sequence_length):
    # THIS ONLY WORKS FOR ADDING
    #cell = BasicRNNCell(n_hidden)
    cell = tanhRNNCell(input_size=1, state_size=n_hidden, output_size=20)
    state_0 = cell.zero_state(batch_size, x.dtype)
    # split up the input so the RNN can accept it...
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, sequence_length, x)]
    outputs, final_state = rnn.rnn(cell, inputs, initial_state=state_0)
    return outputs

# === cells ! === #
class tanhRNNCell(RNNCell):
    def __init__(self, input_size, state_size, output_size):
        self._input_size = input_size
        self._state_size = state_size
        self._output_size = output_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope='tanhRNN'):
        """ 
        Slightly-less-basic RNN: 
            state = linear(previous_state, input)
            output = linear(state)
        """
        with vs.variable_scope(scope):
            new_state = tanh(linear([inputs, state], self._state_size, bias=True, scope='Linear/Transition'))
            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state

#class IRNN(RNNCell):
