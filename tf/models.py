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

def fixed_initializer(n_in_list, n_out, dtype):
    """
    This is a bit of a contrived initialiser to be consistent with the
    'initialize_matrix' function in models.py from the complex_RNN repo

    Basically, n_in is a list of input dimensions, because our linear map is
    folding together a bunch of linear maps, like:
        h = Ax + By + Cz + ...
    where x, y, z etc. might be different sizes
    so n_in is a list of [A.shape[0], B.shape[0], ...] in this example.
    and A.shape[1] == B.shape[1] == ...

    The resulting linear operator will have shape:
        ( sum(n_in), n_out )
    (because then one applies it to [x y z] etc.)

    The trick comes into it because we need A, B, C etc. to have initialisations
    which depend on their specific dimensions... their entries are sampled uniformly from
        sqrt(6/(in + out))
    
    So we have to initialise our special linear operator to have samples from different
    uniform distributions. Sort of gross, right? But it'll be fine.
    """
    matrix = np.empty(shape=(sum(n_in_list), n_out), dtype=dtype)
    row_marker = 0
    for n_in in n_in_list:
        scale = np.sqrt(6.0/ (n_in + n_out))
        values = np.asarray(np.random.uniform(low=-scale, high=scale,
                                              size=(n_in, n_out)),
                                              dtype=dtype)
        matrix[row_marker:(row_marker + n_in), :] = values
    # uxaux
    return tf.constant(values)

def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """
    variant of linear from tensorflow/python/ops/rnn_cell
    ... variant so I can specify the initialiser!

    Original docstring:
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  assert args
  if not isinstance(args, (list, tuple)):
      args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
      if len(shape) != 2:
          raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
        total_arg_size += shape[1]

    # prep for my initialiser
    n_in_list = [a.get_shape().as_list()[1] for a in args]
  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
      matrix = vs.get_variable("Matrix", [total_arg_size, output_size], initializer=fixed_initialiser(n_in_list, output_size, dtype))
    if len(args) == 1:
        res = math_ops.matmul(args[0], matrix)
    else:
        res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
        return res
    bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))
    return res + bias_term

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
