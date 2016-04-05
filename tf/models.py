#!/usr/bin/env ipython
#
# Utility functions.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         20/3/16
# ------------------------------------------

import tensorflow as tf
import numpy as np
import pdb

from tensorflow.models.rnn import rnn
# for testing and learning
from tensorflow.models.rnn.rnn_cell import RNNCell, BasicRNNCell
from tensorflow.python.ops import variable_scope as vs

# === functions to help with implementing the theano version === #

def times_diag(arg, state_size, scope=None):
    with vs.variable_scope(scope or "Times_Diag"):
        thetas = vs.get_variable("Thetas", 
                                 initializer=tf.constant(np.random.uniform(low=-np.pi, 
                                                                           high=np.pi, 
                                                                           size=state_size), 
                                                         dtype=tf.complex64),
                                 dtype=tf.complex64)
        # e(i theta)  = cos(theta) + i sin(theta)
        # form the matrix from this
        # i am sorry about all these casts: diag doesn't take complex64
        real_thetas = tf.cast(thetas, tf.float64)
        matrix = tf.cast(tf.diag(tf.cos(real_thetas)), tf.complex64) + \
                 1j*tf.cast(tf.diag(tf.sin(real_thetas)), tf.complex64)
    # 'cast' input to complex
    # TODO: set dtype based on model during placeholder creation
    return tf.matmul(tf.cast(arg, tf.complex64), matrix)

def fft(arg):
    raise NotImplementedError

def reflection(arg, scope=None):
    raise NotImplementedError

def relu_mod(arg, scope=None):
    raise notImplementedError

def fixed_initializer(n_in_list, n_out, identity=-1, dtype=tf.float32):
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

    Finally: identity: what does it do?
    Well, it is possibly useful to initialise the weights associated with the internal state
    to be the identity. (Specifically, this is done in the IRNN case.)
    So if identity is >0, then it specifies which part of n_in_list (corresponding to a segment
    of the resulting matrix) sholud be initialised to identity, and not uniformly randomly as the rest.
    """
    matrix = np.empty(shape=(sum(n_in_list), n_out))
    row_marker = 0
    for (i, n_in) in enumerate(n_in_list):
        if i == identity:
            values = np.identity(n_in)
        else:
            scale = np.sqrt(6.0/ (n_in + n_out))
            values = np.asarray(np.random.uniform(low=-scale, high=scale,
                                                  size=(n_in, n_out)))
        # NOTE: HARDCODED DTYPE
        matrix[row_marker:(row_marker + n_in), :] = values
        row_marker += n_in
    return tf.constant(matrix, dtype=dtype)

# === more generic functions === #
def linear(args, output_size, bias, bias_start=0.0, 
           scope=None, identity=-1, dtype=tf.float32):
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
    n_in_list = []
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]
            n_in_list.append(shape[1])

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable("Matrix", dtype=dtype, initializer=fixed_initializer(n_in_list, output_size, identity, dtype))
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable("Bias", dtype=dtype, initializer=tf.constant(bias_start, dtype=dtype, shape=[output_size]))
    return res + bias_term

def unitary(arg, state_size, scope=None):
    """
    A linear map:
        arg * U, where U = expm(sum_i lambda_i T_i)
    where lambda_i are variables and T_i are constant Tensors given by basis
    U is square, n x n where n == state_size

    Args:
        arg: a 2D Tensor, batch x n (NOTE: only allowing one, for now)
            note: n must be equal to the state_size
        state_size: int, dimension of U, must equal n
    Returns:
        A 2D Tensor with shape [batch x state_size] equal to arg * U
    Raises:
        ValueError if not arg.shape[1] == state_size
    """
    raise NotImplementedError
    # assert statement here to make sure arg is a tensor etc.
    if not arg.get_shape().as_list()[1] == state_size:
        raise ValueError("Unitary expects shape[1] of first argument to be state size.")

    with vs.variable_scope(scope or "Unitary"):
        # SKETCHTOWN 2016
        lambdas = vs.get_variable("Lambdas")     # is "vector" even a legit variable?
        basis = 0# this is gonna be a list of tensors ... where does it come from?
        U = expm(tf.matmul(lambdas, basis))     # expm not implemented :]
        res = tf.matmul(arg, U)
    return res

def RNN(cell_type, x, input_size, state_size, output_size, sequence_length):
    batch_size = tf.shape(x)[0]
    if cell_type == 'tanhRNN':
        cell = tanhRNNCell(input_size=input_size, state_size=state_size, output_size=output_size)
    elif cell_type == 'IRNN':
        cell = IRNNCell(input_size=input_size, state_size=state_size, output_size=output_size)
    elif cell_type == 'LSTM':
        cell = LSTMCell(input_size=input_size, state_size=2*state_size, output_size=output_size)
    elif cell_type == 'complex_RNN':
        cell = complex_RNNCell(input_size=input_size, state_size=state_size, output_size=output_size)
    else: 
        raise NotImplementedError
    state_0 = cell.zero_state(batch_size, x.dtype)
    # split up the input so the RNN can accept it...
    inputs = [tf.squeeze(input_, [1])
            for input_ in tf.split(1, sequence_length, x)]
    outputs, final_state = rnn.rnn(cell, inputs, initial_state=state_0)
    return outputs

# === cells ! === #
# TODO: better name for this abstract class
class steph_RNNCell(RNNCell):
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

    def __call__(self):
        """
        Run this RNN cell on inputs, starting from the given state.
        
        Args:
            inputs: 2D Tensor with shape [batch_size x self.input_size].
            state: 2D Tensor with shape [batch_size x self.state_size].
            scope: VariableScope for the created subgraph; defaults to class name.
       
       Returns:
            A pair containing:
            - Output: A 2D Tensor with shape [batch_size x self.output_size]
            - New state: A 2D Tensor with shape [batch_size x self.state_size].
        """
        raise NotImplementedError("Abstract method")

class tanhRNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='tanhRNN'):
        """ 
        Slightly-less-basic RNN: 
            state = tanh(linear(previous_state, input))
            output = linear(state)
        """
        with vs.variable_scope(scope):
            new_state = tf.tanh(linear([inputs, state], self._state_size, bias=True, scope='Linear/Transition'))
            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state

class IRNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='IRNN'):
        """ 
        Slightly-less-basic RNN: 
            state = relu(linear(previous_state, input))
            output = linear(state)
        ... but the state linear is initialised in a special way!
        """
        with vs.variable_scope(scope):
            # the identity flag says we initialize the part of the transition matrix corresponding to the 1th element of
            # the first input to linear (a.g. [inputs, state], aka 'state') to the identity
            new_state = tf.nn.relu(linear([inputs, state], self._state_size, bias=True, scope='Linear/Transition', identity=1))
            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state

class LSTMCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='LSTM'):
        """
        Inspired by LSTMCell in tensorflow (python/ops/rnn_cell), but modified to
        be consistent with the version in the Theano implementation. (There are small
        differences...)
        """
        # the state is actually composed of both hidden and state parts:
        # (so they're each half the size of the state, which will be doubled elsewhere)
        # that is confusing nomenclature, I realise
        hidden_size = self._state_size/2
        state_prev = tf.slice(state, [0, 0], [-1, hidden_size])
        hidden_prev = tf.slice(state, [0, hidden_size], [-1, hidden_size])

        with vs.variable_scope(scope):
            i = tf.sigmoid(linear([inputs, hidden_prev], hidden_size, bias=True, scope='Linear/Input'))
            candidate = tf.tanh(linear([inputs, hidden_prev], hidden_size, bias=True, scope='Linear/Candidate'))
            forget = tf.sigmoid(linear([inputs, hidden_prev], hidden_size, bias=True, scope='Linear/Forget'))
            
            intermediate_state = i * candidate + forget * state_prev
            
            # out (not the real output, confusingly)
            # NOTE: this differs from the LSTM implementation in TensorFlow
            # in tf, the intermediate_state doesn't contribute
            out = tf.sigmoid(linear([inputs, hidden_prev, intermediate_state], hidden_size, bias=True, scope='Linear/Out'))
       
            intermediate_hidden = out * tf.tanh(intermediate_state)
            
            # now for the 'actual' output
            output = linear([intermediate_hidden], self._output_size, bias=True, scope='Linear/Output')
            
            # the 'state' to be fed back in (to be split up, again!)
            new_state = tf.concat(1, [intermediate_state, intermediate_hidden])
        return output, new_state

class complex_RNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='complex_RNN'):
        """
        (copying their naming conventions, mkay)
        """
        # TODO: set up data types at time of model selection
        # (for now:) cast inputs to complex
        inputs_complex = tf.cast(inputs, tf.complex64)
        # TODO: fft, reflection, relu_mod
        # constant permutation
        permutation = tf.constant(np.random.permutation(self._state_size), dtype=tf.int32)
        with vs.variable_scope(scope):
            step1 = times_diag(state, self._state_size, scope='Diag/First')
      #      step2 = fft(step1)
      #      step3 = reflection(step2, scope='Reflection/First')
      #      step4 = tf.gather(step3, permutation, name='Permutation')
      #      step5 = times_diag(step4, self._state_size, scope='Diag/Second')
      #      step6 = ifft(step5)
      #      step7 = reflection(step6, scope='Reflection/Second')
      #      step8 = times_diag(step7, self._state_size, scope='Diag/Third')

           # intermediate_state = linear(inputs, self._state_size, bias=True, scope='Linear/Intermediate') + step8
            intermediate_state = linear(inputs_complex, self._state_size, bias=True, scope='Linear/Intermediate', dtype=tf.complex64) + step1
#            new_state = relu_mod(intermediate_state, bias=True, scope='ReLU_mod')
#            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
            output = linear(intermediate_state, self._output_size, bias=True, scope='Linear/Output', dtype=tf.complex64)
        #return output, new_state
        return output, state

class uRNN(steph_RNNCell):
    def __call__(self, inputs, state, scope='uRNN'):
        """
        this unitary RNN shall be my one
        ... but before it can exist, I will have to extend TensorFlow to include expm
        ... fun times ahead
        """
        with vs.variable_scope(scope):
            # probably using sigmoid?
            new_state = tf.nn.sigmoid(unitary(state, self._state_size, bias=False, scope='Unitary/Transition') + linear(inputs, self._state_size, bias=True, scope='Linear/Transition'))
            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state
