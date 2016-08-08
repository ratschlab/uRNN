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

# tf 0.9
#from tf.nn.rnn_cell import RNNCell

# tf 0.7
#from tensorflow.models.rnn import rnn
#from tensorflow.models.rnn.rnn_cell import RNNCell
from tensorflow.python.ops import variable_scope as vs

from unitary import unitary

# === functions to help with implementing the theano version === #
# from http://arxiv.org/abs/1511.06464

def times_diag(arg, state_size, scope=None):
    """
    Multiplication with a diagonal matrix of the form exp(i theta_j)
    """
#    batch_size = arg.get_shape().as_list()[0]
    with vs.variable_scope(scope or "Times_Diag"):
        thetas = vs.get_variable("Thetas", 
                                 initializer=tf.constant(np.random.uniform(low=-np.pi, 
                                                                           high=np.pi, 
                                                                           size=state_size), 
                                                         dtype=tf.float32),
                                 dtype=tf.float32)
        # e(i theta)  = cos(theta) + i sin(theta)
        diagonal = tf.diag(tf.complex(tf.cos(thetas), tf.sin(thetas)))
    return tf.matmul(arg, diagonal)

def reflection(state, state_size, scope=None, theano_reflection=False):
    """
    I do not entirely trust or believe the reflection operator in the theano version. :/
    TODO: indeed, it is wrong, wrongish.
    TODO: do the 'right' version.
    """
    # the reflections are initialised in a weird and tricky way: using initialize_matrix,
    # as if they are columns from a (2, state_size) matrix, so the range of random initialisation
    # is informed by both... but then my fixed_initializer function would return an incorrectly-sized
    # reflection, so I'm just going to do it manually.
    scale = np.sqrt(6.0/ (2 + state_size*2))
    
    with vs.variable_scope(scope or "Reflection"):
        reflect_re = vs.get_variable("Reflection/Real", dtype=tf.float32,
                                     initializer=tf.constant(np.random.uniform(low=-scale, high=scale, size=(state_size)),
                                                             dtype=tf.float32,
                                                             shape=[state_size, 1]))
        reflect_im = vs.get_variable("Reflection/Imaginary", dtype=tf.float32,
                                     initializer=tf.constant(np.random.uniform(low=-scale, high=scale, size=(state_size)),
                                                             dtype=tf.float32,
                                                             shape=[state_size, 1]))

        if theano_reflection:
            # NOTE: I am *directly copying* what they do in the theano code (inasmuch as one can in TF),
            #       (s/input/state/g)
            # even though I think the maths might be incorrect, see this issue: https://github.com/amarshah/complex_RNN/issues/2
            # the function is times_reflection in models.py (not this file, hah hah hah!)
            #
            # Otherwise we could do this 'simply' using complex numbers.

            state_re = tf.real(state)
            state_im = tf.imag(state)
            
            vstarv = tf.reduce_sum(reflect_re**2 + reflect_im**2)

            state_re_reflect_re = tf.matmul(state_re, reflect_re)
            state_re_reflect_im = tf.matmul(state_re, reflect_im)
            state_im_reflect_re = tf.matmul(state_im, reflect_re)
            state_im_reflect_im = tf.matmul(state_im, reflect_im)

            # tf.matmul with transposition is the same as T.outer
            # we need something of the shape [batch_size, state_size] in the end
            a = tf.matmul(state_re_reflect_re - state_im_reflect_im, reflect_re, transpose_b=True)
            b = tf.matmul(state_re_reflect_im + state_im_reflect_re, reflect_im, transpose_b=True)
            c = tf.matmul(state_re_reflect_re - state_im_reflect_im, reflect_im, transpose_b=True)
            d = tf.matmul(state_re_reflect_im + state_im_reflect_re, reflect_re, transpose_b=True)

            # the thing we return is:
            # return_re = state_re - (2/vstarv)(d - c)
            # return_im = state_im - (2/vstarv)(a + b)

            new_state_re = state_re - (2.0 / vstarv) * (d - c)
            new_state_im = state_im - (2.0 / vstarv) * (a + b)
            new_state = tf.complex(new_state_re, new_state_im)
        else:
            vstarv = tf.reduce_sum(reflect_re**2 + reflect_im**2)
            prefactor = tf.complex(2.0/vstarv, 0.0)

            v = tf.complex(reflect_re, reflect_im)
            v_star = tf.conj(v)

            vx = tf.matmul(state, tf.reshape(v_star, [-1, 1]))
            # SO MANY TRANSPOSE
            new_state = state - prefactor *  tf.transpose(tf.matmul(v, tf.transpose(vx)))
    return new_state

def relu_mod(state, scope=None, real=False):
    """
    Rectified linear unit for complex-valued state.
    (Equation 8 in http://arxiv.org/abs/1511.06464)
    """
    state_size = state.get_shape()[1]
    batch_size = state.get_shape()[0]
    with vs.variable_scope(scope or "ReLU_mod"):
        if not real:
            # WARNING: complex_abs has no gradient registered in the docker version for some reason
            # [[ LookupError: No gradient defined for operation 'RNN/complex_RNN_99/ReLU_mod/ComplexAbs' (op type: ComplexAbs) ]]
            #modulus = tf.complex_abs(state)
            modulus = tf.sqrt(tf.real(state)**2 + tf.imag(state)**2)
            bias_term = vs.get_variable("Bias", dtype=tf.float32, 
                                        initializer=tf.constant(np.random.uniform(low=-0.01, high=0.01, size=(state_size)), 
                                                                dtype=tf.float32, 
                                                                shape=[state_size]))
                                        #        bias_tiled = tf.tile(bias_term, [1, batch_size])

            rescale = tf.complex(tf.maximum(modulus + bias_term, 0) / ( modulus + 1e-5*tf.ones_like(modulus)), tf.zeros_like(modulus))
                                        #rescale = tf.complex(tf.maximum(modulus + bias_term, 0) / ( modulus + 1e-5), 0.0)
        else:
            # state is [state_re, state_im]
            modulus = tf.reduce_sum(state**2)
            hidden_size =state_size/2
            bias_re = vs.get_variable("Bias", dtype=tf.float32, 
                                      initializer=tf.constant(np.random.uniform(low=-0.01, high=0.01, size=(hidden_size)), 
                                                              dtype=tf.float32, 
                                                              shape=[hidden_size]))
            bias_term = tf.concat(1, [bias_re, tf.zeros_like(bias_re)])
            rescale = tf.maximum(modulus + bias_term, 0) / ( modulus + 1e-5*tf.ones_like(modulus) )
    return state * rescale

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
    of the resulting matrix) should be initialised to identity, and not uniformly randomly as the rest.
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

    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args:           a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size:    int, second dimension of W[i].
        bias:           boolean, whether to add a bias term or not.
        bias_start:     starting value to initialize the bias; 0 by default.
        scope:          VariableScope for the created subgraph; defaults to "Linear".
        identity:       which matrix corresponding to inputs should be initialised to identity?
        dtype:          data type of linear operators
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
  """
    assert args is not None
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

def linear_complex(arg, output_size, bias, bias_start=0.0, 
           scope=None, identity=-1, dtype=tf.float32):
    """
    NOTE: arg is a single arg, because that's how it is

    Linear map:
                arg * W where W is complex, but really:
            W_re * arg_re - W_im * arg_im + i (W_re * arg_im + W_im * arg_re)
                where W_re, W_im are Variables
    Args:
        arg:            a 2D Tensor (batch x n))))
        output_size:    int, second dimension of W[i].
        bias:           boolean, whether to add a bias term or not.
        bias_start:     starting value to initialize the bias; 0 by default.
        scope:          VariableScope for the created subgraph; defaults to "LinearComplex".
        identity:       which matrix corresponding to inputs should be initialised to identity?
        dtype:          data type of linear operators

    Returns:
        2D tensor with shape [batch x output_size] equal to arg * W
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
  """
    assert arg
    shape = arg.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("LinearComplex is expecting a 2D argument")
    n = shape[1]

    arg_re = tf.real(arg)
    arg_im = tf.imag(arg)

    # Now the computation.
    with vs.variable_scope(scope or "LinearComplex"):
        matrix_re = vs.get_variable("Matrix/Real", dtype=dtype, initializer=fixed_initializer([n], output_size, identity, dtype))
        matrix_im = vs.get_variable("Matrix/Imaginary", dtype=dtype, initializer=fixed_initializer([n], output_size, identity, dtype))
        res_re = tf.matmul(arg_re, matrix_re) - tf.matmul(arg_im, matrix_im)
        res_im = tf.matmul(arg_im, matrix_re) + tf.matmul(arg_re, matrix_im)
        if not bias:
            return tf.complex(res_re, res_im)
        else:
            bias_re = vs.get_variable("Bias/Real", dtype=dtype, initializer=tf.constant(bias_start, dtype=dtype, shape=[output_size]))
            bias_im = vs.get_variable("Bias/Imaginary", dtype=dtype, initializer=tf.constant(bias_start, dtype=dtype, shape=[output_size]))
            return tf.complex(res_re + bias_re, res_im + bias_im)

# === RNNs ! === #

def RNN(cell_type, x, input_size, state_size, output_size, sequence_length):
    batch_size = tf.shape(x)[0]
    if cell_type == 'tanhRNN':
        cell = tanhRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype)
    elif cell_type == 'IRNN':
        cell = IRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype)
    elif cell_type == 'LSTM':
        cell = LSTMCell(input_size=input_size, state_size=2*state_size, output_size=output_size, state_dtype=x.dtype)
    elif cell_type == 'complex_RNN':
        cell = complex_RNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=tf.complex64)
    elif cell_type == 'uRNN':
        #cell = uRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=tf.complex64)
        cell = uRNNCell(input_size=input_size, state_size=2*state_size, output_size=output_size, state_dtype=x.dtype)
    elif cell_type == 'ortho_tanhRNN':
        cell = tanhRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype)
    else: 
        raise NotImplementedError
    state_0 = cell.zero_state(batch_size)
    # split up the input so the RNN can accept it...
    inputs = [tf.squeeze(input_, [1])
            for input_ in tf.split(1, sequence_length, x)]
    # tf 0.7.0
#    outputs, final_state = rnn.rnn(cell, inputs, initial_state=state_0)
    # tf 0.9.0
    outputs, final_state = tf.nn.rnn(cell, inputs, initial_state=state_0)
    return outputs

# === cells ! === #
# TODO: better name for this abstract class
#class steph_RNNCell(RNNCell):      # tf 0.7.0
class steph_RNNCell(tf.nn.rnn_cell.RNNCell):            # tf 0.9.0
    def __init__(self, input_size, state_size, output_size, state_dtype):
        self._input_size = input_size
        self._state_size = state_size
        self._output_size = output_size
        self._state_dtype = state_dtype

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def state_dtype(self):
        return self._state_dtype

    def zero_state(self, batch_size, dtype=None):
        """
        Return state tensor (shape [batch_size x state_size]) filled with 0.

        Args:
            batch_size:     int, float, or unit Tensor representing the batch size.
            dtype:          the data type to use for the state
                            (optional, if None use self.state_dtype)
        Returns:
            A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        if dtype is None:
            dtype = self.state_dtype
        zeros = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype)
        zeros.set_shape([None, self.state_size])
        return zeros

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
            new_state = tf.tanh(linear([inputs], self._state_size, bias=True, scope='Linear/FoldIn') + linear([state], self._state_size, bias=False, scope='Linear/Transition'))
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
        inputs_complex = tf.complex(inputs, tf.constant(0.0, dtype=inputs.dtype))          # tf 0.9.0
        with vs.variable_scope(scope):
            step1 = times_diag(state, self._state_size, scope='Diag/First')
            step2 = tf.batch_fft(step1, name='FFT')
            step3 = reflection(step2, self._state_size, scope='Reflection/First')
            permutation = vs.get_variable("Permutation", dtype=tf.complex64, 
                                          initializer=tf.complex(np.random.permutation(np.eye(self._state_size, dtype=np.float32)), tf.constant(0.0, dtype=tf.float32)),
                                          trainable=False)
            step4 = tf.matmul(step3, permutation)
            step5 = times_diag(step4, self._state_size, scope='Diag/Second')
            step6 = tf.batch_ifft(step5, name='InverseFFT')
            step7 = reflection(step6, self._state_size, scope='Reflection/Second')
            step8 = times_diag(step7, self._state_size, scope='Diag/Third')

            # (folding in the input data) 
            foldin_re = linear(inputs, self._state_size, bias=False, scope='Linear/FoldIn/Real')
            foldin_im = linear(inputs, self._state_size, bias=False, scope='Linear/FoldIn/Imaginary')
            intermediate_state = tf.complex(foldin_re, foldin_im, name='Linear/Intermediate/Complex') + step8
            
            new_state = relu_mod(intermediate_state, scope='ReLU_mod')
#            new_state = intermediate_state

            
            real_state = tf.concat(1, [tf.real(new_state), tf.imag(new_state)])
            output = linear(real_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state

class uRNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='uRNN', split=True, real_valued=False):
        """
        this unitary RNN shall be my one, once I figure it out I guess
        ... fun times ahead

        split produces some different behaviour... if split, real/imag parameters are separate
        """
        # TODO: think about outputs
        inputs_complex = tf.complex(inputs, 0.0)
        with vs.variable_scope(scope):
            if real_valued:
                # this means all the variables etc. are real-valued, we don't need to deal with complex numbers at all
                # (because I am paranoid that TF isn't doing complex gradients as I expect)

                # so, the state will be [real; imaginary]
                raise NotImplementedError

            if split:
                # transform the hidden state
                # these are possibly non-differentiable in tf, need to test :/
   #             state_re = tf.real(state)
#                state_im = tf.imag(state)
#                Ustate = tf.complex(linear(state_re, self._state_size, bias=True, scope='Unitary/Transition/Real'), linear(state_im, self._state_size, bias=True, scope='Unitary/Transition/Imaginary'))
# TODO messing around, making the state real (first and second halfs...)
                hidden_size = self._state_size/2
                state_re = tf.slice(state, [0, 0], [-1, hidden_size])
                state_im = tf.slice(state, [0, hidden_size], [-1, hidden_size])
                Ustate = linear(state_re, self._state_size, bias=True, scope='Unitary/Transition/Real') +  linear(state_im, self._state_size, bias=True, scope='Unitary/Transition/Imaginary')

                # now as in the complex_RNN case
                # (folding in the input data) 
                foldin_re = linear(inputs, self._state_size, bias=False, scope='Linear/FoldIn/Real')
                foldin_im = linear(inputs, self._state_size, bias=False, scope='Linear/FoldIn/Imaginary')

    # TODO messing around
#                intermediate_state = tf.complex(foldin_re, foldin_im, name='Linear/Intermediate/Complex') + Ustate
#                new_state = relu_mod(intermediate_state, scope='ReLU_mod')
                new_state = foldin_re + foldin_im + Ustate          # wat
#                new_state = tf.complex(tf.nn.relu(tf.real(intermediate_state)), tf.nn.relu(tf.imag(intermediate_state)))

                real_state = tf.concat(1, [tf.real(new_state), tf.imag(new_state)])
                output = linear(real_state, self._output_size, bias=True, scope='Linear/Output')
            else:
                raise NotImplementedError
                # probably using sigmoid?
                new_state = tf.nn.sigmoid(linear_complex(inputs_complex, self._state_size, bias=True, scope='Linear/FoldIn') + linear_complex(state, self._state_size, bias=False, scope='Unitary/Transition'), name='Unitary/State')
                output = linear_complex(new_state, self._output_size, bias=True, scope='Linear/Output')
                # for now, output is modulus...
                output = tf.sqrt(tf.real(output)**2 + tf.imag(output)**2)
        return output, new_state
