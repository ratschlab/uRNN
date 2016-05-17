#!/usr/bin/env ipython
#
# Outermost experiment-running script.
# (SKELETON EDITION)
# aka Stephanie is still learning TensorFlow edition
# aka refactoring everything forever edition
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         22/3/16
# ------------------------------------------

import tensorflow as tf
import numpy as np
import pdb
import cPickle
import argparse
from time import time

# local imports
from models import RNN
from data import ExperimentData
# YOLO
from unitary_np import lie_algebra_element, lie_algebra_basis_element, numgrad_lambda_update
from scipy.linalg import expm
# DEYOLO
#from unitary_np import numgrad_lambda_update

# === constants === #
N_TRAIN = int(1e5)
N_TEST = int(1e4)
N_VALI = int(1e4)

DO_TEST = False

def get_cost(outputs, y, loss_type='MSE'):
    """
    Either cross-entropy or MSE.
    This will involve some averaging over a batch innit.

    Let's clarify some shapes:
        outputs is a LIST of length input_size,
            each element is a Tensor of shape (batch_size, output_size)
        y is a Tensor of shape (batch_size, output_size)
    """
    if loss_type == 'MSE':
        # mean squared error
        # discount all but the last of the outputs
        output = outputs[-1]
        # now this object is shape batch_size, output_size
        cost = tf.reduce_mean(tf.sub(output, y) ** 2)
    elif loss_type == 'CE':
        # cross entropy
        # (there may be more efficient ways to do this)
        cost = tf.zeros([1])
        for (i, output) in enumerate(outputs):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y[:, i])
            cost = tf.add(cost, tf.reduce_mean(cross_entropy))
        cost = tf.squeeze(tf.div(cost, i + 1))
    else:
        raise NotImplementedError
    return cost

# == some gradient-specific fns == #
def create_optimiser(learning_rate):
    print 'WARNING: RMSProp does not support complex variables!'
    # TODO: add complex support to RMSProp
    # decay and momentum are copied from theano version values
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                    decay=0.9,
                                    momentum=0.0)
    return opt

def get_gradients(opt, cost, clipping=False, variables=None):
    if variables is None:
        gradient_variables = tf.trainable_variables()
    else:
        gradient_variables = variables
    print 'Calculating gradients of cost with respect to Variables:'
    for var in gradient_variables:
        print var.name, var.dtype
    g_and_v = opt.compute_gradients(cost, gradient_variables)
    if clipping:
        g_and_v = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in g_and_v]
    return g_and_v

def update_variables(opt, g_and_v):
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt
    
def assign_variable(v, newv):
    """
    Just assign a single variable.
    """
    assign_opt = v.assign(newv)
    return assign_opt
    
# do everything all at once
def update_step(cost, learning_rate=0.01, clipping=False):
    print 'WARNING: RMSProp does not support complex variables!'
    # TODO: add complex support to RMSProp
    # decay and momentum are copied from theano version values
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                    decay=0.9,
                                    momentum=0.0)
    print 'By the way, the gradients of cost',
    print 'are with respect to the following Variables:'
    for var in tf.trainable_variables():
        print var.name, var.dtype
    g_and_v = opt.compute_gradients(cost, tf.trainable_variables())
    if clipping:
        g_and_v = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in g_and_v]
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt

# == and now for main == #
def main(experiment='adding', batch_size=10, state_size=20, 
         num_epochs=5, T=100, learning_rate=0.001,
         model='tanhRNN', timestamp=False):
    # randomly select experiment
    if np.random.random() < 0.5:
        experiment = 'adding'
    else:
        experiment = 'memory'
    print 'running', experiment, 'experiment with', model
    # === derivative options/values === #
    gradient_clipping = True
    if model == 'complex_RNN':
        gradient_clipping = False
    num_batches = N_TRAIN / batch_size
    identifier = experiment + '_' + model + '_' + str(T)
    if timestamp:
        identifier = identifier + '_' + str(int(time()))
    best_model_path = 'output/' + identifier + '.best_model.ckpt'
    trace_path = 'output/' + identifier + '.trace_path.pk'

    # === create data === #
    train_data = ExperimentData(N_TRAIN, experiment, T)
    vali_data = ExperimentData(N_VALI, experiment, T)
    test_data = ExperimentData(N_TEST, experiment, T)
  
    # === get shapes and constants === #
    sequence_length = train_data.sequence_length
    input_size = train_data.input_size
    # YOLO: finish doing this bit
    if experiment == 'adding':
        output_size = 1
        loss_type = 'MSE'
        assert input_size == 2
    elif experiment == 'memory':
        output_size = 9
        loss_type = 'CE'
        assert input_size == 10

    # === files in which to record === #
    hidden_gradients_file = open('output/' + identifier + '.hidden_grads.txt', 'w')
    hidden_gradients_file.write('batch t\n')
    
    # === construct the graph === #
    # (doesn't actually matter which one we make placeholders out of)
    x, y = train_data.make_placeholders()

    # === model select === #
    outputs = RNN(model, x, input_size, state_size, output_size, sequence_length=sequence_length)

    # === ops and things === #
    cost = get_cost(outputs, y, loss_type)
    opt = create_optimiser(learning_rate)
    if model == 'uRNN':
        # COMMENCE GRADIENT HACKS
        lambdas = np.random.normal(size=(state_size*state_size))
        U_re_name = 'RNN/uRNN/Unitary/Transition/Matrix/Real:0'
        U_im_name = 'RNN/uRNN/Unitary/Transition/Matrix/Imaginary:0'
        nonU_variables = []
        for var in tf.trainable_variables():
            if var.name == U_re_name:
                U_re_variable = var
            elif var.name == U_im_name:
                U_im_variable = var
            else:
                nonU_variables.append(var)
        U_variables = [U_re_variable, U_im_variable]
        # WARNING: dtype
        U_new_re = tf.placeholder(dtype=tf.float32, shape=[state_size, state_size])
        U_new_im = tf.placeholder(dtype=tf.float32, shape=[state_size, state_size])
        # get gradients (alternately: just store indices and separate afterwards)
        g_and_v_nonU = get_gradients(opt, cost, gradient_clipping, nonU_variables)
        g_and_v_U = get_gradients(opt, cost, gradient_clipping, U_variables)
        # now for ops 
        train_op = update_variables(opt, g_and_v_nonU)
        assign_re_op = assign_variable(U_re_variable, U_new_re)
        assign_im_op = assign_variable(U_im_variable, U_new_im)
                    
        # save-specific thing: saving lambdas
        lambda_file = open('output/' + identifier + '_lambdas.txt', 'w')
        lambda_file.write('batch ' + ' '.join(map(lambda x: 'lambda_' + str(x), xrange(len(lambdas)))) + '\n')
    else:
        # nothing special here, movin' along...
        train_op = update_step(cost, learning_rate, gradient_clipping)

    # === for checkpointing the model === #
    saver = tf.train.Saver()

    # === init === #
    train_cost_trace = []
    vali_cost_trace = []
    best_vali_cost = 1e6
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        
        # === train loop === #
        for epoch in xrange(num_epochs):
            for batch_index in xrange(num_batches):
                # definitely scope for fancy iterator but yolo
                batch_x, batch_y = train_data.get_batch(batch_index, batch_size)
           
                if model == 'uRNN':
                    # CONTINUE GRADIENT HACKS
                    # TODO: OPTIMISATION, TESTING
                    # YOLO making sure U is being updated part 1
                    U_re_orig, U_im_orig = session.run(U_variables)
                    # DEYOLO
                    dcost_dU_re, dcost_dU_im = session.run([g_and_v_U[0][0], g_and_v_U[1][0]], {x:batch_x, y:batch_y})
                    U_new_re_array, U_new_im_array, lambdas, dlambdas = numgrad_lambda_update(dcost_dU_re, dcost_dU_im, lambdas)
                    # YOLO checking numerical gradients (EXPENSIVE)
                    basic_cost = session.run(cost, {x: batch_x, y:batch_y})
                    L = lie_algebra_element(state_size, lambdas)
                    numerical_dcost_dlambdas = np.zeros_like(lambdas)
                    EPSILON=1e-6
                    for e in xrange(len(lambdas)):
                        print 100.0*e/len(lambdas)
                        perturbed_L = L + EPSILON*lie_algebra_basis_element(state_size, e, complex_out=True)
                        perturbed_U = expm(perturbed_L)
                        perturbed_U_re = np.real(perturbed_U)
                        perturbed_U_im = np.imag(perturbed_U)
                        perturbed_cost = session.run(cost, {x: batch_x, y: batch_y, U_re_variable: perturbed_U_re, U_im_variable: perturbed_U_im})
                        gradient = (perturbed_cost - basic_cost)/EPSILON
                        print gradient, dlambdas[e]
                        numerical_dcost_dlambdas[e] = gradient
                    # now compare with dlambdas
                    np.mean(dlambdas - numerical_dcost_dlambdas)
                    pdb.set_trace()
                    # DEYOLO
                    train_cost, _, _, _ = session.run([cost, train_op, assign_re_op, assign_im_op], {x: batch_x, y:batch_y, U_new_re: U_new_re_array, U_new_im: U_new_im_array})
                else:
                    train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                train_cost_trace.append(train_cost)
                print epoch, '\t', batch_index, '\t', loss_type + ':', train_cost

                if batch_index % 50 == 0:
                    vali_cost = session.run(cost, {x: vali_data.x, y: vali_data.y})
                    vali_cost_trace.append(vali_cost)
              
                    # save best parameters
                    if vali_cost < best_vali_cost:
                        best_vali_cost = vali_cost
                        save_path = saver.save(session, best_model_path)
                        print epoch, '\t', batch_index, '\t*** VALI', loss_type + ':', vali_cost, '\t('+save_path+')'
                    else:
                        print epoch, '\t', batch_index, '\t    VALI', loss_type + ':', vali_cost

                    # NOTE: format consistent with theano version
                    # TODO: update alongside plotting script
                    save_vals = {'train_loss': train_cost_trace,
                                 'test_loss': vali_cost_trace,
                                 'best_test_loss': best_vali_cost,
                                 'model': model,
                                 'time_steps': T}

                    cPickle.dump(save_vals, file(trace_path, 'wb'),
                                 cPickle.HIGHEST_PROTOCOL)

                    # uRNN specific stuff: save lambdas
                    if model == 'uRNN':
                        lambda_file.write(str(batch_index) + ' ' + ' '.join(map(str, lambdas)) + '\n')

                # occasionally, calculate all the gradients
                if batch_index % 100 == 0:
                    pass
                    #hidden_states = tf.variables(...) # ... figure out which these are, do I need to name them while defining in RNNCell?
                    #gradz = tf.gradients(cost, hidden_states)
                    #grad_vals = session.run(gradz)
                    #grad_mags = ... get their magnitudes, also values
                    #hidden_gradients_file.write(str(batch_index) + ' ' + str('\n' + str(batch_index) + ' ').join(map(str, grad_mags)) + '\n') # haha what is wrong with me

            # shuffle the data at each epoch
            train_data.shuffle()

        print 'Training completed.'
        if DO_TEST:
            print 'Loading best model from', best_model_path
            saver.restore(session, best_model_path)
            test_cost = session.run(cost, {x: test_data.x, y: test_data.y})
            print 'Performance on test set:', test_cost

def runmo(model):
    """ wrapper script because that's how lazy I am """
    main(model=model)
    return True

#if __name__ == "__main__":
#    main()
parser = argparse.ArgumentParser(description="run an experiment")
parser.add_argument("--experiment", type=str, default='adding')
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--state_size", type=int, default=512)
parser.add_argument("--T", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--model", default='orthogonal_RNN')
parser.add_argument("--timestamp", type=bool, default=True)
args = parser.parse_args()
