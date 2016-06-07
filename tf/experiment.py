#!/usr/bin/env ipython
#
# Outermost experiment-running script.
# aka Stephanie is still learning TensorFlow edition
# aka refactoring everything forever edition
# ... PART TWO!
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         22/3/16, 1/6/16
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
from unitary_np import lie_algebra_element, lie_algebra_basis_element, numgrad_lambda_update, eigtrick_lambda_update
from scipy.linalg import expm

# === constants === #

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
def update_step(cost, learning_rate, clipping=False):
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

def get_data(load_path, task, T, ntrain=int(1e5), nvali=int(1e4), ntest=int(1e4)):
    """
    Either load or generate data.
    """
    if load_path == '':
        save_path = 'input/' + task + '/' + str(int(time())) + '_' + str(T) + '.pk'
        print 'No data path provided, generating and saving to', save_path
        # generate it
        train = ExperimentData(ntrain, task, T)
        vali = ExperimentData(nvali, task, T)
        test = ExperimentData(ntest, task, T)
        # save it
        save_dict = {'train': train, 'vali': vali, 'test': test}
        cPickle.dump(save_dict, open(save_path, 'wb'))
    else:
        print 'Loading data from', load_path
        load_dict = cPickle.load(open(load_path, 'rb'))
        train = load_dict['train']
        vali = load_dict['vali']
        test = load_dict['test']
    return train, vali, test

# == and now for main == #
def main(task, batch_size, state_size, T, model, data_path, gradient_clipping,
         num_epochs=5, learning_rate=0.1, timestamp=False):
    print 'running', task, 'task with', model
 
    # === data === #
    train_data, vali_data, test_data = get_data(data_path, task, T)
    num_batches = train_data.N / batch_size
    x, y = train_data.make_placeholders() # (doesn't actually matter which one we make placeholders out of)

    # === set up the model === #
    sequence_length = train_data.sequence_length
    input_size = train_data.input_size
    if task == 'adding':
        output_size = 1
        loss_type = 'MSE'
        assert input_size == 2
    elif task == 'memory':
        output_size = 9
        loss_type = 'CE'
        assert input_size == 10
    
    # p. important line here
    outputs = RNN(model, x, input_size, state_size, output_size, 
                  sequence_length=sequence_length)

    # === logging === #
    identifier = model + '_' + str(T)
    if timestamp:
        identifier = identifier + '_' + str(int(time()))
    
    best_model_path = 'output/' + task + '/' + identifier + '.best_model.ckpt'
    best_vali_cost = 1e6
    
    train_cost_trace = []
    vali_cost_trace = []
    trace_path = 'output/' + identifier + '.trace.pk'

    hidden_gradients_path = 'output/' + identifier + '.hidden_gradients.pk' #TODO: internal monitoring

    # === ops for training === #
    cost = get_cost(outputs, y, loss_type)
    if model in {'ortho_tanhRNN', 'uRNN'}:
        # COMMENCE GRADIENT HACKS
        opt = create_optimiser(learning_rate)
        nonU_variables = []
        if model == 'uRNN':
            lambdas = np.random.normal(size=(state_size*state_size))
            U_re_name = 'RNN/uRNN/Unitary/Transition/Real/Matrix:0'
            U_im_name = 'RNN/uRNN/Unitary/Transition/Imaginary/Matrix:0'
            for var in tf.trainable_variables():
                print var.name
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
            # ops
            assign_re_op = assign_variable(U_re_variable, U_new_re)
            assign_im_op = assign_variable(U_im_variable, U_new_im)
        else:
            lambdas = np.random.normal(size=(state_size*(state_size-1)/2))
            # TODO: check this name
            U_name = 'RNN/tanhRNN/Linear/Transition/Matrix:0'
            for var in tf.trainable_variables():
                if var.name == U_name:
                    U_variable = var
                else:
                    nonU_variables.append(var)
            U_variables = [U_variable]
            U_new = tf.placeholder(dtype=tf.float32, shape=[state_size, state_size])
            # ops
            assign_op = assign_variable(U_variable, U_new)
        # get gradients (alternately: just store indices and separate afterwards)
        g_and_v_nonU = get_gradients(opt, cost, gradient_clipping, nonU_variables)
        g_and_v_U = get_gradients(opt, cost, gradient_clipping, U_variables)
        # normal train op
        train_op = update_variables(opt, g_and_v_nonU)
                    
        # save-specific thing: saving lambdas
        lambda_file = open('output/' + identifier + '_lambdas.txt', 'w')
        lambda_file.write('batch ' + ' '.join(map(lambda x: 'lambda_' + str(x), xrange(len(lambdas)))) + '\n')
    else:
        # nothing special here, movin' along...
        train_op = update_step(cost, learning_rate, gradient_clipping)

   
    # === for checkpointing the model === #
    saver = tf.train.Saver()        # for checkpointing the model

    # === let's do it! === #
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        
        # === train loop === #
        for epoch in xrange(num_epochs):
            # shuffle the data at each epoch
            train_data.shuffle()
            for batch_index in xrange(num_batches):
                # definitely scope for fancy iterator but yolo
                batch_x, batch_y = train_data.get_batch(batch_index, batch_size)
         
                # === gradient hacks etc. === #
                # TODO: speed-profiling
                if model == 'uRNN':
                    # extract dcost/dU terms from tf
                    dcost_dU_re, dcost_dU_im = session.run([g_and_v_U[0][0], g_and_v_U[1][0]], {x:batch_x, y:batch_y})
                    # calculate gradients of lambdas using eigenvalue decomposition trick
                    U_new_re_array, U_new_im_array, dlambdas = eigtrick_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, speedy=True)
                    # calculate train cost, update variables
                    train_cost, _, _, _ = session.run([cost, train_op, assign_re_op, assign_im_op], {x: batch_x, y:batch_y, U_new_re: U_new_re_array, U_new_im: U_new_im_array})
                elif model == 'ortho_tanhRNN':
                    # extract dcost/dU terms from tf
                    dcost_dU_re = session.run(g_and_v_U[0][0], {x:batch_x, y:batch_y})
                    dcost_dU_im = np.zeros_like(dcost_dU_re)
                    # calculate gradients of lambdas using eigenvalue decomposition trick
                    U_new_re_array, U_new_im_array, dlambdas = eigtrick_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, speedy=True)
                    assert np.array_equal(U_new_im_array, np.zeros_like(U_new_im_array))
                    # calculate train cost, update variables
                    train_cost, _, _ = session.run([cost, train_op, assign_op], {x: batch_x, y:batch_y, U_new: U_new_re_array})
                else:
                    train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})

                # TEST
                COMPARE_NUMERICAL_GRADIENT = False
                if COMPARE_NUMERICAL_GRADIENT:
                    # checking numerical gradients (EXPENSIVE)
                    # TODO: figure out why these are different :/
                    basic_cost = session.run(cost, {x: batch_x, y:batch_y})
                    L = lie_algebra_element(state_size, lambdas)
                    numerical_dcost_dlambdas = np.zeros_like(lambdas)
                    EPSILON=1e-5
                    if model == 'uRNN':
                        T_indices = xrange(len(lambdas))
                    else:
                        T_indices = []
                        for i in xrange(0, state_size):
                            for j in xrange(0, i):
                                T_indices.append(state_size*i + j)
                    lambda_index = 0
                    for e in T_indices:
                        print 100.0*e/len(lambdas)
                        perturbed_L = L + EPSILON*lie_algebra_basis_element(state_size, e, complex_out=True)
                        perturbed_U = expm(perturbed_L)
                        if model == 'uRNN':
                            perturbed_U_re = np.real(perturbed_U)
                            perturbed_U_im = np.imag(perturbed_U)
                            perturbed_cost = session.run(cost, {x: batch_x, y: batch_y, U_re_variable: perturbed_U_re, U_im_variable: perturbed_U_im})
                        else:
                            perturbed_cost = session.run(cost, {x: batch_x, y: batch_y, U_variable: perturbed_U})
                        gradient = (perturbed_cost - basic_cost)/EPSILON
                        print gradient, dlambdas[lambda_index]
                        numerical_dcost_dlambdas[lambda_index] = gradient
                        lambda_index += 1
                    # now compare with dlambdas
                    np.mean(dlambdas - numerical_dcost_dlambdas)
                # DETEST

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
                                 'vali_loss': vali_cost_trace,
                                 'best_vali_loss': best_vali_cost,
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


# === parse inputs === #
parser = argparse.ArgumentParser(description='Run task of long-term memory capacity of RNN.')
parser.add_argument('--task', type=str, help='which task? adding/memory', default='adding')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--state_size', type=int, help='size of internal state', default=40)
parser.add_argument('--T', type=int, help='either memory time-scale or addition input length', default=100)
parser.add_argument('--model', type=str, help='which RNN model to use?', default='tanhRNN')
parser.add_argument('--data_path', type=str, help='path to dict of ExperimentData objects (if empty, generate data)', default='')
options = vars(parser.parse_args())

# === derivative options === #
if options['model'] in {'complex_RNN', 'ortho_tanhRNN', 'uRNN'}:
    options['gradient_clipping'] = False
else:
    options['gradient_clipping'] = True

if options['data_path'] == '':
    if options['task'] == 'adding':
        pass
    elif options['task'] == 'memory':
        pass

