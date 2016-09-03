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

from copy import deepcopy
import cProfile

# === some bools === #

DO_TEST = False
COMPARE_NUMERICAL_GRADIENT = False
SAVE_INTERNAL_GRADS = False

# === fns === #

def save_options(options):
    """ so I can stop forgetting what learning rate I used... """
    if options['identifier']:
        mname = options['identifier'] + '_' + options['model'] + '_T' + str(options['T']) + '_n' + str(options['state_size'])
    else:
        mname = options['model'] + '_T' + str(options['T']) + '_n' + str(options['state_size'])
    options_path = 'output/' + options['task'] + '/' + mname + '.options.txt'
    print 'Saving run options to', options_path
    options_file = open(options_path, 'w')
    for (key, value) in options.iteritems():
        options_file.write(key + ' ' + str(value) + '\n')
    options_file.close()
    return True

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
        # now this object is shape batch_size, output_size (= 1, it should be)
        cost = tf.reduce_mean(tf.sub(output, y) ** 2, 0)[0]
    elif loss_type == 'CE':
        # cross entropy
        # (there may be more efficient ways to do this)
        cost = tf.zeros([1])
        for (i, output) in enumerate(outputs):
            # maybe this is wrong
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y[:, i])
            cost = tf.add(cost, tf.reduce_mean(cross_entropy))
        cost = tf.squeeze(tf.div(cost, i + 1))
    elif loss_type == 'mnist':
        # just use the last output (this is a column for the whole batch, remember)
        output = outputs[-1]
        # mean categorical cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y)
        cost = tf.reduce_mean(cross_entropy)
    else:
        raise NotImplementedError
#    tf.scalar_summary('cost', cost)
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
    g_and_v = opt.compute_gradients(cost, gradient_variables)
    print 'Calculating gradients of cost with respect to Variables:'
    for (g, v) in g_and_v:
        print v.name, v.dtype, v.get_shape()
#        if not v is None and not g is None:
#            tf.histogram_summary(v.name + 'grad', g)
#            tf.histogram_summary(v.name, v)
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
    g_and_v = opt.compute_gradients(cost, tf.trainable_variables())
    print 'By the way, the gradients of cost',
    print 'are with respect to the following Variables:'
    for (g, v) in g_and_v:
        print v.name, v.dtype, v.get_shape()
        if not v is None and not g is None:
            tf.histogram_summary(v.name + 'grad', g)
            tf.histogram_summary(v.name, v)
    if clipping:
        g_and_v = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in g_and_v]
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt

def get_data(load_path, task, T, ntrain=int(1e6), nvali=int(1e4), ntest=int(1e4)):
    """
    Either load or generate data.
    """
    if load_path == '':
        print 'No data path provided...'
        # generate it
        if task == 'mnist':
            train = ExperimentData(ntrain, 'mnist_train', T)
            vali = ExperimentData(nvali, 'mnist_vali', T)
            test = ExperimentData(ntest, 'mnist_test', T)
            save_path = 'input/' + task + '/mnist.pk'
        elif task == 'mnist_perm':
            train = ExperimentData(ntrain, 'mnist_train', T, mnist_perm=True)
            vali = ExperimentData(nvali, 'mnist_vali', T, mnist_perm=True)
            test = ExperimentData(ntest, 'mnist_test', T, mnist_perm=True)
            save_path = 'input/' + task + '/mnist_perm.pk'
        else:
            train = ExperimentData(ntrain, task, T)
            vali = ExperimentData(nvali, task, T)
            test = ExperimentData(ntest, task, T)
            save_path = 'input/' + task + '/' + str(int(time())) + '_' + str(T) + '.pk'
        print '...generating and saving to', save_path
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
def run_experiment(task, batch_size, state_size, T, model, data_path, 
                  gradient_clipping, learning_rate, num_epochs, identifier, 
                  verbose):
    print 'running', task, 'task with', model, 'and state size', state_size
 
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
        assert sequence_length == T
    elif task == 'memory':
        output_size = 9
        loss_type = 'CE'
        assert input_size == 10
        assert sequence_length == T + 20
    elif 'mnist' in task:
        output_size = 10
        loss_type = 'mnist'
        assert input_size == 1
        assert sequence_length == 28*28

    if verbose: print 'setting up RNN...'
    if model == 'uRNN':
        # generate initial lambdas
        lambdas = np.random.normal(size=(state_size*state_size))
        # transpose because that's how it goes in the RNN
        Uinit = expm(lie_algebra_element(state_size, lambdas)).T
        Uinit_re = np.real(Uinit)
        Uinit_im = np.imag(Uinit)
        # now create the RNN
        outputs = RNN(model, x, input_size, state_size, output_size, 
                      sequence_length=sequence_length,
                      init_re=Uinit_re, init_im=Uinit_im)
    else:
        outputs = RNN(model, x, input_size, state_size, output_size, 
                      sequence_length=sequence_length)

    # === logging === #
    if identifier:
        mname = identifier + '_' + model + '_T' + str(T) + '_n' + str(state_size)
    else:
        mname = model + '_T' + str(T) + '_n' + str(state_size)
  
    # update options with path...?# 
    options_path = 'output/' + options['task'] + '/' + mname + '.options.txt'
    
    best_model_path = 'output/' + task + '/' + mname + '.best_model.ckpt'
    best_vali_cost = 1e6
    
    vali_trace_path = 'output/' + task + '/' + mname + '.vali.txt'
    vali_trace_file = open(vali_trace_path, 'w')
    vali_trace_file.write('epoch batch vali_cost\n')
    train_trace_path = 'output/' + task + '/' + mname + '.train.txt'
    train_trace_file = open(train_trace_path, 'w')
    train_trace_file.write('epoch batch train_cost\n')
    if 'mnist' in task:
        vali_acc_trace_path = 'output/' + task + '/' + mname + '.vali_acc.txt'
        vali_acc_trace_file = open(vali_acc_trace_path, 'w')
        vali_acc_trace_file.write('epoch batch vali_acc_cost\n')

    if SAVE_INTERNAL_GRADS:
        hidden_gradients_path = 'output/' + task + '/' + mname + '.hidden_gradients.txt'
        hidden_gradients_file = open(hidden_gradients_path, 'w')
        hidden_gradients_file.write('batch ' + ' '.join(map(str, xrange(train_data.sequence_length))) + '\n')

    # === ops for training === #
    if verbose: print 'setting up train ops...'
    cost = get_cost(outputs, y, loss_type)
    if model in {'ortho_tanhRNN', 'uRNN'}:
        # COMMENCE GRADIENT HACKS
        opt = create_optimiser(learning_rate)
        nonU_variables = []
        if model == 'uRNN':
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
        lambda_file = open('output/' + task + '/' + mname + '_lambdas.txt', 'w')
        lambda_file.write('batch ' + ' '.join(map(lambda x: 'lambda_' + str(x), xrange(len(lambdas)))) + '\n')
    else:
        train_op = update_step(cost, learning_rate, gradient_clipping)
   
    # === for checkpointing the model === #
    saver = tf.train.Saver()        # for checkpointing the model

    # === gpu stuff === #
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25

    # === let's do it! === #
    if verbose: print 'initialising session...'
    with tf.Session(config=config) as session:
        # summaries
#        merged = tf.merge_all_summaries()
#        train_writer = tf.train.SummaryWriter('./log/' + model, session.graph)
        
        if verbose: print 'initialising variables...'
        session.run(tf.initialize_all_variables())

        # === get relevant operations for calculating internal gradient norms === #
        if SAVE_INTERNAL_GRADS:
            graph_ops = session.graph.get_operations()
            internal_grads = [None]*train_data.sequence_length
            internal_norms = np.zeros(shape=train_data.sequence_length)
            o_counter = 0
            for o in graph_ops:
                if 'new_state' in o.name and not 'grad' in o.name:
                    # internal state
                    internal_grads[o_counter] = tf.gradients(cost, o.values()[0])[0]
                    o_counter += 1
            pdb.set_trace()
            assert o_counter == train_data.sequence_length

        # === train loop === #
        if verbose: print 'preparing to train!'
        for epoch in xrange(num_epochs):
            # shuffle the data at each epoch
            if verbose: print 'shuffling training data at epoch', epoch
            train_data.shuffle()
            for batch_index in xrange(num_batches):
                # definitely scope for fancy iterator but yolo
                batch_x, batch_y = train_data.get_batch(batch_index, batch_size)
         
                # === gradient hacks etc. === #
                # TODO: speed-profiling
                if model == 'uRNN' or model == 'ortho_tanhRNN':
                    if verbose: print 'preparing for gradient hacks'
                    # we can use the eigtrick, lambdas is defined...
                    if model == 'uRNN':
                        # extract dcost/dU terms from tf
                        dcost_dU_re, dcost_dU_im = session.run([g_and_v_U[0][0], g_and_v_U[1][0]], {x:batch_x, y:batch_y})
                        # calculate gradients of lambdas using eigenvalue decomposition trick
                        U_new_re_array, U_new_im_array, dlambdas = eigtrick_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, learning_rate, speedy=True)

                        train_cost = session.run(cost, {x:batch_x, y:batch_y})
                        _, _, _ = session.run([train_op, assign_re_op, assign_im_op], {x: batch_x, y:batch_y, U_new_re: U_new_re_array, U_new_im: U_new_im_array})
                    else:
                        #model == 'ortho_tanhRNN':
                        # extract dcost/dU terms from tf
                        dcost_dU_re = session.run(g_and_v_U[0][0], {x:batch_x, y:batch_y})
                        dcost_dU_im = np.zeros_like(dcost_dU_re)
                        # calculate gradients of lambdas using eigenvalue decomposition trick
                        U_new_re_array, U_new_im_array, dlambdas = eigtrick_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, learning_rate, speedy=True)
                        assert np.array_equal(U_new_im_array, np.zeros_like(U_new_im_array))
                        # calculate train cost, update variables
#                        train_cost, _, _, summary = session.run([cost, train_op, assign_op, merged], {x: batch_x, y:batch_y, U_new: U_new_re_array})
                        train_cost, _, _ = session.run([cost, train_op, assign_op], {x: batch_x, y:batch_y, U_new: U_new_re_array})
               
                else:
                    if verbose: print 'calculating cost and updating parameters...'
                    # no eigtrick required, no numerical gradients, all is fine
                    train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                
                 # TODO ... this section can be retired soon-ish
                if COMPARE_NUMERICAL_GRADIENT:
                    print 'comparing numerical and tensorflow gradients...'
                    EPSILON=1e-4
                    basic_cost = session.run(cost, {x: batch_x, y:batch_y})
                    # check the normal TF gradients regardless
                    #Gradvar_name = 'RNN/complex_RNN/Reflection/Second/Reflection/Real:0'
                    Gradvar_name = ''
                    for v in tf.trainable_variables():
                        if v.name == Gradvar_name:
                            Gradvar = v
                            break
                    else:
                        # uh oh
                        pick = np.random.choice(len(tf.trainable_variables()))
                        Gradvar = tf.trainable_variables()[pick]
                        print 'WARNING: Gradvar with name', Gradvar_name, 'not found in trainable variables, selecting', Gradvar.name, 'instead.'
                    # interested in this specific array, "Gradvar"
                    dcost_dGradvar_tf = session.run(tf.gradients(cost, Gradvar), {x:batch_x, y:batch_y})[0]
                    Gradvar_array = session.run(Gradvar, {x:batch_x, y:batch_y})
                    dcost_dGradvar_num = np.zeros_like(dcost_dGradvar_tf)
                    perturbed_Gradvar = deepcopy(Gradvar_array)
                    if len(dcost_dGradvar_tf.shape) == 2:
                        # go over the shape of Gradvar
                        imax, jmax = dcost_dGradvar_tf.shape
                        for i in xrange(imax):
                            for j in xrange(jmax):
                                perturbed_Gradvar[i, j] += EPSILON
                                # recalculate the cost
                                perturbed_cost = session.run(cost, {x:batch_x, y:batch_y, Gradvar: perturbed_Gradvar})
                                gradient = (perturbed_cost - basic_cost)/EPSILON
                                dcost_dGradvar_num[i, j] = gradient
                                # deperturb
                                perturbed_Gradvar[i, j] -= EPSILON
                    else:
                        # assume 1D
                        imax = len(dcost_dGradvar_tf)
                        for i in xrange(imax):
                            perturbed_Gradvar[i] += EPSILON
                            perturbed_cost = session.run(cost, {x:batch_x, y:batch_y, Gradvar: perturbed_Gradvar})
                            gradient = (perturbed_cost - basic_cost)/EPSILON
                            dcost_dGradvar_num[i] = gradient
                            # deperturb
                            perturbed_Gradvar[i] -= EPSILON
                    # now compare
                    print np.mean(np.abs(dcost_dGradvar_tf - dcost_dGradvar_num))
                    pdb.set_trace()
                    # NOW check the eigtrick lambda updates part (if relevant)
                    if model == 'uRNN' or 'orthogonal' in model:
                        L = lie_algebra_element(state_size, lambdas)
                        numerical_dcost_dlambdas = np.zeros_like(lambdas)
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
                            perturbed_U = expm(perturbed_L).T
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
                        print np.mean(np.abs(dlambdas - numerical_dcost_dlambdas))

          
                # TODO OFF FOR NOW
#                train_writer.add_summary(summary, batch_index)

                if batch_index % 150 == 0:
                    print epoch, '\t', batch_index, '\t', loss_type + ':', train_cost
                    vali_cost = session.run(cost, {x: vali_data.x, y: vali_data.y})

                    train_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(train_cost) + '\n')
                    vali_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(vali_cost) + '\n')
                    train_trace_file.flush()
                    vali_trace_file.flush()
              
                    # save best parameters
                    if vali_cost < best_vali_cost:
                        best_vali_cost = vali_cost
                        save_path = saver.save(session, best_model_path)
                        print epoch, '\t', batch_index, '\t*** VALI', loss_type + ':', vali_cost, '\t('+save_path+')'
                    else:
                        print epoch, '\t', batch_index, '\t    VALI', loss_type + ':', vali_cost

                    if 'mnist' in task:
                        # get preds
                        last_outs = session.run(outputs[-1], {x: vali_data.x, y:vali_data.y})
                        class_predictions = np.argmax(np.exp(last_outs)/np.sum(np.exp(last_outs), axis=1).reshape(6000, -1), axis=1)
                        vali_acc = 100 * np.mean(class_predictions == vali_data.y)
                        vali_acc_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(vali_acc) + '\n')
                        vali_acc_trace_file.flush()
                        print epoch, '\t', batch_index, '\t    VALI ACC:', vali_acc

#                if batch_index % 500 == 0:
#                    if verbose: print 'saving traces to', trace_path
#                    save_vals = {'train_loss': train_cost_trace,
#                                 'vali_loss': vali_cost_trace,
#                                 'vali_acc': vali_acc_trace,
#                                 'best_vali_loss': best_vali_cost,
#                                 'model': model,
#                                 'time_steps': T,
#                                 'batch_size': batch_size}

#                    cPickle.dump(save_vals, file(trace_path, 'wb'),
#                                 cPickle.HIGHEST_PROTOCOL)
#                    if verbose: print 'finished saving!'

                if batch_index % 500 and model == 'uRNN':
                    lambda_file.write(str(batch_index) + ' ' + ' '.join(map(str, lambdas)) + '\n')

                # calculate gradients of cost with respect to internal states
                # save the mean (over the batch) norms of these
                if SAVE_INTERNAL_GRADS and (batch_index == 0 or batch_index == 100):
                    print 'calculating internal gradients...'
                    internal_grads_np = session.run(internal_grads, {x:batch_x, y:batch_y})
                    # get norms of each gradient vector, then average over the batch
                    for (k, grad_at_k) in enumerate(internal_grads_np):
                        norm_at_k = np.mean(np.linalg.norm(grad_at_k, axis=1))
                        internal_norms[k] = norm_at_k
                    hidden_gradients_file.write(str(batch_index) + ' ' + ' '.join(map(str, internal_norms)) + '\n')
                    hidden_gradients_file.flush()

        print 'Training completed.'
        if DO_TEST:
            print 'Loading best model from', best_model_path
            saver.restore(session, best_model_path)
            test_cost = session.run(cost, {x: test_data.x, y: test_data.y})
            print 'Performance on test set:', test_cost

# === parse inputs === #
parser = argparse.ArgumentParser(description='Run task of long-term memory capacity of RNN.')
parser.add_argument('--task', type=str, help='which task? adding/memory', 
                    default='adding')
parser.add_argument('--batch_size', type=int, 
                    default=20)
parser.add_argument('--state_size', type=int, help='size of internal state', 
                    default=5)
parser.add_argument('--T', type=int, help='memory time-scale or addition input length', 
                    default=100)
parser.add_argument('--model', type=str, help='which RNN model to use?', 
                    default='uRNN')
parser.add_argument('--data_path', type=str, help='path to dict of ExperimentData objects (if empty, generate data)', 
                    default='')
parser.add_argument('--learning_rate', type=float, help='prefactor of gradient in gradient descent parameter update', 
                    default=0.001)
parser.add_argument('--num_epochs', type=int, help='number of times to run through training data', 
                    default=10)
parser.add_argument('--identifier', type=str, help='a string to identify the experiment',
                    default='')
parser.add_argument('--verbose', type=bool, help='verbosity?',
                    default=False)
options = vars(parser.parse_args())

# === derivative options === #
if options['model'] in {'complex_RNN', 'ortho_tanhRNN', 'uRNN'}:
    options['gradient_clipping'] = False
else:
    options['gradient_clipping'] = True

# --- load pre-calculated data --- #
T = options['T']
if options['task'] == 'adding':
    if T == 100:
        options['data_path'] = 'input/adding/1470744790_100.pk'
        #options['data_path'] = ''
    elif T == 200:
        options['data_path'] = 'input/adding/1470744860_200.pk'
    elif T == 400:
        options['data_path'] = 'input/adding/1470744994_400.pk'
    elif T == 750:
        options['data_path'] = 'input/adding/1470745056_750.pk'
    else:
        options['data_path'] = ''
elif options['task'] == 'memory':
    if T == 100:
        options['data_path'] = 'input/memory/1472550931_100.pk'
    elif T == 200:
        options['data_path'] = ''
    elif T == 300:
        options['data_path'] = ''
    elif T == 500:
        options['data_path'] = ''
    else:
        options['data_path'] = ''
elif options['task'] == 'mnist':
    options['data_path'] = 'input/mnist/mnist.pk'    # (T is meaningless here...)
elif options['task'] == 'mnist_perm':
    options['data_path'] = 'input/mnist_perm/mnist_perm.pk'
else:
    raise ValueError(options['task'])

# === suggestions (param values from paper) === #
print 'Suggested state sizes:'
if options['task'] == 'adding' or options['task'] == 'mnist':
    print 'tanhRNN:\t80'
    print 'IRNN:\t\t80'
    print 'LSTM:\t\t40'
    print 'complex_RNN:\t128'
    print 'ortho_tanhRHH:\t20, 64'
    print 'uRNN:\t\t30'
elif options['task'] == 'memory':
    print 'tanhRNN:\t128'
    print 'IRNN:\t\t128'
    print 'LSTM:\t\t128'
    print 'complex_RNN:\t512'
    print 'uRNN:\t60'

# === print stuff ===#
print 'Created dictionary of options'
for (key, value) in options.iteritems():
    print key, ':\t', value

# === now run (auto mode) === #
AUTO = True
if AUTO:
    save_options(options)
    run_experiment(**options)
