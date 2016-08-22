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

import test_rnn_internal
from copy import deepcopy

# === constants === #

DO_TEST = False
COMPARE_NUMERICAL_GRADIENT = False

# === fns === #

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
def run_experiment(task, batch_size, state_size, T, model, data_path, 
                  gradient_clipping, learning_rate, num_epochs, timestamp=False):
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
    elif task == 'memory':
        output_size = 9
        loss_type = 'CE'
        assert input_size == 10
    
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
    identifier = model + '_' + str(T)
    if timestamp:
        identifier = identifier + '_' + str(int(time()))
    
    best_model_path = 'output/' + task + '/' + identifier + '.best_model.ckpt'
    best_vali_cost = 1e6
    
    train_cost_trace = []
    vali_cost_trace = []
    trace_path = 'output/' + task + '/' + identifier + '.trace.pk'

    hidden_gradients_path = 'output/' + task + '/' + identifier + '.hidden_gradients.pk' #TODO: internal monitoring

    # === ops for training === #
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
        lambda_file = open('output/' + identifier + '_lambdas.txt', 'w')
        lambda_file.write('batch ' + ' '.join(map(lambda x: 'lambda_' + str(x), xrange(len(lambdas)))) + '\n')
    else:
        # nothing special here, movin' along...
        train_op = update_step(cost, learning_rate, gradient_clipping)
   
    # === for checkpointing the model === #
    saver = tf.train.Saver()        # for checkpointing the model

    # === gpu stuff === #
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25           # only use 25% of available GPUs (we have 4 on pex)

    # === let's do it! === #
    with tf.Session(config=config) as session:
        # summaries
#        merged = tf.merge_all_summaries()
#        train_writer = tf.train.SummaryWriter('./log/' + model, session.graph)
        
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
                if model == 'uRNN' or model == 'ortho_tanhRNN':
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
                    # no eigtrick required, no numerical gradients, all is fine
                    train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                 
                 # TODO ... this section can be retired soon-ish
                if COMPARE_NUMERICAL_GRADIENT:
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
#                train_cost_trace.append(train_cost)

                if np.random.random() < 0.01:
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
                                 'time_steps': T,
                                 'batch_size': batch_size}

                    cPickle.dump(save_vals, file(trace_path, 'wb'),
                                 cPickle.HIGHEST_PROTOCOL)

                    # uRNN specific stuff: save lambdas
                    if model == 'uRNN':
                        lambda_file.write(str(batch_index) + ' ' + ' '.join(map(str, lambdas)) + '\n')


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
                    default='complex_RNN')
parser.add_argument('--data_path', type=str, help='path to dict of ExperimentData objects (if empty, generate data)', 
                    default='')
parser.add_argument('--learning_rate', type=float, help='prefactor of gradient in gradient descent parameter update', 
                    default=0.001)
parser.add_argument('--num_epochs', type=int, help='number of times to run through training data', 
                    default=10)
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
        options['data_path'] = 'input/memory/1470766867_100.pk'
    elif T == 200:
        options['data_path'] = 'input/memory/1470767064_200.pk'
    elif T == 300:
        options['data_path'] = 'input/memory/1470767409_300.pk'
    elif T == 500:
        options['data_path'] = 'input/memory/1470767936_500.pk'
    else:
        options['data_path'] = ''
else:
    raise ValueError(options['task'])

# === suggestions (param values from paper) === #
print 'Suggested state sizes:'
if options['task'] == 'adding':
    print 'tanhRNN:\t80'
    print 'IRNN:\t\t80'
    print 'LSTM:\t\t40'
    print 'complex_RNN:\t128'
    print 'ortho_tanhRHH:\t20, 64'
elif options['task'] == 'memory':
    print 'tanhRNN:\t128'
    print 'IRNN:\t\t128'
    print 'LSTM:\t\t128'
    print 'complex_RNN:\t512'

# === print stuff ===#
print 'Created dictionary of options'
for (key, value) in options.iteritems():
    print key, ':\t', value

# === now run (auto mode) === #
AUTO = True
if AUTO:
    run_experiment(**options)
