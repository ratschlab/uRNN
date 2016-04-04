#!/usr/bin/env ipython
#
# Outermost experiment-running script.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         20/3/16
# ------------------------------------------

import tensorflow as tf
import numpy as np

import models
import utils
import data

# --- constants --- #
N_TRAIN = int(1e5)
N_TEST = int(1e4)
N_VALI = int(1e4)

# --- hyperparameters and options --- #
options = utils.ExperimentOptions()

# --- experiment-specific things --- #
if options.experiment == 'adding':
    train_x, train_y = data.generate_adding(options.T, N_TRAIN)
    vali_x, vali_y = data.generate_adding(options.T, N_VALI)
    test_x, test_y = data.generate_adding(options.T, N_TEST)
    xy_dtype = tf.float32
elif options.experiment == 'memory':
    train_x, train_y = data.generate_memory(options.T, N_TRAIN)
    vali_x, vali_y = data.generate_memory(options.T, N_VALI)
    test_x, test_y = data.generate_memory(options.T, N_TEST)
    xy_dtype = tf.int32
else:
    raise NotImplementedError

# --- training data placeholders --- #
# shapes (disregard batch size)
x_shape = list(train_x.shape[1:])
y_shape = list(train_y.shape[1:])
x = tf.placeholder(xy_dtype, shape=[None] + x_shape)
y = tf.placeholder(xy_dtype, shape=[None] + y_shape)
inputs = [x, y]

# --- get cost and parameters --- #
if options.model == 'trivial':
    cost, accuracy, parameters = models.trivial(inputs, 
                                                options.n_input, 
                                                options.n_hidden, 
                                                options.n_output,
                                                options.experiment)
else:
    raise NotImplementedError

# --- fold in optimizer --- #
opt = tf.train.RMSPropOptimizer(options.learning_rate, 
                                options.decay, 
                                options.momentum, 
                                options.epsilon)
grads_and_vars = opt.compute_gradients(cost, parameters)
if options.clipping:
    clipped_grads_and_vars = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in grads_and_vars]
    grads_and_vars = clipped_grads_and_vars
train_step = opt.apply_gradients(grads_and_vars)

# --- init/saving --- #
init_op = tf.initialize_all_variables()
#saver = tf.train.Saver()

# --- training loop --- #
train_loss_trace = []
vali_loss_trace = []
vali_acc_trace = []
best_params = parameters        # TODO: get arrays
best_rms = rmsprop              # TODO: can we get the rmsprop vals?
best_vali_loss = 1e6

for i in xrange(options.n_iter):
    batch_index = i % int(N_TRAIN/options.batch_size)
    if (batch_index == 0):
        # if we've been through the whole training data already, shuffle
        # NOTE: this deviates from the theano code, but I think they have a bug
        print 'shuffling training data'
        shuffle_indices = np.random.permutation(N_TRAIN)
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]
    
    batch_x = train_x[options.batch_size * batch_index : options.batch_size * (batch_index + 1)]
    batch_y = train_x[options.batch_size * batch_index : options.batch_size * (batch_index + 1)]

    train_loss = train_step.run(feed_dict={x: batch_x, y: batch_y})

    if (i % 50 == 0):
        vali_loss, vali_acc = test_step.run(feed_dict={x: vali_x, y: vali_y})
        print
        print "TEST"
        print
        print "cost measure:", #TODO
        print
        vali_loss_trace.append(vali_loss)
        vali_acc_trace.append(vali_acc)
    
        if vali_loss > best_vali_loss:
            best_params = 0 #TODO
            best_rms = 0 #TODO
            best_vali_loss = vali_loss

#        save_path = saver.save(sess, "...")
#        print("Model saved to %s" % save_path)

# --- evaluation on test set --- #
# TODO: evaluate best params on test set
