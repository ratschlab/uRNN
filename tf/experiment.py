#!/usr/bin/env ipython
#
# Outermost experiment-running script.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         20/3/16
# ------------------------------------------

import tensorflow as tf
import models
import utils
import data

# --- constants --- #
N_TRAIN = 1e5
N_TEST = 1e4

# --- hyperparameters and options --- #
options = utils.ExperimentOptions()

# --- experiment-specific things --- #
if options.experiment == 'adding':
    train_x, train_y = data.generate_adding(options.T, N_TRAIN)
    test_x, test_y = data.generate_adding(options.T, N_TEST)
    xy_dtype = tf.float32
elif options.experiment == 'memory':
    train_x, train_y = data.generate_memory(options.T, N_TRAIN)
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
    cost, parameters = models.trivial(inputs, 
                                      options.n_input, 
                                      options.n_hidden, 
                                      options.n_output,
                                      options.experiment)
else:
    raise NotImplementedError

# --- fold in optimizer --- #
opt = tf.train.RMSPropOptimizer(options.learning_rate, options.decay, options.momentum, options.epsilon)
grads_and_vars = opt.compute_gradients(cost, parameters)
if options.clipping:
    clipped_grads_and_vars = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in grads_and_vars]
    grads_and_vars = clipped_grads_and_vars
train_step = opt.apply_gradients(grads_and_vars)

sys.exit()
# TODO:  all from here, obviously

# --- run through batches --- #
for i in xrange(options.n_iter):
    batch = get_batch(... something about batch_size)
    train_step.run(feed_dict={x: batch_x, y: batch_y})
