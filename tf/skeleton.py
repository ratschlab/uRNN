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

# local imports
import models
import data

# === constants === #
N_TRAIN = int(1e5)
N_TEST = int(1e4)
N_VALI = int(1e4)

DO_TEST = False

# EVERYTHING IS FIRE

# need a way of controlling all the experimental options

def get_cost(outputs, y, loss_fn='MSE'):
    """
    Either cross-entropy or MSE.
    This will involve some averaging over a batch innit.

    Let's clarify some shapes:
        outputs is a LIST of length input_size,
            each element is a Tensor of shape (BATCH_SIZE, output_size)
        y is a Tensor of shape (BATCH_SIZE, output_size)
    """
    if loss_fn == 'MSE':
        # discount all but the last of the outputs
        output = outputs[-1]
        # now this object is shape BATCH_SIZE, output_size
        cost = tf.reduce_mean(tf.sub(output, y) ** 2)
    elif loss_fn == 'CE':
        # TODO: this is returning a list rather than a single value: fix
        #ok, cross-entropy!
        # (there may be more efficient ways to do this)
        cost = tf.zeros([1])
        for (i, output) in enumerate(outputs):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y[:, i])
            cost = tf.add(cost, tf.reduce_mean(cross_entropy))
        cost = tf.div(cost, i + 1)
    else:
        raise NotImplementedError
    return cost

def update_step(cost, learning_rate=0.01, decay=0.01, 
                momentum=0.9, epsilon=0.0001, clipping=False):
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                    decay=decay,
                                    momentum=momentum,
                                    epsilon=epsilon)
    print 'By the way, the gradients of cost are with respect to the following Variables:'
    for v in tf.trainable_variables():
        print v.name
    if clipping:
        # optional clipping may occur
        # TODO: clipping
        pass
    g_and_v = opt.compute_gradients(cost, tf.trainable_variables())
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt

def main(experiment='adding'):
    # === get options === #
    # temporary constants (will be folded into a cfg)
    # TODO: fix all
    state_size = 20
    BATCH_SIZE = 10
    NUM_EPOCHS = 5
    NUM_BATCHES = N_TRAIN / BATCH_SIZE
    T = 100
    best_model_path = 'best_model.ckpt'
    trace_path = 'eyo'

    # === create data === #
    train_data = data.ExperimentData(N_TRAIN, experiment, T)
    vali_data = data.ExperimentData(N_VALI, experiment, T)
    test_data = data.ExperimentData(N_TEST, experiment, T)
  
    # === get shapes and constants === #
    sequence_length = train_data.sequence_length
    input_size = train_data.input_size
    # YOLO: finish doing this bit
    if experiment == 'adding':
        output_size = 1
        loss_fn = 'MSE'
        assert input_size == 2
    elif experiment == 'memory':
        output_size = 9
        loss_fn = 'CE'
        assert input_size == 10

    # === construct the graph === #
    # (doesn't actually matter which one we make placeholders out of)
    x, y = train_data.make_placeholders()

    # === model select === #
    outputs = models.simple_RNN(x, input_size, state_size, output_size, sequence_length=sequence_length)

    # === ops and things === #
    cost = get_cost(outputs, y, loss_fn)
    train_op = update_step(cost)

    # === for checkpointing the model === #
    saver = tf.train.Saver()

    # === init === #
    train_cost_trace = []
    vali_cost_trace = []
    best_vali_cost = 1e6
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        
        # === train loop === #
        for epoch in xrange(NUM_EPOCHS):
            for batch_index in xrange(NUM_BATCHES):
                # definitely scope for fancy iterator but yolo
                batch_x, batch_y = train_data.get_batch(batch_index, BATCH_SIZE)
             
                train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                train_cost_trace.append(train_cost)
                print epoch, '\t', batch_index, '\t', train_cost
                if batch_index % 50 == 0:
                    vali_cost = session.run(cost, {x: vali_data.x, y: vali_data.y})
                    vali_cost_trace.append(vali_cost)
                    if vali_cost < best_vali_cost:
                        # save best parameters
                        best_vali_cost = vali_cost
                        save_path = saver.save(session, best_model_path)
                        print epoch, '\t', batch_index, '\t*** VALI:', vali_cost, '\t('+save_path+')'
                    else:
                        print epoch, '\t', batch_index, '\t    VALI:', vali_cost

                    # NOTE: format consistent with theano version
                    # TODO: update alongside plotting script
                    save_vals = {'parameters': None,
                                 'rmsprop': None,
                                 'train_loss': train_cost_trace,
                                 'test_loss': vali_cost_trace,
                                 'best_params': None,
                                 'best_rms': None,
                                 'best_test_loss': best_vali_cost,
                                 'model': None,
                                 'time_steps': T}

                    cPickle.dump(save_vals, file(trace_path, 'wb'),
                                 cPickle.HIGHEST_PROTOCOL)

            # shuffle the data at each epoch
            train_data.shuffle()

        print 'Training completed.'
        if DO_TEST:
            print 'Loading best model from', best_model_path
            saver.restore(session, best_model_path)
            test_cost = session.run(cost, {x: test_data.x, y: test_data.y})
            print 'Performance on test set:', test_cost

#if __name__ == "__main__":
#    main()

