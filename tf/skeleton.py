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

# === constants === #
N_TRAIN = int(1e5)
N_TEST = int(1e4)
N_VALI = int(1e4)

DO_TEST = False

# EVERYTHING IS FIRE

# need a way of controlling all the experimental options

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
        # discount all but the last of the outputs
        output = outputs[-1]
        # now this object is shape batch_size, output_size
        cost = tf.reduce_mean(tf.sub(output, y) ** 2)
    elif loss_type == 'CE':
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

def update_step(cost, learning_rate=0.01, clipping=False):
    # decay and momentum are copied from theano version values
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                    decay=0.9,
                                    momentum=0.0)
    print 'By the way, the gradients of cost',
    print 'are with respect to the following Variables:'
    for var in tf.trainable_variables():
        print var.name
    g_and_v = opt.compute_gradients(cost, tf.trainable_variables())
    if clipping:
        g_and_v = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in g_and_v]
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt

def main(experiment='adding', batch_size=10, state_size=20, 
         num_epochs=5, T=100, learning_rate=0.001,
         model='tanhRNN', timestamp=False):
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

    # === construct the graph === #
    # (doesn't actually matter which one we make placeholders out of)
    x, y = train_data.make_placeholders()

    # === model select === #
    outputs = RNN(model, x, input_size, state_size, output_size, sequence_length=sequence_length)

    # === ops and things === #
    cost = get_cost(outputs, y, loss_type)
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
             
                train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                train_cost_trace.append(train_cost)
                print epoch, '\t', batch_index, '\t', loss_type + ':', train_cost
                if batch_index % 50 == 0:
                    vali_cost = session.run(cost, {x: vali_data.x, y: vali_data.y})
                    vali_cost_trace.append(vali_cost)
                    if vali_cost < best_vali_cost:
                        # save best parameters
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
parser = argparse.ArgumentParser(description="run an experiment")
parser.add_argument("--experiment", type=str, default='adding')
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--state_size", type=int, default=512)
parser.add_argument("--T", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--model", default='orthogonal_RNN')
parser.add_argument("--timestamp", type=bool, default=True)
args = parser.parse_args()
