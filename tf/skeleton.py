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

# local imports
import models
import data

# === constants === #
N_TRAIN = int(1e5)
N_TEST = int(1e4)
N_VALI = int(1e4)
# temporary constants (will be folded into a cfg)
BATCH_SIZE = 10
NUM_EPOCHS = 5
NUM_BATCHES = N_TRAIN / BATCH_SIZE
SEQ_LENGTH = 10
T = 100
DO_VALI = False

# EVERYTHING IS FIRE

# need a way of controlling all the experimental options
#train_op = # thing what updates variables given thems and the costs
#get_batch = # thing what prepares batch of data to be fed into model
# need thing to get predictions from RNN (model-specific, also experiment-ish)
# need thing to get cost from predictions and labels (experiment-specific)
#   (this can be fed test or train or vali data!)
# need thing to save parameters/model/graph for 'best results'
#   (will need to reload these values for the testing at the end)

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
        cost = 2*tf.nn.l2_loss(tf.sub(output, y))
    elif loss_fn == 'CE':
        raise NotImplementedError
        # include all the outputs, but we need to do some weirdery, ugh
        wat = tf.pack(output)
        # now we have a tensor of shape (input_size, batch_size, output_size)
        # TODO: all omg oall
        # DO THINGS!
        # DO thINGS!
#        cost = tf.zeros(shape=(SEQ_LENGTH, 1))
#        for i in xrange(SEQ_LENGTH):
#            out = outputs[i][:SEQ_LENGTH, 0]
#            y_i = y[:, i]
            # THIS IS ABSURD
#            intermediate = tf.add(out, tf.cast(y_i, tf.float32))
#            cost = tf.add(cost, intermediate)

    else:
        raise NotImplementedError
    return cost

def update_step(cost):
    """
    uhh
    """
    opt = tf.train.RMSPropOptimizer(learning_rate=0.01,
                                    decay=0.01,
                                    momentum=0.9,
                                    epsilon=0.0001)
    print 'By the way, the gradients of cost are with respect to the following Variables:'
    for v in tf.trainable_variables():
        print v.name
    # optional clipping may occur
    g_and_v = opt.compute_gradients(cost, tf.trainable_variables())
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt

def main(experiment='adding'):
    # === create data === #
    train_data = data.ExperimentData(N_TRAIN, BATCH_SIZE, experiment, T)
    vali_data = data.ExperimentData(N_VALI, BATCH_SIZE, experiment, T)
    test_data = data.ExperimentData(N_TEST, BATCH_SIZE, experiment, T)
  
    # === get shapes and constants === #
    sequence_length = train_data.sequence_length
    input_size = train_data.input_size
    # YOLO: finish doing this bit
    if experiment == 'adding':
        output_size = 1
        loss_fn = 'MSE'
    elif experiment == 'memory':
        output_size = 9
        loss_fn = 'CE'
    hidden_size = 20

    with tf.Session() as session:
        # === construct the graph === #
        # (doesn't actually matter which one we make placeholders out of)
        x, y = train_data.make_placeholders()

        # === model select === #
        outputs = models.simple_RNN(x, n_hidden=hidden_size, batch_size=BATCH_SIZE, sequence_length=sequence_length)
        pdb.set_trace()

        # === ops and things === #
        cost = get_cost(outputs, y)
        train_op = update_step(cost)

        # === train loop === #
        tf.initialize_all_variables().run()
        
        for epoch in xrange(NUM_EPOCHS):
            for batch_index in xrange(NUM_BATCHES):
                # definitely scope for fancy iterator but yolo
                batch_x, batch_y = train_data.get_batch(batch_index)
              
                # TODO: BUG IN COST
                train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                pdb.set_trace()
                print epoch, '\t', batch_index, '\t', train_cost
                if DO_VALI:
                    vali_cost = session.run([cost], {x: vali_x, y: vali_y})
                    print '\t\tTEST:', vali_cost
                    # check if best, save parameters, etc.

            # shuffle the data at each epoch
            train_data.shuffle()

        print 'Training completed. Performance of best model (by validation set) on test data:',
        #test_cost = session.run([cost], {x: test_x, y: test_y, parameters: best_parameters})

#if __name__ == "__main__":
#    main()

