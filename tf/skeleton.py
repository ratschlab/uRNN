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

#
BATCH_SIZE = 10
CELL_INPUT_SIZE = 1
SEQ_LENGTH = 120

# EVERYTHING IS FIRE

# need a way of controlling all the experimental options
#train_op = # thing what updates variables given thems and the costs
#get_batch = # thing what prepares batch of data to be fed into model
# need thing to get predictions from RNN (model-specific, also experiment-ish)
# need thing to get cost from predictions and labels (experiment-specific)
#   (this can be fed test or train or vali data!)
# need thing to save parameters/model/graph for 'best results'
#   (will need to reload these values for the testing at the end)

def get_cost(outputs, y):
    """
    check experimental stuff also
    """
    cost = tf.zeros(shape=(SEQ_LENGTH, 1))
    for i in xrange(SEQ_LENGTH):
        out = outputs[i][:, 0]
        y_i = y[:, i]
        # THIS IS ABSURD
        intermediate = tf.add(out, tf.cast(y_i, tf.float32))
        cost = tf.add(cost, intermediate)
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

def main():
    
    with tf.Session() as session:
        # === construct the graph === #
        batch_x = tf.placeholder(tf.float32, [BATCH_SIZE, SEQ_LENGTH, CELL_INPUT_SIZE])
        batch_y = tf.placeholder(tf.float32, [BATCH_SIZE, CELL_INPUT_SIZE])

        # === model select === #
        predictions = models.simple_RNN(batch_x, n_hidden=20, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH)

        # === ops and things === #
        cost = get_cost(predictions, batch_y)
        train_op = update_step(cost)

        # === train loop === #
        tf.initialize_all_variables()
        pdb.set_trace()
        sys.exit()
        
        for epoch in xrange(config.num_epochs):
            for batch in xrange(config.num_batches):
                train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                print epoch, '\t', batch, '\t', train_cost
                if DO_VALI:
                    vali_cost = session.run([cost], {x: vali_x, y: vali_y})
                    print '\t\tTEST:', vali_cost
                    # check if best, save parameters, etc.

            # shuffle the data at each epoch
            #data.shuffle()

        print 'Training completed. Performance of best model (by validation set) on test data:',
        test_cost = session.run([cost], {x: test_x, y: test_y, parameters: best_parameters})

#if __name__ == "__main__":
#    main()

