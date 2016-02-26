#!/usr/bin/env ipython
# VISUALISATION IN PYTHON
# ... just cost curves etc.

import cPickle
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- initialise --- #
# grab inputs
T = int(sys.argv[1])
task = sys.argv[2]
assert task in {'adding', 'memory'}

# constants etc.
#   we get train loss after every iteration
#   we get test loss after every 50 iterations
#   each iteration contains n_batch training examples
n_batch = 20
models = ['RNN', 'IRNN', 'LSTM', 'complex_RNN']

# --- load all the data --- #
# (very deep dicts)
traces = dict()
for model in models:
    trace_file = task + '_' + model + '_' + str(T)
    try:
        traces[model] = cPickle.load(open(trace_file))
        # check consistency
        assert traces[model]['time_steps'] == T
        assert traces[model]['model'] == model
        print 'Loaded', trace_file
    except IOError:
        print 'Missing:', trace_file

# --- create plots --- #
# === train loss
xmax = 0
ymax = -1000
ymin = 1000
for (model, trace) in traces.iteritems():
    n_train = 20*len(trace['train_loss'])
    if n_train > xmax:
        xmax = n_train
    train_max = np.nanmax(trace['train_loss'])
    if train_max > ymax:
        ymax = train_max
    train_min = np.nanmin(trace['train_loss'])
    if train_min < ymin:
        ymin = train_min

print 0, xmax
print ymin, ymax

#plt.axis([0, xmax, ymin, ymax])
# construct the arguments to plot
args = np.array([[20*np.arange(len(x['train_loss'])), np.array(x['train_loss']), 'r'] for x in traces.values()]).flatten()
print args
plt.plot(*args)
