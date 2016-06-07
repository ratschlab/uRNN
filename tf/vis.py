#!/usr/bin/env ipython
# VISUALISATION IN PYTHON
# ... just cost curves etc.

import cPickle
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.clf()

# === constants === #
BATCH_SIZE = 20
models = ['tanhRNN', 'IRNN', 'LSTM', 'complex_RNN', 'ortho20_tanhRNN', 'ortho_tanhRNN', 'uRNN']
colours = {'tanhRNN': 'r', 'IRNN': 'pink', 'LSTM': 'g', 'complex_RNN': 'b', 'ortho_tanhRNN': 'purple', 'uRNN': 'grey', 'ortho20_tanhRNN': 'orange'}

# === grab inputs === #
#T = int(sys.argv[1])
#task = sys.argv[2]
#plot_vali = sys.argv[3] == 'vali'
T = 100
task = 'adding'
plot_vali = sys.argv[1] == 'vali'

# === initialise === #
if task == 'adding':
    score = 'MSE'
elif task == 'memory':
    score = 'CE'
else:
    sys.exit('Unknown task', task)

if plot_vali:
    #   each iteration contains n_batch training examples
    #   we get vali loss after every 50 iterations
    scaling_factor = 50*BATCH_SIZE
    plot_fname = task + '/' + str(T) + '_vali.png'
else:
    #   we get train loss after every iteration
    scaling_factor = BATCH_SIZE
    plot_fname = task + '/' + str(T) + '_train.png'

plot_fname = 'output/' + plot_fname
print plot_fname

# === load all the data === #
# (very deep dicts)
traces = dict()
for model in models:
    trace_file = 'output/' + task + '/' + model + '_' + str(T) + '.trace.pk'
    try:
        traces[model] = cPickle.load(open(trace_file))
        # check consistency
        assert traces[model]['time_steps'] == T
#        assert traces[model]['model'] == model
        print 'Loaded', trace_file
    except IOError:
        print 'Missing:', trace_file

# --- create plots --- #
# vali or train?
if plot_vali:
    loss = 'vali_loss'
else:
    loss = 'train_loss'

# determine y/x min/max
xmax = 0
ymax = -1000
ymin = 1000
for (model, trace) in traces.iteritems():
    n_train = len(trace[loss])
    if n_train > xmax:
        xmax = n_train
    train_max = np.nanmax(trace[loss])
    if train_max > ymax:
        ymax = train_max
    train_min = np.nanmin(trace[loss])
    if train_min < ymin:
        ymin = train_min

xmax = scaling_factor*xmax
#xmax = 10000
ymax = 0.25
print 0, xmax
print ymin, ymax
ymin = -0.001

plt.axis([0, xmax, ymin, ymax])
# construct the arguments to plot
data_series = dict()
for model in traces.keys():
    series_x = scaling_factor*np.arange(len(traces[model][loss]))
    series_y = np.array(traces[model][loss])
    colour = colours[model]
    data_series[model], = plt.plot(series_x, series_y, colour, label=model, alpha=0.8)
plt.xlabel("training examples")
plt.ylabel(score)
plt.legend(loc='upper right')
plt.title(loss)
# now save ok
plt.savefig(plot_fname)
