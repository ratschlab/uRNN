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
models = ['IRNN', 
          'LSTM', 
          'tanhRNN',
          'complex_RNN',
          'uRNN']
 
#          '80_tanhRNN', 
#          '32_tanhRNN',
#          '18_tanhRNN',
#          '17_tanhRNN',
#          '16_tanhRNN',
#          '15_tanhRNN',
#          '14_tanhRNN',
#          '2_tanhRNN']
         #          'complex_RNN', 
          #          'ortho_tanhRNN',
          #          'ortho16_tanhRNN', 
          #          'ortho64_tanhRNN', 
          #          'ortho128_tanhRNN', 
cmap = mpl.cm.get_cmap('gist_rainbow')

colours = {'80_tanhRNN': mpl.colors.rgb2hex(cmap(0)),
           '32_tanhRNN': mpl.colors.rgb2hex(cmap(1.0*(1.0/len(models)))),
           '18_tanhRNN': mpl.colors.rgb2hex(cmap(2.0*(1.0/len(models)))),
           '17_tanhRNN': mpl.colors.rgb2hex(cmap(3.0*(1.0/len(models)))),
           '16_tanhRNN': mpl.colors.rgb2hex(cmap(4.0*(1.0/len(models)))),
           '15_tanhRNN': mpl.colors.rgb2hex(cmap(5.0*(1.0/len(models)))),
           '14_tanhRNN': mpl.colors.rgb2hex(cmap(6.0*(1.0/len(models)))),
           '2_tanhRNN': mpl.colors.rgb2hex(cmap(7.0*(1.0/len(models)))),
           'tanhRNN': 'red',
           'IRNN': 'pink', 
           'LSTM': 'green', 
           'complex_RNN': 'blue', 
           'ortho_tanhRNN': 'purple', 
           'uRNN': 'grey', 
           'ortho16_tanhRNN': 'orange', 
           'ortho64_tanhRNN': 'magenta', 
           'ortho128_tanhRNN': 'cyan'}

# === grab inputs === #
#T = int(sys.argv[1])
#task = sys.argv[2]
#plot_vali = sys.argv[3] == 'vali'
T = 100
#task = 'memory'
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
    plot_fname = task + '/' + str(T) + '_vali.png'
else:
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
    try:
        batch_size = trace['batch_size']
    except KeyError:
        batch_size = 20 # probably
    if plot_vali:
        trace['scaling_factor'] =  50*batch_size
    else:
        trace['scaling_factor'] = batch_size
        # TODO: finish fixing this
    n_train = trace['scaling_factor']*len(trace[loss])
    if n_train > xmax:
        xmax = n_train
    train_max = np.nanmax(trace[loss])
    if train_max > ymax:
        ymax = train_max
    train_min = np.nanmin(trace[loss])
    if train_min < ymin:
        ymin = train_min

if task == 'memory':
    display_scaling = 1000
else:
    display_scaling = 100

ymax = 0.25
xmax = xmax/display_scaling
print 0, xmax
print ymin, ymax
ymin = -0.001

plt.axis([0, xmax, ymin, ymax])
# construct the arguments to plot
data_series = dict()
for model in traces.keys():
    series_x = traces[model]['scaling_factor']*np.arange(len(traces[model][loss]))/display_scaling
    series_y = np.array(traces[model][loss])
    colour = colours[model]
    data_series[model], = plt.plot(series_x, series_y, colour, label=model, alpha=0.8)
if task == 'memory':
    plt.xlabel("training examples (thousands)")
else:
    plt.xlabel("training examples (hundreds)")
plt.ylabel(score)
plt.legend(loc='upper right')
plt.title(loss)
# now save ok
plt.savefig(plot_fname)
