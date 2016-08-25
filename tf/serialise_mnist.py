#!/usr/bin/env

import mnist
import numpy as np

path = '/home/hyland/git/complex_RNN/tf/input/mnist'

# === train/vali split === #
images, labels = mnist.load_mnist('training', path=path)
vali_indices = np.random.choice(60000, 6000, replace=False)
train_indices = [j for j in xrange(60000) if not j in vali_indices]
assert len(set(vali_indices).intersection(set(train_indices))) == 0

x = np.zeros(shape=(54000, 784))
for (i, j) in enumerate(train_indices):
    x[i, :] = images[j].flatten()

y = np.int64(labels[train_indices])

np.save('train_x', x)
np.save('train_y', y)

# === vali! === #

x = np.zeros(shape=(6000, 784))
for (i, j) in enumerate(vali_indices):
    x[i, :] = images[j].flatten()

y = np.int64(labels[vali_indices])

np.save('vali_x', x)
np.save('vali_y', y)
# === test! === #
images, labels = mnist.load_mnist('testing', path=path)

x = np.zeros(shape=(10000, 784))
for (i, image) in enumerate(images):
    x[i, :] = image.flatten()

y = np.int64(labels)

np.save('test_x', x)
np.save('test_y', y)
