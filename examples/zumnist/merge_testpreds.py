#! /usr/bin/env python

import h5py
import numpy as np
import sys

numnets = int(sys.argv[1]);
f = h5py.File(sys.argv[2],"w")

data = f.create_dataset("data", (60000*numnets,1,28,28), dtype='f')

mnist = open('../../data/mnist/train-images-idx3-ubyte', 'r');
magic_s = mnist.read(4)
n_images_s = mnist.read(4)
n_rows_s = mnist.read(4)
n_columns_s = mnist.read(4)

n=0
for chunk in iter(lambda: mnist.read(28*28), ''):
  for i in range(0, numnets):
    data[n] = np.array([float(ord(c))/256.0 for c in list(chunk)]).reshape(1,28,28)
    n=n+1

labels = f.create_dataset("label", (60000*numnets,1), dtype='f')

n=0
epsilon=1e-6
for line in sys.stdin:
  values = [float(digit) for digit in line.split()]
  for i in range(0, numnets):
    labels[n]=values[i]
    n=n+1

f.close()
