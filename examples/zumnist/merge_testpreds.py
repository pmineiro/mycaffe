#! /usr/bin/env python

import h5py
import numpy as np
import sys

f = h5py.File(sys.argv[1],"w")

data = f.create_dataset("data", (60000,1,28,28), dtype='f')

mnist = open('../../data/mnist/train-images-idx3-ubyte', 'r');
magic_s = mnist.read(4)
n_images_s = mnist.read(4)
n_rows_s = mnist.read(4)
n_columns_s = mnist.read(4)

n=0

for chunk in iter(lambda: mnist.read(28*28), ''):
  data[n] = np.array([float(ord(c)) for c in list(chunk)]).reshape(1,28,28)
  n=n+1

labels = f.create_dataset("label", (60000,10), dtype='f')

n=0
for line in sys.stdin:
  values = [float(digit) for digit in line.split()]
  valuesreshape=np.array(values).reshape(-1,10)
  nummodels=valuesreshape.shape[0]
  valuesarray = np.sum(valuesreshape,axis=0)/nummodels
  labels[n]=valuesarray
  n=n+1

f.close()
