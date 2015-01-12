#! /usr/bin/env python

import h5py
import numpy as np
import os
import sys

def read_data(handle):
    handle.read(1)
    return handle.read(3*32*32)

numnets = int(sys.argv[1]);
f = h5py.File(sys.argv[2],"w")

data = f.create_dataset("data", (60000*numnets,3,32,32), dtype='f')
n=0

for batch in range(1, 6):
    cifar = open("../../data/cifar10/data_batch_%u.bin"%(batch), 'r');

    for chunk in iter(lambda: read_data(cifar), ''):
      for i in range(0, numnets):
        try:
            data[n] = np.array([float(ord(c)) for c in list(chunk)]).reshape(3,32,32)
        except:
            sys.stderr.write("batch = %u n = %u chunk len = %u\n"%(batch,n,len(list(chunk))))
            raise
        n=n+1

labels = f.create_dataset("label", (60000*numnets,1), dtype='f')

n=0
for line in sys.stdin:
  values = [float(digit) for digit in line.split()]
  for i in range(0, numnets):
    labels[n]=values[i]
    n=n+1

f.close()
