#! /usr/bin/env python

import h5py
import math
import numpy as np
import sys

np.random.seed(69)

train = h5py.File(sys.argv[1] + ".train.hdf5" ,"w")
test = h5py.File(sys.argv[1] + ".test.hdf5" ,"w")

numlines=10000
vocabsize=80000
invocabsize=vocabsize+2

testnumlines=int(math.floor(0.1*numlines))
trainnumlines=numlines-testnumlines
perm=np.random.permutation(numlines)
testlines=dict((i,1) for i in perm[0:testnumlines])

# TODO: compression
traindata = train.create_dataset("data", (trainnumlines,1,1,invocabsize), dtype='f')
trainlabel = train.create_dataset("label", (trainnumlines,1), dtype='f')
testdata = test.create_dataset("data", (testnumlines,1,1,invocabsize), dtype='f')
testlabel = test.create_dataset("label", (testnumlines,1), dtype='f')

n=0
ntrain=0
ntest=0
for line in sys.stdin:
  x = np.zeros(invocabsize,dtype='f');
  xy=[word for word in line.split('\t')]
  for word in xy[1].split():
    iv=[subword for subword in word.split(':')]
    x[int(iv[0])-1]=float(iv[1])

  if n in testlines:
    testdata[ntest]=x.reshape(1,1,invocabsize)
    testlabel[ntest]=int(xy[0])-1
    ntest=ntest+1
  else:
    traindata[ntrain]=x.reshape(1,1,invocabsize)
    trainlabel[ntrain]=int(xy[0])-1
    ntrain=ntrain+1
  n=n+1

  if n >= numlines:
    break

train.close()
test.close()
