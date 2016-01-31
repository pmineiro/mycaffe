import os

import caffe
import collections
import math
import numpy as np
import random
import time

import MovieLens
from Pretty import nicetime, nicecount
import TheNet

import pdb

random.seed (8675309)
np.random.seed (90210)

batchsize = 1000
numusers = 3900
nummovies = 6400
numratings = 5
maxshufbuf = 10000
eta = 1.0
etamin = 1e-3
etadecay = 0.999
weightdecay = 1e-4
protofilename = 'fullnet.proto'
netfilename = 'fullnet.model'

spec = TheNet.full (batchsize, numusers, nummovies, numratings)

protofile = os.path.join (os.path.dirname (os.path.realpath (__file__)), protofilename)

with open (protofile, 'w') as f:
  f.write (str (spec))

net = caffe.Net(protofile, caffe.TRAIN)

for name, layer in zip (net._layer_names, net.layers):
  blobnum=0
  for blob in layer.blobs:
    if blobnum == 0:
      if name == "dot":
        blob.data[:] = 1
      else:
        blob.data[:] = (1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
      blobnum = blobnum + 1

lrs = collections.defaultdict (lambda: 1.0)
lrs[('dot', 0)] = 0

print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta","average","since","acc","example","pass","learning")
print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("t","loss","last","since","counter","num","rate")

starttime=time.time()
numsinceupdates=0
numupdates=0
sumloss=0
sumsinceloss=0
sumsinceacc=0
nextprint=1
labels=np.zeros((batchsize,1,1,1),dtype='f')
inputs=np.zeros((batchsize,1,1,numusers+nummovies),dtype='f')
bindex=0

for passes in range(1):
  for user, movie, rating in MovieLens.data('./ml-1m/', maxshufbuf):
    inputs[bindex, 0, 0, :] = 0
    inputs[bindex, 0, 0, user - 1] = 1
    inputs[bindex, 0, 0, numusers + movie - 1] = 1

    labels[bindex, 0, 0, 0] = rating - 1

    bindex += 1

    if bindex >= batchsize:
      net.set_input_arrays (inputs, labels)
      res = net.forward ()

      sumloss += res['loss']
      sumsinceloss += res['loss']
      sumsinceacc += res['acc']    

      net.backward ()

      for (name,layer) in zip (net._layer_names, net.layers):
        blobnum = 0
        for blob in layer.blobs:
          myeta = lrs[(name, blobnum)] * eta
          if myeta > 0:
            blob.data[:] -= myeta * blob.diff
            if weightdecay > 0:
              blob.data[:] *= (1.0 - weightdecay * myeta)
            blobnum = blobnum + 1

      bindex = 0 

      numupdates += 1
      numsinceupdates += 1
      eta = etadecay * eta + (1.0 - etadecay) * etamin

      if numupdates >= nextprint:
#        try:
#          os.remove (netfilename + "." + str (numupdates))
#        except:
#          pass
#
#        net.save (netfilename + "." + str (numupdates))

        now = time.time ()
        print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"% (
            nicetime (float (now-starttime)),
            sumloss/numupdates,
            sumsinceloss/numsinceupdates,
            100 * (sumsinceacc / numsinceupdates), 
            nicecount (numupdates*batchsize),
            passes,
            eta)

        nextprint = 2 * nextprint
        numsinceupdates = 0
        sumsinceloss = 0
        sumsinceacc = 0

now = time.time ()
print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"% (
    nicetime (float (now-starttime)),
    sumloss/numupdates,
    sumsinceloss/numsinceupdates,
    100 * (sumsinceacc / numsinceupdates), 
    nicecount (numupdates*batchsize),
    passes,
    eta)

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./full.py
#   delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
#  7.548s   1.611   1.611  17.900    1000    0    9.990e-01
# 15.473s   1.583   1.555  33.600    2000    0    9.980e-01
# 30.797s   1.528   1.473  32.600    4000    0    9.960e-01
# 59.633s   1.514   1.500  33.875    8000    0    9.920e-01
#  1.974m   1.514   1.514  36.150     16K    0    9.841e-01
#  3.906m   1.505   1.496  37.244     32K    0    9.685e-01
#  7.684m   1.476   1.447  37.681     64K    0    9.380e-01
# 15.286m   1.473   1.469  37.777    128K    0    8.799e-01
# 30.347m   1.519   1.566  36.535    256K    0    7.743e-01
#  1.010h   1.541   1.563  36.311    512K    0    5.995e-01
