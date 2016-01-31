import os

import caffe
import collections
import math
import numpy as np
import random
import time

import draw
import MovieLens
from Pretty import nicetime, nicecount
import TheNet

import pdb

random.seed (8675309)
np.random.seed (90210)

batchsize = int (os.environ['batchsize'])
numusers = int (os.environ['numusers'])
nummovies = int (os.environ['nummovies'])
numratings = int (os.environ['numratings'])
rank = int (os.environ['rank'])
maxshufbuf = int (os.environ['maxshufbuf'])
eta = float (os.environ['eta'])
etamin = float (os.environ['etamin'])
etadecay = float (os.environ['etadecay'])
weightdecay = float (os.environ['weightdecay'])
myname = os.path.splitext (os.path.basename (os.path.realpath (__file__)))
protofilename = myname[0] + 'net.proto'
netfilename = myname[0] + 'net.model'
vizfilename = myname[0] + 'net.png'

spec = TheNet.fm (batchsize, numusers, nummovies, numratings, rank)
draw.draw_net_to_file (spec, vizfilename)

protofile = os.path.join (os.path.dirname (os.path.realpath (__file__)), protofilename)

with open (protofile, 'w') as f:
  f.write (str (spec))

caffe.set_mode_gpu ()
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

#GLOG_minloglevel=5 PYTHONPATH=../../python python ./fm.py                         delta average   since     acc example pass     learning
#      t    loss    last   since counter  num         rate
#   84ms   1.609   1.609  19.100    1000    0    9.991e-04
#  128ms   1.609   1.609  20.500    2000    0    9.982e-04
#  214ms   1.609   1.609  21.850    4000    0    9.964e-04
#  387ms   1.609   1.608  22.275    8000    0    9.928e-04
#  732ms   1.607   1.605  26.550     16K    0    9.857e-04
# 1.402s   1.600   1.593  31.988     32K    0    9.716e-04
# 2.589s   1.576   1.553  35.181     64K    0    9.442e-04
# 5.238s   1.528   1.479  35.139    128K    0    8.918e-04
#10.281s   1.533   1.538  34.515    256K    0    7.966e-04
#20.974s   1.523   1.512  32.095    512K    0    6.392e-04
#39.976s   1.509   1.496  31.712   1000K    0    4.309e-04
