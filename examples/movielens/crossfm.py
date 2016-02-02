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
numips = [ int (x) for x in os.environ['numips'].split (',') ]
maxshufbuf = int (os.environ['maxshufbuf'])
eta = float (os.environ['eta'])
etamin = float (os.environ['etamin'])
etadecay = float (os.environ['etadecay'])
weightdecay = float (os.environ['weightdecay'])
myname = os.path.splitext (os.path.basename (os.path.realpath (__file__)))
protofilename = myname[0] + 'net.proto'
netfilename = myname[0] + 'net.model'
vizfilename = myname[0] + 'net.png'

def do_test (net, labels, inputs, starttime):
  bindex = 0
  sumloss = 0
  sumacc = 0
  numbatches = 0
  for movie, user, rating in MovieLens.data ('./ml-1m/', maxshufbuf, True):
    inputs[bindex, 0, 0, :] = 0
    inputs[bindex, 0, 0, user - 1] = 1
    inputs[bindex, 0, 0, numusers + movie - 1] = 1

    labels[bindex, 0, 0, 0] = rating - 1

    bindex += 1

    if bindex >= batchsize:
      net.set_input_arrays (inputs, labels)
      res = net.forward ()

      sumloss += res['loss']
      sumacc += res['acc']    
      numbatches += 1
      bindex = 0

  now = time.time ()
  print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4s\t%9s"% (
      nicetime (float (now-starttime)),
      sumloss/numbatches,
      sumloss/numbatches,
      100 * (sumacc / numbatches), 
      nicecount (numbatches*batchsize),
      '*',
      'test')

spec = TheNet.crossfm (batchsize, numusers, nummovies, numratings, rank, numips)
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

for passes in range(100):
  for movie, user, rating in MovieLens.data ('./ml-1m/', maxshufbuf, False):
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
        try:
          os.remove (netfilename + "." + str (numupdates))
        except:
          pass

        net.save (netfilename + "." + str (numupdates))

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

        if passes > 0:
          testnet = caffe.Net(protofile, 
                              netfilename + "." + str (numupdates),
                              caffe.TEST)
          do_test (testnet, labels, inputs, starttime)
          del testnet

try:
  os.remove (netfilename)
except:
  pass

net.save (netfilename)

now = time.time ()
print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"% (
    nicetime (float (now-starttime)),
    sumloss/numupdates,
    sumsinceloss/numsinceupdates,
    100 * (sumsinceacc / numsinceupdates), 
    nicecount (numupdates*batchsize),
    passes,
    eta)

testnet = caffe.Net(protofile, netfilename, caffe.TEST)
do_test (testnet, labels, inputs, starttime)
del testnet

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./crossfm.py
#   delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
#   105ms   1.612   1.612  16.200    1000    0    9.990e-03
#   143ms   1.611   1.609  21.600    2000    0    9.980e-03
#   214ms   1.608   1.604  27.300    4000    0    9.960e-03
#   350ms   1.596   1.585  32.050    8000    0    9.920e-03
#   620ms   1.561   1.525  35.512     16K    0    9.841e-03
#  1.157s   1.518   1.475  34.513     32K    0    9.685e-03
#  2.267s   1.512   1.506  34.363     64K    0    9.380e-03
#  4.749s   1.511   1.510  34.658    128K    0    8.798e-03
#  9.786s   1.502   1.494  33.872    256K    0    7.740e-03
# 19.965s   1.470   1.437  36.100    512K    0    5.991e-03
# 38.819s   1.441   1.412  37.615   1024K    1    3.590e-03
# 41.846s   1.509   1.509  33.700    3000    *         test
#  1.298m   1.393   1.346  41.465   2048K    2    1.289e-03
#  1.349m   1.428   1.428  38.800    3000    *         test
#  2.538m   1.353   1.313  42.611   4096K    4    1.661e-04
#  2.588m   1.467   1.467  36.467    3000    *         test
#  4.860m   1.320   1.287  43.534   8192K    8    2.857e-06
#  4.912m   1.434   1.434  37.900    3000    *         test
#  9.461m   1.300   1.280  43.752     16M   16    1.008e-07
#  9.513m   1.421   1.421  38.367    3000    *         test
# 18.546m   1.290   1.280  43.846     32M   32    1.000e-07
# 18.597m   1.418   1.418  37.900    3000    *         test
# 36.484m   1.284   1.279  43.822     65M   65    1.000e-07
# 36.536m   1.425   1.425  37.667    3000    *         test
# 55.236m   1.282   1.277  43.913     99M   99    1.000e-07
# 55.290m   1.421   1.421  38.133    3000    *         test
