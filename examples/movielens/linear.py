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

spec = TheNet.linear (batchsize, numusers, nummovies, numratings)
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

        if passes > 0:
          do_test (net, labels, inputs, starttime)

# try:
#   os.remove (netfilename)
# except:
#   pass
# 
# net.save (netfilename)

now = time.time ()
print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"% (
    nicetime (float (now-starttime)),
    sumloss/numupdates,
    sumsinceloss/numsinceupdates,
    100 * (sumsinceacc / numsinceupdates), 
    nicecount (numupdates*batchsize),
    passes,
    eta)

do_test (net, labels, inputs, starttime)

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./linear.py
#   delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
#    86ms   1.611   1.611  17.800    1000    0    9.990e-03
#   121ms   1.611   1.611  17.300    2000    0    9.980e-03
#   186ms   1.609   1.608  24.950    4000    0    9.960e-03
#   317ms   1.604   1.599  30.450    8000    0    9.920e-03
#   576ms   1.586   1.567  35.238     16K    0    9.841e-03
#  1.112s   1.544   1.502  34.513     32K    0    9.685e-03
#  2.225s   1.503   1.461  35.175     64K    0    9.380e-03
#  4.432s   1.533   1.564  32.983    128K    0    8.798e-03
#  8.791s   1.545   1.557  34.346    256K    0    7.740e-03
# 17.256s   1.541   1.536  33.768    512K    0    5.991e-03
# 33.722s   1.534   1.527  34.799   1024K    1    3.590e-03
# 36.751s   1.543   1.543  31.033    3000    *         test
#  1.166m   1.500   1.466  38.548   2048K    2    1.289e-03
#  1.216m   1.718   1.718  37.133    3000    *         test
#  2.258m   1.445   1.390  41.038   4096K    4    1.661e-04
#  2.307m   1.469   1.469  34.533    3000    *         test
#  4.392m   1.381   1.316  42.725   8192K    8    2.857e-06
#  4.441m   1.408   1.408  38.967    3000    *         test
#  8.640m   1.335   1.290  43.432     16M   16    1.008e-07
#  8.689m   1.429   1.429  37.633    3000    *         test
# 17.002m   1.312   1.289  43.372     32M   32    1.000e-07
# 17.053m   1.447   1.447  36.700    3000    *         test
# 33.702m   1.300   1.288  43.381     65M   65    1.000e-07
# 33.752m   1.437   1.437  36.900    3000    *         test
# 51.144m   1.296   1.286  43.592     99M   99    1.000e-07
# 51.194m   1.430   1.430  37.267    3000    *         test
