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
      elif name == "linearterms":
        blob.data[:] = 0
      else:
        blob.data[:] = (1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:] = 0
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

# LOG_minloglevel=5 PYTHONPATH=../../python python ./crossfm.py
#   delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
#   107ms   1.612   1.612  15.600    1000    0    9.990e-03
#   160ms   1.611   1.610  17.000    2000    0    9.980e-03
#   244ms   1.608   1.604  28.700    4000    0    9.960e-03
#   408ms   1.596   1.585  32.250    8000    0    9.920e-03
#   746ms   1.561   1.525  35.525     16K    0    9.841e-03
#  1.419s   1.518   1.474  34.513     32K    0    9.685e-03
#  2.747s   1.512   1.507  34.225     64K    0    9.380e-03
#  5.420s   1.510   1.507  34.502    128K    0    8.799e-03
# 10.589s   1.501   1.492  33.998    256K    0    7.743e-03
# 21.026s   1.468   1.435  36.108    512K    0    5.995e-03
# 43.045s   1.440   1.411  37.540   1024K    1    3.596e-03
# 46.136s   1.507   1.507  34.800    3000    *         test
#  1.508m   1.393   1.346  41.373   2048K    2    1.297e-03
#  1.559m   1.433   1.433  39.133    3000    *         test
#  3.072m   1.353   1.313  42.555   4096K    4    1.759e-04
#  3.124m   1.459   1.459  35.967    3000    *         test
#  6.119m   1.320   1.287  43.499   8192K    8    1.275e-05
#  6.171m   1.414   1.414  38.500    3000    *         test
# 12.174m   1.299   1.278  43.903     16M   16    1.000e-05
# 12.225m   1.419   1.419  38.133    3000    *         test
# 22.271m   1.282   1.265  44.497     32M   32    1.000e-05
# 22.321m   1.402   1.402  38.233    3000    *         test
# 40.388m   1.262   1.241  45.550     65M   65    1.000e-05
# 40.438m   1.387   1.387  39.267    3000    *         test
#  1.015h   1.247   1.218  46.420     99M   99    1.000e-05
#  1.015h   1.389   1.389  38.100    3000    *         test
