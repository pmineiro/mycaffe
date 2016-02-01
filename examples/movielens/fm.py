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
          do_test (net, labels, inputs, starttime)

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

do_test (net, labels, inputs, starttime)

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./fm.py                         delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
#   111ms   1.609   1.609  21.500    1000    0    9.990e-03
#   161ms   1.609   1.608  23.400    2000    0    9.980e-03
#   250ms   1.607   1.606  27.300    4000    0    9.960e-03
#   422ms   1.603   1.599  30.625    8000    0    9.920e-03
#   745ms   1.585   1.567  35.200     16K    0    9.841e-03
#  1.400s   1.544   1.502  34.513     32K    0    9.685e-03
#  2.667s   1.503   1.462  35.175     64K    0    9.380e-03
#  5.248s   1.533   1.563  32.853    128K    0    8.798e-03
# 10.399s   1.545   1.556  34.580    256K    0    7.740e-03
# 20.630s   1.528   1.512  33.480    512K    0    5.991e-03
# 39.219s   1.511   1.493  34.688   1024K    1    3.590e-03
# 42.283s   1.639   1.639  30.767    3000    *         test
#  1.339m   1.547   1.583  38.486   2048K    2    1.289e-03
#  1.389m   1.686   1.686  36.133    3000    *         test
#  2.628m   1.551   1.555  39.659   4096K    4    1.661e-04
#  2.678m   1.536   1.536  36.667    3000    *         test
#  5.178m   1.442   1.333  42.262   8192K    8    2.857e-06
#  5.229m   1.430   1.430  37.067    3000    *         test
# 10.260m   1.366   1.291  43.317     16M   16    1.008e-07
# 10.310m   1.436   1.436  37.100    3000    *         test
# 20.220m   1.327   1.288  43.489     32M   32    1.000e-07
# 20.270m   1.443   1.443  37.867    3000    *         test
# 40.198m   1.307   1.286  43.545     65M   65    1.000e-07
# 40.250m   1.430   1.430  38.467    3000    *         test
#  1.019h   1.298   1.282  43.632     99M   99    1.000e-07
#  1.020h   1.426   1.426  38.833    3000    *         test
