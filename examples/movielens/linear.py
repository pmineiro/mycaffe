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
      elif name == "Vtopa" or name == "Utopq":
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
#    87ms   1.609   1.609  24.700    1000    0    9.990e-03
#   124ms   1.609   1.609  35.200    2000    0    9.980e-03
#   193ms   1.608   1.606  35.200    4000    0    9.960e-03
#   331ms   1.603   1.598  33.275    8000    0    9.920e-03
#   604ms   1.585   1.567  35.537     16K    0    9.841e-03
#  1.129s   1.543   1.502  34.513     32K    0    9.685e-03
#  2.132s   1.502   1.461  35.175     64K    0    9.380e-03
#  4.336s   1.533   1.564  32.795    128K    0    8.799e-03
#  8.726s   1.545   1.557  34.591    256K    0    7.743e-03
# 17.372s   1.540   1.536  33.900    512K    0    5.995e-03
# 32.883s   1.534   1.527  34.780   1024K    1    3.596e-03
# 35.881s   1.545   1.545  31.033    3000    *         test
#  1.074m   1.500   1.466  38.563   2048K    2    1.297e-03
#  1.124m   1.712   1.712  37.167    3000    *         test
#  2.079m   1.444   1.388  40.966   4096K    4    1.759e-04
#  2.128m   1.485   1.485  34.067    3000    *         test
#  4.282m   1.380   1.315  42.729   8192K    8    1.275e-05
#  4.333m   1.490   1.490  34.367    3000    *         test
#  8.742m   1.338   1.296  43.323     16M   16    1.000e-05
#  8.792m   1.432   1.432  37.033    3000    *         test
# 17.596m   1.309   1.280  43.998     32M   32    1.000e-05
# 17.647m   1.456   1.456  37.967    3000    *         test
# 35.961m   1.281   1.253  44.961     65M   65    1.000e-05
# 36.011m   1.398   1.398  37.833    3000    *         test
# 54.493m   1.264   1.232  45.747     99M   99    1.000e-05
# 54.542m   1.395   1.395  38.767    3000    *         test

