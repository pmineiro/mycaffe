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

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./fm.py
#   delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
#   113ms   1.609   1.609  21.200    1000    0    9.990e-03
#   162ms   1.609   1.608  24.200    2000    0    9.980e-03
#   250ms   1.607   1.606  27.100    4000    0    9.960e-03
#   418ms   1.603   1.599  30.625    8000    0    9.920e-03
#   749ms   1.585   1.567  35.137     16K    0    9.841e-03
#  1.459s   1.544   1.502  34.513     32K    0    9.685e-03
#  2.924s   1.503   1.462  35.175     64K    0    9.380e-03
#  5.822s   1.533   1.563  32.869    128K    0    8.798e-03
# 11.283s   1.545   1.556  34.543    256K    0    7.740e-03
# 22.554s   1.530   1.515  33.366    512K    0    5.991e-03
# 44.087s   1.516   1.501  34.312   1024K    1    3.590e-03
# 47.185s   1.609   1.609  31.033    3000    *         test
#  1.472m   1.511   1.507  38.649   2048K    2    1.289e-03
#  1.523m   1.589   1.589  36.267    3000    *         test
#  2.792m   1.497   1.483  39.859   4096K    4    1.661e-04
#  2.845m   1.447   1.447  36.033    3000    *         test
#  5.443m   1.420   1.342  42.231   8192K    8    2.857e-06
#  5.494m   1.469   1.469  37.700    3000    *         test
# 10.492m   1.356   1.293  43.313     16M   16    1.008e-07
# 10.542m   1.423   1.423  36.567    3000    *         test
# 20.425m   1.322   1.289  43.331     32M   32    1.000e-07
# 20.473m   1.432   1.432  36.667    3000    *         test
# 40.254m   1.304   1.286  43.408     65M   65    1.000e-07
# 40.304m   1.435   1.435  37.333    3000    *         test
#  1.015h   1.297   1.284  43.622     99M   99    1.000e-07
#  1.016h   1.419   1.419  37.333    3000    *         test
