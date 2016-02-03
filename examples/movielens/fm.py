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
#   105ms   1.609   1.609  22.500    1000    0    9.990e-03
#   153ms   1.609   1.609  33.600    2000    0    9.980e-03
#   252ms   1.608   1.606  35.050    4000    0    9.960e-03
#   464ms   1.603   1.598  33.275    8000    0    9.920e-03
#   905ms   1.585   1.567  35.537     16K    0    9.841e-03
#  1.814s   1.543   1.502  34.513     32K    0    9.685e-03
#  3.494s   1.502   1.461  35.175     64K    0    9.380e-03
#  6.854s   1.533   1.564  32.786    128K    0    8.798e-03
# 12.773s   1.545   1.557  34.587    256K    0    7.741e-03
# 24.892s   1.529   1.513  33.361    512K    0    5.992e-03
# 47.423s   1.517   1.506  34.682   1024K    1    3.590e-03
# 50.681s   1.546   1.546  31.167    3000    *         test
#  1.631m   1.559   1.600  37.799   2048K    2    1.289e-03
#  1.684m   2.034   2.034  33.767    3000    *         test
#  3.296m   1.531   1.504  40.027   4096K    4    1.670e-04
#  3.348m   1.511   1.511  33.467    3000    *         test
#  6.260m   1.428   1.324  42.652   8192K    8    3.757e-06
#  6.313m   1.437   1.437  35.067    3000    *         test
# 12.039m   1.360   1.292  43.382     16M   16    1.001e-06
# 12.092m   1.454   1.454  35.833    3000    *         test
# 23.761m   1.322   1.284  43.659     32M   32    1.000e-06
# 23.815m   1.427   1.427  37.267    3000    *         test
# 48.091m   1.297   1.273  44.075     65M   65    1.000e-06
# 48.147m   1.413   1.413  38.500    3000    *         test
#  1.240h   1.284   1.258  44.755     99M   99    1.000e-06
#  1.241h   1.389   1.389  39.200    3000    *         test
