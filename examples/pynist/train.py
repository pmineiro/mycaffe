import caffe
from caffe import layers as L
from caffe import params as P
import gzip
import math
import numpy as np
import random
import struct
import os
import sys
import time

from Pretty import nicetime, nicecount

batchsize=64
eta=1e-3
etamin=5e-7
etadecay=0.999
alpha=0.0
weightdecay=4

random.seed(69)
np.random.seed(8675309)

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

def xavier(blobnum, blob):
  if blobnum == 0:
    blob.data[:]=(1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape) 
  else:
    blob.data[:]=0

def layer_foreach(net, func):
  for layer in net.layers:
    for blobnum,blob in enumerate(layer.blobs):
      func(blobnum, blob)

def net():
  n = caffe.NetSpec()
  n.data, n.labels = L.MemoryData(batch_size=batchsize, channels=1, height=28, width=28, ntop=2)
  n.conv1 = L.Convolution(n.data, num_output=20, kernel_size=5, stride=1)
  n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
  n.ip1 = L.InnerProduct(n.pool1, num_output=500)
  n.relu1 = L.ReLU(n.ip1, in_place=True)
  n.lastip = L.InnerProduct(n.relu1, num_output=10)
  n.loss = L.SoftmaxWithLoss(n.lastip, n.labels)
  n.acc = L.Accuracy(n.lastip, n.labels)
  return n.to_proto()

with gzip.GzipFile('t10k-images-idx3-ubyte.gz') as f:
  magic, testnimages, testnrows, testncolumns = struct.unpack("!LLLL", f.read(16))
  assert (magic == 2051)

  testdata = np.zeros((testnimages,1,testnrows,testncolumns),dtype='f')

  for n in range(testnimages):
    testdata[n,0,:,:] = np.array(struct.unpack("%uB"%(testnrows*testncolumns), f.read(testnrows*testncolumns)),dtype='f').reshape(28,28)

  testdata /= 256.0

with gzip.GzipFile('t10k-labels-idx1-ubyte.gz') as f:
  magic, testnlabels = struct.unpack("!LL", f.read(8))
  assert (magic == 2049)
  assert (testnlabels == testnimages)

  testlabels = np.zeros((testnlabels,1,1,1),dtype='f')

  testlabels[:,0,0,0] = np.array(struct.unpack("%uB"%testnlabels, f.read(testnlabels)),dtype='f')

with gzip.GzipFile('train-images-idx3-ubyte.gz') as f:
  magic, nimages, nrows, ncolumns = struct.unpack("!LLLL", f.read(16))
  assert (magic == 2051)

  traindata = np.zeros((nimages,1,nrows,ncolumns),dtype='f')

  for n in range(nimages):
    traindata[n,0,:,:] = np.array(struct.unpack("%uB"%(nrows*ncolumns), f.read(nrows*ncolumns)),dtype='f').reshape(28,28)

  traindata /= 256.0

with gzip.GzipFile('train-labels-idx1-ubyte.gz') as f:
  magic, nlabels = struct.unpack("!LL", f.read(8))
  assert (magic == 2049)
  assert (nlabels == nimages)

  trainlabels = np.zeros((nlabels,1,1,1),dtype='f')

  trainlabels[:,0,0,0] = np.array(struct.unpack("%uB"%nlabels, f.read(nlabels)),dtype='f')

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train.prototxt')
with open(protofilename,'w') as f:
  f.write(str(net()))

caffe.set_mode_cpu()
net = caffe.Net(protofilename, caffe.TRAIN)
layer_foreach (net, xavier)

if alpha > 0:
  momnet = caffe.Net(protofilename, caffe.TRAIN)
  layer_foreach (momnet, lambda blobnum, blob: xavier(1,blob))

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta t","average","since","holdout","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("","loss","last","loss","counter","num","rate") 

start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
nextprint=1 

# NB: if i don't make these permanent arrays to hold the input,
# memory leaks until the script crashes
subtrain=np.zeros((batchsize,1,28,28),dtype='f')
sublabel=np.zeros((batchsize,1,1,1),dtype='f')

for passes in range(60):
  perm=np.random.permutation(nimages)

  for pos in range(0, nimages, batchsize):
    if pos + batchsize > nimages:
      continue

    subtrain[:,:,:,:]=traindata[perm[pos:pos+batchsize],:,:,:]
    sublabel[:,:,:,:]=trainlabels[perm[pos:pos+batchsize],:,:,:]
    net.set_input_arrays (subtrain, sublabel)
    res=net.forward()
    sumloss+=res['acc']
    sumsinceloss+=res['acc']
    numupdates+=1
    numsinceupdates+=1

    net.backward()

    if alpha > 0:
      for (layer,momlayer) in zip(net.layers,momnet.layers):
        for (blob,momblob) in zip(layer.blobs,momlayer.blobs): 
          momblob.data[:]*=alpha
          momblob.data[:]+=eta*blob.diff
          momblob.data[:]+=eta*weightdecay*blob.data[:]
          blob.data[:]-=momblob.data
    else:
      for layer in net.layers:
        for blob in layer.blobs:
          blob.data[:]-=eta*blob.diff
          blob.data[:]-=eta*weightdecay*blob.data[:]

    if numupdates >= nextprint:
      testloss=0
      numtestupdates=0

      for testpos in range(0, testnimages, batchsize):
        if testpos + batchsize > testnimages:
          continue

        subtrain[:,:,:,:]=testdata[testpos:testpos+batchsize,:,:,:]
        sublabel[:,:,:,:]=testlabels[testpos:testpos+batchsize,:,:,:]
        net.set_input_arrays (subtrain, sublabel)
        res=net.forward()
        testloss+=res['acc']
        numtestupdates+=1

      now=time.time() 
      print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,testloss/numtestupdates,nicecount(numupdates*batchsize),passes,eta) 
      nextprint=2*nextprint 
      numsinceupdates=0 
      sumsinceloss=0 

    eta=eta*etadecay+etamin*(1.0-etadecay)


testloss=0
numtestupdates=0

for testpos in range(0, testnimages, batchsize):
  if testpos + batchsize > testnimages:
    continue

  subtrain[:,:,:,:]=testdata[testpos:testpos+batchsize,:,:,:]
  sublabel[:,:,:,:]=testlabels[testpos:testpos+batchsize,:,:,:]
  net.set_input_arrays (subtrain, sublabel)
  res=net.forward()
  testloss+=res['acc']
  numtestupdates+=1

now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,testloss/numtestupdates,nicecount(numupdates*batchsize),passes,eta) 

netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train.model')
net.save(netfilename)

# delta t average   since holdout example pass     learning
#            loss    last    loss counter  num         rate
#  1.785s   0.047   0.047   0.081      64    0    1.000e-03
#  3.538s   0.070   0.094   0.085     128    0    9.990e-04
#  5.241s   0.094   0.117   0.104     256    0    9.970e-04
#  7.072s   0.113   0.133   0.207     512    0    9.930e-04
#  8.964s   0.210   0.307   0.437    1024    0    9.851e-04
# 11.088s   0.383   0.557   0.615    2048    0    9.695e-04
# 13.580s   0.561   0.738   0.808    4096    0    9.389e-04
# 16.907s   0.692   0.823   0.852    8192    0    8.807e-04
# 22.238s   0.770   0.847   0.853     16K    0    7.749e-04
# 31.974s   0.828   0.887   0.926     32K    0    5.999e-04
# 47.012s   0.884   0.939   0.960     65K    1    3.597e-04
#  1.264m   0.927   0.971   0.980    131K    2    1.294e-04
#  2.202m   0.958   0.988   0.986    262K    4    1.711e-05
#  3.996m   0.975   0.993   0.987    524K    8    7.759e-07
#  7.475m   0.985   0.995   0.988   1048K   17    5.001e-07
# 14.504m   0.991   0.997   0.989   2097K   34    5.000e-07
# 24.505m   0.995   0.999   0.988   3598K   59    5.000e-07
# 
