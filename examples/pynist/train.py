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
etamin=1e-6
etadecay=0.999
alpha=0.0
weightdecay=1

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
#  1.814s   0.047   0.047   0.081      64    0    1.000e-03
#  3.657s   0.070   0.094   0.085     128    0    9.990e-04
#  5.532s   0.094   0.117   0.104     256    0    9.970e-04
#  7.486s   0.113   0.133   0.208     512    0    9.930e-04
#  9.542s   0.210   0.307   0.443    1024    0    9.851e-04
# 11.798s   0.388   0.566   0.644    2048    0    9.695e-04
# 14.380s   0.576   0.763   0.826    4096    0    9.390e-04
# 17.840s   0.707   0.838   0.859    8192    0    8.808e-04
# 22.856s   0.776   0.845   0.813     16K    0    7.750e-04
# 31.381s   0.830   0.884   0.882     32K    0    6.001e-04
# 46.391s   0.878   0.927   0.939     65K    1    3.600e-04
#  1.242m   0.920   0.963   0.971    131K    2    1.299e-04
#  2.150m   0.951   0.982   0.981    262K    4    1.761e-05
#  3.940m   0.972   0.992   0.984    524K    8    1.276e-06
#  7.516m   0.983   0.995   0.986   1048K   17    1.000e-06
# 14.601m   0.991   0.998   0.987   2097K   34    1.000e-06
# 24.915m   0.994   0.999   0.985   3598K   59    1.000e-06
