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
rank=4

random.seed(69)
np.random.seed(8675309)

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

def mysvds(A,k):
  p=10
  (n,d)=A.shape
  # A = U S V.T
  # A.T A = V S^2 V.T
  # U = A V Sinv
  Omega=np.random.randn(d,k+p)
  Z=A.T.dot(A.dot(Omega))
  Omega,_=np.linalg.qr(Z)
  Z=A.T.dot(A.dot(Omega))
  V,s,_=np.linalg.svd(Z,full_matrices=False)
  V=V[:,0:k]
  s=s[0:k:]**0.5
  U=A.dot(V.dot(np.diag([1.0/sk if sk > 1e-6 else 0 for sk in s])))
  return (U,s,V)

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
resnet = caffe.Net(protofilename, caffe.TRAIN)
layer_foreach (net, xavier)
layer_foreach (resnet, lambda blobnum, blob: xavier(1,blob))

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
      for (layer,momlayer,reslayer) in zip(net.layers,momnet.layers,resnet.layers):
        for (blob,momblob,resblob) in zip(layer.blobs,momlayer.blobs,reslayer.blobs): 
          momblob.data[:]*=alpha

          if len(blob.diff.shape) == 2:
            resblob.data[:]+=blob.diff
            u,s,v=mysvds(resblob.data,rank)
            momblob.data[:]+=eta*u.dot(np.diag(s).dot(v.T))
            resblob.data[:]-=u.dot(np.diag(s).dot(v.T))
          else:
            momblob.data[:]+=eta*blob.diff

          momblob.data[:]+=eta*weightdecay*blob.data[:]
          blob.data[:]-=momblob.data
    else:
      for layer,reslayer in zip(net.layers,resnet.layers):
        for blob,resblob in zip(layer.blobs,reslayer.blobs):
          if len(blob.diff.shape) == 2:
            resblob.data[:]+=blob.diff
            u,s,v=mysvds(resblob.data,rank)
            resblob.data[:]-=u.dot(np.diag(s).dot(v.T))
            blob.data[:]-=eta*u.dot(np.diag(s).dot(v.T))
          else:
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

netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'trainlowrank.model')
net.save(netfilename)

# rank=2
# delta t average   since holdout example pass     learning
#            loss    last    loss counter  num         rate
#  1.870s   0.047   0.047   0.081      64    0    1.000e-03
#  3.760s   0.070   0.094   0.082     128    0    9.990e-04
#  5.664s   0.090   0.109   0.095     256    0    9.970e-04
#  8.016s   0.098   0.105   0.177     512    0    9.930e-04
# 10.557s   0.191   0.285   0.381    1024    0    9.851e-04
# 14.516s   0.355   0.520   0.595    2048    0    9.695e-04
# 20.704s   0.547   0.739   0.807    4096    0    9.390e-04
# 30.909s   0.679   0.811   0.825    8192    0    8.808e-04
# 50.991s   0.710   0.742   0.815     16K    0    7.750e-04
#  1.429m   0.774   0.838   0.855     32K    0    6.001e-04
#  2.588m   0.831   0.888   0.907     65K    1    3.600e-04
#  4.911m   0.880   0.930   0.933    131K    2    1.299e-04
#  9.495m   0.920   0.960   0.960    262K    4    1.761e-05
# 18.688m   0.949   0.977   0.975    524K    8    1.276e-06
# 37.010m   0.967   0.985   0.978   1048K   17    1.000e-06
#  1.230h   0.979   0.990   0.980   2097K   34    1.000e-06
#  2.101h   0.985   0.995   0.981   3598K   59    1.000e-06
# 
# rank=4
# delta t average   since holdout example pass     learning
#            loss    last    loss counter  num         rate
#  1.824s   0.047   0.047   0.081      64    0    1.000e-03
#  3.649s   0.070   0.094   0.084     128    0    9.990e-04
#  5.573s   0.094   0.117   0.100     256    0    9.970e-04
#  7.699s   0.105   0.117   0.191     512    0    9.930e-04
# 10.260s   0.200   0.295   0.422    1024    0    9.851e-04
# 14.250s   0.374   0.548   0.630    2048    0    9.695e-04
# 20.181s   0.564   0.755   0.818    4096    0    9.390e-04
# 30.009s   0.697   0.829   0.849    8192    0    8.808e-04
# 49.437s   0.767   0.836   0.796     16K    0    7.750e-04
#  1.454m   0.823   0.880   0.910     32K    0    6.001e-04
#  2.684m   0.870   0.917   0.947     65K    1    3.600e-04
#  5.143m   0.911   0.952   0.957    131K    2    1.299e-04
#  9.962m   0.944   0.977   0.977    262K    4    1.761e-05
# 19.701m   0.966   0.989   0.984    524K    8    1.276e-06
# 38.989m   0.980   0.993   0.986   1048K   17    1.000e-06
#  1.292h   0.988   0.997   0.986   2097K   34    1.000e-06
#  2.210h   0.993   0.999   0.985   3598K   59    1.000e-06
