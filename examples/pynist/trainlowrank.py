import caffe
from caffe import layers as L
from caffe import params as P
import gzip
import math
import numpy as np
import random
import scipy.sparse.linalg
import struct
import os
import sys
import time

from Pretty import nicetime, nicecount

import pdb

batchsize=64
eta=1e-6
etamin=1e-7
etadecay=0.9999
alpha=0.9
weightdecay=0

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
#  megu,megs,megv=scipy.sparse.linalg.svds(A,1)
#  V=V.T
#  print np.max(np.abs(V[0:1,:]-megv))
#  print np.max(np.abs(U[:,0:1]-megu))
  return (U,s,V)

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
momnet = caffe.Net(protofilename, caffe.TRAIN)
resnet = caffe.Net(protofilename, caffe.TRAIN)

for (layer,momlayer,reslayer) in zip(net.layers,momnet.layers,resnet.layers):
  for blobnum,(blob,momblob,resblob) in enumerate(zip(layer.blobs,momlayer.blobs,reslayer.blobs)):
    if blobnum == 0:
      blob.data[:]=(1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:]=0
    momblob.data[:]=0
    resblob.data[:]=0


#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta t","average","since","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("","loss","last","counter","num","rate") 

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

    for (layer,momlayer,reslayer) in zip(net.layers,momnet.layers,resnet.layers):
      for (blob,momblob,resblob) in zip(layer.blobs,momlayer.blobs,reslayer.blobs): 
        momblob.data[:]*=alpha

        if len(blob.diff.shape) == 2:
          resblob.data[:]+=blob.diff
          #u,s,v=scipy.sparse.linalg.svds(resblob.data,1,tol=1e-4,maxiter=1000)
          u,s,v=mysvds(resblob.data,1)
          momblob.data[:]+=eta*s*np.outer(u,v)
          resblob.data[:]-=s*np.outer(u,v)
        else:
          momblob.data[:]+=eta*blob.diff
        blob.data[:]-=momblob.data
        blob.data[:]*=(1.0-weightdecay*eta)

    if numupdates >= nextprint:
      now=time.time() 
      print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 
      nextprint=2*nextprint 
      numsinceupdates=0 
      sumsinceloss=0 

    eta=eta*etadecay+etamin*(1.0-etadecay)


now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 

netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'trainlowrank.model')
net.save(netfilename)

# delta t average   since example pass     learning
#            loss    last counter  num         rate
#   190ms   0.047   0.047      64    0    1.000e-06
#   516ms   0.070   0.094     128    0    9.999e-07
#  1.120s   0.090   0.109     256    0    9.997e-07
#  2.447s   0.078   0.066     512    0    9.994e-07
#  4.608s   0.075   0.072    1024    0    9.987e-07
#  9.480s   0.077   0.078    2048    0    9.972e-07
# 19.339s   0.084   0.091    4096    0    9.943e-07
# 37.113s   0.148   0.211    8192    0    9.886e-07
#  1.167m   0.306   0.464     16K    0    9.773e-07
#  2.370m   0.522   0.737     32K    0    9.552e-07
#  4.802m   0.670   0.819     65K    1    9.125e-07
#  9.691m   0.728   0.787    131K    2    8.334e-07
# 19.447m   0.748   0.767    262K    4    6.976e-07
# 39.169m   0.818   0.889    524K    8    4.967e-07
#  1.144h   0.876   0.934   1048K   17    2.749e-07
#  1.847h   0.920   0.963   2097K   34    1.340e-07
#  2.849h   0.943   0.977   3598K   59    1.033e-07

# make testlowrank
# delta t average   since example pass     learning
#            loss    last counter  num         rate
#    15ms   0.953   0.953      64    0    1.000e-06
#    29ms   0.953   0.953     128    0    1.000e-06
#    55ms   0.965   0.977     256    0    1.000e-06
#   104ms   0.959   0.953     512    0    1.000e-06
#   205ms   0.964   0.969    1024    0    1.000e-06
#   383ms   0.958   0.951    2048    0    1.000e-06
#   756ms   0.955   0.952    4096    0    1.000e-06
#  1.584s   0.963   0.971    8192    0    1.000e-06
#  1.999s   0.966   0.982    9984    0    1.000e-06
