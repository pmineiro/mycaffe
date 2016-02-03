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
rank=2

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

# rank=4
#
# delta t average   since holdout example pass     learning
#            loss    last    loss counter  num         rate
#  1.896s   0.047   0.047   0.081      64    0    1.000e-03
#  3.827s   0.070   0.094   0.084     128    0    9.990e-04
#  5.762s   0.094   0.117   0.100     256    0    9.970e-04
#  7.846s   0.105   0.117   0.192     512    0    9.930e-04
# 10.467s   0.201   0.297   0.415    1024    0    9.851e-04
# 14.474s   0.368   0.534   0.594    2048    0    9.695e-04
# 20.912s   0.548   0.729   0.800    4096    0    9.389e-04
# 30.829s   0.680   0.812   0.833    8192    0    8.807e-04
# 49.176s   0.752   0.823   0.785     16K    0    7.749e-04
#  1.445m   0.818   0.884   0.903     32K    0    5.999e-04
#  2.709m   0.871   0.924   0.952     65K    1    3.597e-04
#  5.237m   0.918   0.964   0.970    131K    2    1.294e-04
# 10.215m   0.951   0.984   0.983    262K    4    1.711e-05
# 20.107m   0.971   0.992   0.985    524K    8    7.759e-07
# 39.880m   0.982   0.994   0.987   1048K   17    5.001e-07
#  1.314h   0.989   0.996   0.988   2097K   34    5.000e-07
#  2.238h   0.993   0.999   0.987   3598K   59    5.000e-07

# rank=2
#
# delta t average   since holdout example pass     learning
#            loss    last    loss counter  num         rate
#  1.928s   0.047   0.047   0.081      64    0    1.000e-03
#  3.756s   0.070   0.094   0.082     128    0    9.990e-04
#  5.640s   0.090   0.109   0.095     256    0    9.970e-04
#  7.715s   0.098   0.105   0.176     512    0    9.930e-04
# 10.333s   0.191   0.285   0.373    1024    0    9.851e-04
# 13.646s   0.349   0.507   0.566    2048    0    9.695e-04
# 19.719s   0.531   0.712   0.798    4096    0    9.389e-04
# 30.093s   0.662   0.794   0.804    8192    0    8.807e-04
# 48.715s   0.718   0.773   0.784     16K    0    7.749e-04
#  1.417m   0.797   0.875   0.879     32K    0    5.999e-04
#  2.605m   0.862   0.927   0.954     65K    1    3.597e-04
#  4.890m   0.913   0.963   0.968    131K    2    1.294e-04
#  9.430m   0.946   0.980   0.981    262K    4    1.711e-05
# 18.796m   0.968   0.990   0.985    524K    8    7.759e-07
# 37.330m   0.980   0.992   0.985   1048K   17    5.001e-07
#  1.240h   0.988   0.995   0.986   2097K   34    5.000e-07
#  2.120h   0.992   0.998   0.986   3598K   59    5.000e-07
