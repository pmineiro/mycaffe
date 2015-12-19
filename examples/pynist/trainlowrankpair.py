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

import pdb

from Pretty import nicetime, nicecount

batchsize=32
eta=1e-3
etamin=1e-6
etadecay=0.99
weightdecay=0
rank=8

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
nets = [ caffe.Net(protofilename, caffe.TRAIN) for n in range(2) ]
layer_foreach(nets[0],xavier)
for layer,layer1 in zip(nets[0].layers,nets[1].layers):
  for blob,blob1 in zip(layer.blobs,layer1.blobs):
    blob1.data[:]=blob.data[:]

resnets = [ caffe.Net(protofilename, caffe.TRAIN) for n in range(2) ]
for resnet in resnets:
  layer_foreach (resnet, (lambda bn, b: xavier(1, b)))

perm=np.random.permutation(nimages)
subsetsize=int(math.floor(float(nimages)/2.0))
trains=[traindata[perm[i:i+subsetsize],:,:,:] for i in range(0,nimages,subsetsize)]
labels=[trainlabels[perm[i:i+subsetsize],:,:,] for i in range(0,nimages,subsetsize)]

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
subtrains=[ np.zeros((batchsize,1,28,28),dtype='f') for n in range(2) ]
sublabels=[ np.zeros((batchsize,1,1,1),dtype='f') for n in range(2) ]

for passes in range(1):
  perm=[np.random.permutation(subsetsize) for n in range(2)]

  for layer,layer1 in zip(nets[0].layers,nets[1].layers):
    for blob,blob1 in zip(layer.blobs,layer1.blobs):
      blob1.data[:]=blob.data[:]

  for pos in range(0, subsetsize, batchsize):
    if pos + batchsize > subsetsize:
      continue

    for n,net in enumerate(nets):
      subtrains[n][:,:,:,:]=trains[n][perm[n][pos:pos+batchsize],:,:,:]
      sublabels[n][:,:,:,:]=labels[n][perm[n][pos:pos+batchsize],:,:,:]
      net.set_input_arrays (subtrains[n], sublabels[n])

    res=[ net.forward() for net in nets ]
    sumloss+=np.sum([ r['acc'] for r in res ])
    sumsinceloss+=np.sum([ r['acc'] for r in res ])
    numupdates+=2
    numsinceupdates+=2

    for net in nets:
      net.backward()

    for net,resnet,net1,resnet1 in [(nets[0],resnets[0],nets[1],resnets[1])]:
      for layer,reslayer,layer1,reslayer1 in zip(net.layers,resnet.layers,net1.layers,resnet1.layers):
        for blob,resblob,blob1,resblob1 in zip(layer.blobs,reslayer.blobs,layer1.blobs,reslayer1.blobs):
          if len(blob.diff.shape) == 2:
            resblob.data[:]+=blob.diff
            resblob1.data[:]+=blob1.diff
            u,s,v=mysvds(resblob.data,rank)
            u1,s1,v1=mysvds(resblob1.data,rank)
            ru,rs,rv=mysvds(0.5 * (u.dot(np.diag(s).dot(v.T)) +
                                   u1.dot(np.diag(s1).dot(v1.T))),
                            rank)
            resblob.data[:]-=ru.dot(np.diag(rs).dot(rv.T))
            resblob1.data[:]-=ru.dot(np.diag(rs).dot(rv.T))
            blob.data[:]-=eta*ru.dot(np.diag(rs).dot(rv.T))
            blob1.data[:]-=eta*ru.dot(np.diag(rs).dot(rv.T))
          else:
            blob.data[:]-=eta*(blob.diff+blob1.diff)/2.0
            blob1.data[:]-=eta*(blob.diff+blob1.diff)/2.0

          blob.data[:]-=eta*weightdecay*blob.data[:]
          blob1.data[:]-=eta*weightdecay*blob1.data[:]

    if numupdates >= nextprint:
      testloss=0
      numtestupdates=0

      for testpos in range(0, testnimages, batchsize):
        if testpos + batchsize > testnimages:
          continue

        subtrains[0][:,:,:,:]=testdata[testpos:testpos+batchsize,:,:,:]
        sublabels[0][:,:,:,:]=testlabels[testpos:testpos+batchsize,:,:,:]
        nets[0].set_input_arrays (subtrains[0], sublabels[0])
        res=nets[0].forward()
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

  subtrains[0][:,:,:,:]=testdata[testpos:testpos+batchsize,:,:,:]
  sublabels[0][:,:,:,:]=testlabels[testpos:testpos+batchsize,:,:,:]
  nets[0].set_input_arrays (subtrains[0], sublabels[0])
  res=nets[0].forward()
  testloss+=res['acc']
  numtestupdates+=1

now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,testloss/numtestupdates,nicecount(numupdates*batchsize),passes,eta) 

netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'trainlowrank.model')
net.save(netfilename)

