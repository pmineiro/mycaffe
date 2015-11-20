import bz2
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import math
import numpy as np
import os
import random
import sys
import time
from scipy.sparse import csr_matrix

import CaffeFinetuner 

import pdb

maxtags=13039
numtags=100
numtokens=260544
windowsize=10
embedd=300
batchsize=5001
numconvk=100

alpha=0.0
eta=1e-3
etadecay=0.9999
weightdecay=1e-6
labelnoise=1e-6

lrs=dict()
lrs['embedding']=10
lrs[('conv1',0)]=5
lrs[('conv1',1)]=10
lrs[('conv2',0)]=4
lrs[('conv2',1)]=8
lrs[('conv3',0)]=3
lrs[('conv3',1)]=6
lrs[('ip1',0)]=1
lrs[('ip1',1)]=2
lrs[('ip2',0)]=1
lrs[('ip2',1)]=2
lrs[('lastip',0)]=1
lrs[('lastip',1)]=0

random.seed(69)
np.random.seed(8675309)

#-------------------------------------------------
# beautification
#-------------------------------------------------

def nicetime(dt): 
   if (dt < 1): 
     return "%4.0fms"%(1000.0*dt) 
   elif (dt < 60): 
     return "%2.3fs"%(dt) 
   elif (dt < 60*60): 
     return "%2.3fm"%(dt/60) 
   elif (dt < 60*60*24): 
     return "%2.3fh"%(dt/(60*60)) 
   else: 
     return "%2.4fd"%(dt/(60*60*24)) 
 
def nicecount(n): 
  if (n < 10*1000): 
    return "%4u"%(n) 
  elif (n < 10*1000*1000): 
    return "%4uK"%(n/1000) 
  elif (n < 10*1000*1000*1000): 
    return "%4uM"%(n/(1000*1000)) 
  else: 
    return "%4uB"%(n/(1000*1000*1000)) 

#-------------------------------------------------
# define model
#-------------------------------------------------

def net(batchsize,embedd,windowsize,numtags,numconvk):
  n = caffe.NetSpec()
  n.data, n.bogus = L.MemoryData(batch_size=batchsize, channels=(windowsize*embedd+numtags), height=1, width=1, ntop=2)
  n.prefeatures, n.labels = L.Slice(n.data, slice_param=dict(axis=1,slice_point=[windowsize*embedd]),ntop=2)
  n.features = L.Reshape(n.prefeatures, reshape_param=dict(shape=dict(dim=[0,embedd,1,windowsize])))
  n.conv1 = L.Convolution(n.features, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool1 = L.Pooling(n.conv1, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.conv2 = L.Convolution(n.pool1, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool2 = L.Pooling(n.conv2, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.conv3 = L.Convolution(n.pool2, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool3 = L.Pooling(n.conv3, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.ip1 = L.InnerProduct(n.pool3, num_output=200)
  n.relu1 = L.ReLU(n.ip1, in_place=True)
  n.ip2 = L.InnerProduct(n.ip1, num_output=200)
  n.relu2 = L.ReLU(n.ip2, in_place=True)
  n.lastip = L.InnerProduct(n.relu2, num_output=numtags)
  n.loss = L.SigmoidCrossEntropyLoss(n.lastip, n.labels)
  n.silence = L.Silence(n.bogus,ntop=0)
  return n.to_proto()

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.embedding.h5f')

with open(protofilename,'w') as f:
  f.write("force_backward: true\n")
  f.write(str(net(batchsize,embedd,windowsize,numtags,numconvk)))

caffe.set_mode_gpu()
solver = caffe.SGDSolver('pretrain_solver.prototxt')
if (alpha > 0):
  momsolver = caffe.SGDSolver('pretrain_solver.prototxt')

#-------------------------------------------------
# initialize
#-------------------------------------------------

for layer in solver.net.layers:
  blobnum=0 
  for blob in layer.blobs:
    if blobnum == 0:
      blob.data[:]=(1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:]=0
    blobnum=blobnum+1 

if (alpha > 0):
  for layer in momsolver.net.layers:
    for blob in layer.blobs:
      blob.data[:]=0

# prior for bias

labelcounts=np.zeros(maxtags,dtype='d')
examplecount=0

try:
  with open('pretrain.labelcounts.txt', 'r') as f:
    print "using precomputed label counts"
    for line in f:
      lc=[word for word in line.split('\t')]
      labelcounts[int(lc[0])]=float(lc[1])
except:
  print "computing label counts"
  with bz2.BZ2File('pretrain.bz2', 'rb') as inputfile:
    for line in inputfile:
      yx=[word for word in line.split('\t')]
  
      tokens=yx[1].split(' ')
  
      if (len(tokens) < windowsize):
        continue
  
      for l in yx[0].split(','):
        labelcounts[int(l)-1]+=1
  
      examplecount+=1

  labelcounts=(1+labelcounts)/examplecount

  print "saving label counts"
  with open('pretrain.labelcounts.txt', 'w') as f:
    for ii in range(maxtags):
      f.write('%u\t%g\n'%(ii,labelcounts[ii]))

for (name,layer) in zip(solver.net._layer_names,solver.net.layers):
  blobnum=0 
  for blob in layer.blobs:
    if name == "lastip" and blobnum == 1:
      blob.data[:]=np.log(np.divide(labelcounts[0:numtags],1-labelcounts[0:numtags]))
    blobnum=blobnum+1 

embedding=(1.0/math.sqrt(embedd))*np.random.standard_normal(size=(embedd,numtokens+1)).astype(float)
if (alpha > 0):
  momembeddiff=np.zeros(shape=(embedd,numtokens+1),dtype='f')

params = { 'lrs': lrs, 'solver': solver, 'embedding': embedding, 
           'windowsize': windowsize, 'embedd': embedd, 'numtags': numtags, 
           'batchsize': batchsize, 'labelnoise': labelnoise, 
           'alpha': alpha, 'eta': eta, 'weightdecay': weightdecay }

if alpha > 0:
  params['momsolver'] = momsolver
  params['momembeddiff'] = momembeddiff

finetuner = CaffeFinetuner.CaffeFinetuner (params)
                             
#-------------------------------------------------
# iterate
#-------------------------------------------------

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 
 
print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta t","average","since","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("","loss","last","counter","num","rate") 

start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
nextprint=1 

for passes in range(24):
  batchnum=0
  with bz2.BZ2File('pretrain.bz2', 'rb') as inputfile:
    for line in inputfile:
      yx=[word for word in line.split('\t')]
  
      tokens=[int(word) for word in yx[1].split(' ')]
  
      if (len(tokens) < windowsize):
        continue
  
      tokstart=random.randint(0,len(tokens)-windowsize)
      tokens=tokens[tokstart:tokstart+windowsize]

      labels=[int(word) for word in yx[0].split(',')]
  
      rv = finetuner.update (tokens, labels)

      if rv[0]:
        sumloss+=rv[1]
        sumsinceloss+=rv[1]

        numupdates+=1
        numsinceupdates+=1
        finetuner.eta*=etadecay

        if numupdates >= nextprint:
          try:
            os.remove(netfilename+"."+str(numupdates)) 
          except:
            pass
            
          solver.net.save(netfilename+"."+str(numupdates)) 
          try:
            os.remove(embeddingfilename+"."+str(numupdates)) 
          except:
            pass

          h5f=h5py.File(embeddingfilename+"."+str(numupdates)) 
          h5f.create_dataset('embedding',data=embedding) 
          h5f.close() 
          now=time.time() 
          print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 
          nextprint=2*nextprint 
          numsinceupdates=0 
          sumsinceloss=0 

solver.net.save(netfilename)
try:
  os.remove(embeddingfilename)
except:
  pass
h5f=h5py.File(embeddingfilename)
h5f.create_dataset('embedding',data=embedding) 
h5f.close() 
now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./pretrain.py
# using precomputed label counts
# delta t average   since example pass     learning
#            loss    last counter  num         rate
#  1.935s   2.336   2.336    5001    0    9.999e-04
#  3.834s   2.348   2.360     10K    0    9.998e-04
#  7.025s   2.320   2.292     20K    0    9.996e-04
# 12.678s   2.304   2.289     40K    0    9.992e-04
# 23.392s   2.283   2.261     80K    0    9.984e-04
# 44.113s   2.274   2.265    160K    0    9.968e-04
#  1.421m   2.268   2.263    320K    0    9.936e-04
#  2.767m   2.270   2.272    640K    0    9.873e-04
#  5.430m   2.271   2.272   1280K    0    9.747e-04
# 10.685m   2.270   2.269   2560K    1    9.501e-04
# 21.165m   2.255   2.241   5121K    2    9.027e-04
# 42.023m   2.218   2.180     10M    4    8.148e-04
#  1.395h   2.160   2.101     20M    9    6.639e-04
#  2.784h   2.102   2.044     40M   19    4.408e-04
#  3.471h   2.086   2.024     51M   23    3.602e-04
