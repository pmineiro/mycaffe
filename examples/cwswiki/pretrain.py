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

import pdb

maxtags=13039
numtags=10
numtokens=260544
windowsize=10
embedd=300
batchsize=3001
numconvk=100

alpha=0.0
eta=1e-3
etadecay=0.9999
weightdecay=1e-6
labelnoise=1e-6

lrs=dict()
lrs['embedding']=1
lrs[('conv1',0)]=1
lrs[('conv1',1)]=2
lrs[('conv2',0)]=1
lrs[('conv2',1)]=2
lrs[('conv3',0)]=1
lrs[('conv3',1)]=2
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
  f.write("force_backward: true\n");
  f.write(str(net(batchsize,embedd,windowsize,numtags,numconvk)))

caffe.set_mode_gpu()
solver = caffe.SGDSolver('pretrain_solver.prototxt')
momsolver = caffe.SGDSolver('pretrain_solver.prototxt')

#-------------------------------------------------
# initialize
#-------------------------------------------------

for (name,layer,momlayer) in zip(solver.net._layer_names,solver.net.layers,momsolver.net.layers):
  blobnum=0 
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    if blobnum == 0:
      blob.data[:]=(1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:]=0
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')
    blobnum=blobnum+1 

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

## hellinger pca
#
#row=[]
#col=[]
#val=[]
#with bz2.BZ2File('tokencooc.bz2', 'rb') as f:
#  for line in f:
#    lc=[word for word in line.split('\t')]
#    for cooc in lc[1:]:
#      indc=[word for word in cooc.split(':')]
#      row.append(int(lc[0]));
#      col.append(int(indc[0]));
#      val.append(float(indc[1]));
#
#  pdb.set_trace()
#  X=csr_matrix(val,(row,col),shape=(max(row),max(col)),dtype='d')
#  pdb.set_trace()
#  XtX=X*X.transpose().todense()
#  pdb.set_trace()
#  S,V=np.linalg.eigh(XtX)
#  pdb.set_trace()

embedding=(1.0/math.sqrt(embedd))*np.random.standard_normal(size=(embedd,numtokens+1)).astype(float)
momembeddiff=np.zeros(shape=(embedd,numtokens+1),dtype='f')

#-------------------------------------------------
# iterate
#-------------------------------------------------

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 
 
print "%7s\t%8s\t%8s\t%7s\t%4s\t%11s"%("delta t","average","since","example","pass","learning") 
print "%7s\t%8s\t%8s\t%7s\t%4s\t%11s"%("","loss","last","counter","num","rate") 

bindex=0
start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
nextprint=1 

comboinputs=np.zeros((batchsize,windowsize*embedd+numtags,1,1),dtype='f')
bogus=np.zeros((batchsize,1,1,1),dtype='f')
batchtokens=np.zeros((batchsize,windowsize),dtype='i')
  
for passes in range(64):
  with bz2.BZ2File('pretrain.bz2', 'rb') as inputfile:
    for line in inputfile:
      yx=[word for word in line.split('\t')]
  
      tokens=yx[1].split(' ')
  
      if (len(tokens) < windowsize):
        continue
  
      tokstart=random.randint(0,len(tokens)-windowsize)
  
      for ii in range(windowsize):
        batchtokens[bindex,ii]=int(tokens[tokstart+ii])
        comboinputs[bindex,ii*embedd:(ii+1)*embedd,0,0]=embedding[:,int(tokens[tokstart+ii])]
  
      comboinputs[bindex,(windowsize*embedd):,0,0]=labelnoise
      for l in yx[0].split(','):
        if int(l) <= numtags:
          comboinputs[bindex,(windowsize*embedd)+int(l)-1,0,0]=1.0-labelnoise
  
      bindex+=1
  
      if bindex >= batchsize:
        solver.net.set_input_arrays(comboinputs,bogus)
        res=solver.net.forward()
        sumloss+=res['loss']
        sumsinceloss+=res['loss']
        saveresloss=res['loss']+0
        solver.net.backward()
        data_diff=solver.net.blobs['data'].diff.reshape(batchsize,windowsize*embedd+numtags,1,1)

        # TODO: scale learning rate by token frequency ...
  
        momembeddiff*=alpha;
        for ii in range(batchsize):
          for jj in range(windowsize):
            momembeddiff[:,batchtokens[ii,jj]]+=lrs['embedding']*eta*data_diff[ii,jj*embedd:(jj+1)*embedd,0,0]
        embedding-=momembeddiff
        embedding*=(1-lrs['embedding']*weightdecay*eta)
  
        for (name,layer,momlayer) in zip(solver.net._layer_names,solver.net.layers,momsolver.net.layers):
           blobnum=0 
           for (blob,momblob) in zip(layer.blobs,momlayer.blobs): 
             myeta=lrs[(name,blobnum)]*eta 
             momblob.data[:]*=alpha
             momblob.data[:]+=myeta*blob.diff
             blob.data[:]-=momblob.data[:] 
             blob.data[:]*=(1-weightdecay*myeta)
             blobnum=blobnum+1 

        eta=eta*etadecay
        bindex=0
        numupdates+=1
        numsinceupdates+=1
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
          print "%7s\t%8.3f\t%8.3f\t%7s\t%4u\t%11.4g"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 
          nextprint=2*nextprint 
          numsinceupdates=0 
          sumsinceloss=0 


solver.net.save(netfilename)
try:
  os.remove(embeddingfilename)
except:
  pass
h5f=h5py.File(embeddingfilename);
h5f.create_dataset('embedding',data=embedding) 
h5f.close() 
now=time.time() 
print "%7s\t%8.3f\t%8.3f\t%7s\t%4u\t%11.6g"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./pretrain.py
# using precomputed label counts
# delta t  average           since        example pass       learning
#             loss            last        counter  num           rate
#  1.891s    0.938           0.938           3001    0      0.0009999
#  3.663s    0.937           0.937           6002    0      0.0009998
#  6.529s    0.931           0.924            12K    0      0.0009996
# 11.384s    0.929           0.927            24K    0      0.0009992
# 20.341s    0.921           0.912            48K    0      0.0009984
# 37.720s    0.919           0.917            96K    0      0.0009968
#  1.179m    0.916           0.912           192K    0      0.0009936
#  2.279m    0.916           0.916           384K    0      0.0009873
#  4.425m    0.917           0.917           768K    0      0.0009747
#  8.702m    0.918           0.919          1536K    0      0.0009501
# 17.273m    0.914           0.910          3073K    1      0.0009027
# 34.322m    0.902           0.889          6146K    2      0.0008148
#  1.142h    0.885           0.869            12M    5      0.0006639
#  2.281h    0.863           0.841            24M   11      0.0004408
#  4.555h    0.841           0.818            49M   23      0.0001943
# ...
