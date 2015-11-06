import bz2
import caffe
from caffe import layers as L
import h5py
import math
import numpy as np
import os
import random
import sys
import time
from scipy.sparse import csr_matrix

random.seed(69)
np.random.seed(8675309)

numtags=13039
numtokens=260544
windowsize=10
embedd=300
batchsize=3000

alpha=0.5 
eta=1e-2
etadecay=0.999
weightdecay=1e-5 

lrs=dict()
lrs['embedding']=0.1
lrs[('lastip',0)]=1
lrs[('lastip',1)]=1

def net(batchsize,embedd,windowsize,numtags):
  n = caffe.NetSpec()
  n.data, n.bogus = L.MemoryData(batch_size=batchsize, channels=(windowsize*embedd+numtags), height=1, width=1, ntop=2)
  n.features, n.labels = L.Slice(n.data,slice_param=dict(axis=1,slice_point=[windowsize*embedd]),ntop=2)
  n.lastip = L.InnerProduct(n.features, num_output=numtags)
  n.loss = L.SigmoidCrossEntropyLoss(n.lastip, n.labels)
  n.silence = L.Silence(n.bogus,ntop=0)
  return n.to_proto()

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
  if (n < 1000): 
    return "%4u"%(n) 
  elif (n < 1000*1000): 
    return "%4uK"%(n/1000) 
  elif (n < 1000*1000*1000): 
    return "%4uM"%(n/(1000*1000)) 
  else: 
    return "%4uB"%(n/(1000*1000*1000)) 



protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.embedding.h5f')

with open(protofilename,'w') as f:
  f.write("force_backward: true\n");
  f.write(str(net(batchsize,embedd,windowsize,numtags)))

caffe.set_mode_gpu()
solver = caffe.SGDSolver('pretrain_solver.prototxt')

for (layer,momlayer) in zip(solver.net.layers,solver.test_nets[0].layers):
  blobnum=0 
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    if blobnum == 0:
      blob.data[:]=0.1/math.sqrt(blob.data.shape[-1])*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:]=0
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')
    blobnum=blobnum+1 

embedding=(1.0/math.sqrt(embedd))*np.random.standard_normal(size=(embedd,numtokens+1)).astype(float)
momembeddiff=np.zeros(shape=(embedd,numtokens+1),dtype='f')

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 
 
print "%7s\t%8s\t%8s\t%7s\t%11s"%("delta t","average","since","example","learning") 
print "%7s\t%8s\t%8s\t%7s\t%11s"%("","loss","last","counter","rate") 

for passes in range(2):
  bindex=0
  start=time.time()
  numsinceupdates=0 
  numupdates=0 
  sumloss=0 
  sumsinceloss=0 
  nextprint=1 

  with bz2.BZ2File('pretrain.bz2', 'rb') as inputfile:
    comboinputs=np.zeros((batchsize,windowsize*embedd+numtags,1,1),dtype='f')
    bogus=np.zeros((batchsize,1,1,1),dtype='f')
    batchtokens=np.zeros((batchsize,windowsize),dtype='i')
  
    for line in inputfile:
      yx=[word for word in line.split('\t')]
  
      tokens=yx[1].split(' ')
  
      if (len(tokens) < windowsize):
        continue
  
      tokstart=random.randint(0,len(tokens)-windowsize)
  
      for ii in range(windowsize):
        batchtokens[bindex,ii]=int(tokens[tokstart+ii])
        comboinputs[bindex,ii*embedd:(ii+1)*embedd,0,0]=embedding[:,int(tokens[tokstart+ii])]
  
      comboinputs[bindex,(windowsize*embedd):-1,0,0]=0
      for l in yx[0].split(','):
        comboinputs[bindex,(windowsize*embedd)+int(l)-1,0,0]=1
  
      bindex+=1
  
      if bindex >= batchsize:
        solver.net.set_input_arrays(comboinputs,bogus)
        res=solver.net.forward()
        sumloss+=res['loss']
        sumsinceloss+=res['loss']
        solver.net.backward()
        data_diff=solver.net.blobs['data'].diff.reshape(batchsize,windowsize*embedd+numtags,1,1)
  
        momembeddiff*=alpha;
        for ii in range(batchsize):
          for jj in range(windowsize):
            momembeddiff[:,batchtokens[ii,jj]]+=lrs['embedding']*eta*data_diff[ii,jj*embedd:(jj+1)*embedd,0,0]
        embedding-=momembeddiff
        embedding*=(1-lrs['embedding']*weightdecay*eta)
  
        for (name,layer,momlayer) in zip(solver.net._layer_names,solver.net.layers,solver.test_nets[0].layers): 
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
          print "%7s\t%8.3f\t%8.3f\t%7s\t%11.6g"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),eta) 
          nextprint=2*nextprint 
          numsinceupdates=0 
          sumsinceloss=0 


solver.net.save(netfilename)
h5f=h5py.File(embeddingfilename);
h5f.create_dataset('embedding',data=embedding) 
h5f.close() 
now=time.time() 
print "%7s\t%8.3f\t%8.3f\t%7s\t%11.6g"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),eta) 
