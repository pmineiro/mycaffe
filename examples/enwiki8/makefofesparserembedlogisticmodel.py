#! /usr/bin/env python

import caffe
import math
from math import sqrt
import h5py
import numpy as np
import sys
import time
import warnings
from scipy.sparse import csr_matrix

alpha=0.9
eta=10.0
etadecay=0.9998
weightdecay=1e-5

# UGH ... so much for DRY

lrs=dict()
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

np.random.seed(8675309)

h5f=h5py.File(sys.argv[1])
embedding=np.array(h5f['embedding'])
h5f.close()

vocabsize=80000
batchsize=6000 

windowsize=int(embedding.shape[0]/(vocabsize+2))
rawembeddingsize=int(embedding.shape[1]/windowsize)
embeddingsize=windowsize*rawembeddingsize
invocabsize=windowsize*(vocabsize+2)
outvocabsize=vocabsize+1
embedding=embedding.reshape(invocabsize,embeddingsize) # sanity check

pretrain=caffe.Net(sys.argv[2],sys.argv[3])
pretrain.set_phase_test()

net=caffe.Net(sys.argv[4])
#net.set_mode_gpu()
net.set_phase_train()

momnet = caffe.Net(sys.argv[4])
#momnet.set_mode_gpu()
momnet.set_phase_train()

for (layer,momlayer) in zip(net.layers,momnet.layers):
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    blob.data[:]=0.01*np.random.standard_normal(size=blob.data.shape)
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')

maxlinenum=int(sys.argv[5])

row=[]
col=[]
value=[]
labels=np.zeros(batchsize,dtype='f')
bindex=0
start=time.time()
numsinceupdates=0
numupdates=0
sumloss=0
sumsinceloss=0
nextprint=1

for passnum in range(2):
  with open(sys.argv[6],'r') as f:
    print "%10s\t%10s\t%10s\t%11s\t%11s"%("delta t","average","since","example","learning")
    print "%10s\t%10s\t%10s\t%11s\t%11s"%("","loss","last","counter","rate")
    
    linenum=0
    for line in f:
      linenum=linenum+1
      if linenum > maxlinenum:
        break

      yx=[word for word in line.split(' ')]
      labels[bindex]=int(yx[0])-2
    
      for word in yx[1:]:
        iv=[subword for subword in word.split(':')]
        row.append(bindex)
        col.append(int(iv[0])-1)
        value.append(float(iv[1]))
      
      bindex=bindex+1
    
      if bindex >= batchsize:
        sd=csr_matrix((value, (row, col)), shape=(batchsize,invocabsize), dtype='f')
        data=sd.dot(embedding).reshape(batchsize,1,1,embeddingsize)
        pretrain.set_input_arrays(data,labels)
        preres=pretrain.forward()
        net.set_input_arrays(preres['ip3'],labels)
        res=net.forward()
        sumloss+=res['loss'][0,0,0,0]
        sumsinceloss+=res['loss'][0,0,0,0]
        net.backward()
  
        for (name,layer,momlayer) in zip(net._layer_names,net.layers,momnet.layers):
          blobnum=0
          for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
            myeta=lrs[(name,blobnum)]*eta
            momblob.data[:]=alpha*momblob.data[:]+myeta*blob.diff
            blob.data[:]-=momblob.data[:]
            blob.data[:]=(1-weightdecay*myeta)*blob.data[:]
            blobnum=blobnum+1
    
        eta=eta*etadecay
        value=[]
        row=[]
        col=[]
        labels[:]=0
        bindex=0
        numupdates=numupdates+1
        numsinceupdates=numsinceupdates+1
        if numupdates >= nextprint:
          net.save(sys.argv[7]+"."+str(numupdates))
          now=time.time()
          print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
          nextprint=2*nextprint
          numsinceupdates=0
          sumsinceloss=0
  
now=time.time()
print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
net.save(sys.argv[7])

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');

#   delta t         average           since          example        learning
#                      loss            last          counter            rate
#     7.381         11.2894         11.2894             1500           9.998
#    14.362         11.2342         11.1789             3000           9.996
#    28.023         11.0291         10.8240             6000           9.992
#    55.038         10.4174          9.8057            12000         9.98401
#   108.869          9.6095          8.8017            24000         9.96805
#   216.483          8.8316          8.0536            48000          9.9362
#   428.370          8.1538          7.4761            96000          9.8728
#   853.607          7.5786          7.0034           192000         9.74722
#  1702.521          7.1346          6.6906           384000         9.50084
#  3394.884          6.8252          6.5159           768000         9.02659
#  6783.919          6.6008          6.3763          1536000         8.14794
# 13569.806          6.4500          6.2991          3072000         6.63889
# 27156.923          6.3499          6.2499          6144000         4.40748
