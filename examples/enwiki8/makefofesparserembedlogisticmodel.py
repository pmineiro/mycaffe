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
eta=1.0
etadecay=0.99999
weightdecay=1e-5

# UGH ... so much for DRY

lrs=dict()
lrs[('ip3',0)]=1
lrs[('ip3',1)]=2

np.random.seed(8675309)

h5f=h5py.File(sys.argv[1])
embedding=np.array(h5f['embedding'])
h5f.close()

vocabsize=80000
batchsize=1500 

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

with open(sys.argv[6],'r') as f:
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
  
  print "%10s\t%10s\t%10s\t%11s\t%11s"%("delta t","average","since","example","learning")
  print "%10s\t%10s\t%10s\t%11s\t%11s"%("","loss","last","counter","rate")
  
  for line in f:
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

# GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparsemodel.py fofe_sparse_small_unigram_train <(head -n `wc -l fofengram9.txt | perl -lane 'print int(0.9*$F[0])'` fofengram9.txt) fofesparsemodel9
#    delta t         average           since          example        learning
#                       loss            last          counter            rate
#      3.324         11.2893         11.2893             1500         0.99999
#      6.609         11.2832         11.2772             3000         0.99998
#     12.333         11.2623         11.2413             6000         0.99996
#     22.722         11.1827         11.1032            12000         0.99992
#     42.765         10.7750         10.3673            24000         0.99984
#     81.470         10.0174          9.2598            48000         0.99968
#    158.956          9.0810          8.1445            96000         0.99936
#    307.732          8.1580          7.2351           192000        0.998721
#    598.225          7.4279          6.6979           384000        0.997443
#   1180.870          6.8014          6.1749           768000        0.994893
#   2646.052          6.3289          5.8563          1536000        0.989812
#   6218.354          5.9774          5.6260          3072000        0.979728
#  14266.073          5.7044          5.4313          6144000        0.959867
#  31046.187          5.4776          5.2508         12288000        0.921345

