#! /usr/bin/env python

import caffe
import math
import numpy as np
import sys
import time
from scipy.sparse import csr_matrix

import pprint

alpha=0.9
eta=1.0
etadecay=0.999995
weightdecay=1e-4

# UGH ... so much for DRY

lrs=dict()
lrs['embedding']=0.75
lrs[('ip1',0)]=1
lrs[('ip1',1)]=1.5
lrs[('ip2',0)]=1
lrs[('ip2',1)]=1.5
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

#np.seterr(divide='raise',over='raise',invalid='raise')

np.random.seed(8675309)

vocabsize=80000
windowsize=2
rawembeddingsize=200
batchsize=1500 

embeddingsize=windowsize*rawembeddingsize
invocabsize=windowsize*(vocabsize+2)
outvocabsize=vocabsize+1

preembed=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')
preembed[:]=np.random.standard_normal(size=(invocabsize,embeddingsize))
embedding=math.sqrt(embeddingsize)*np.linalg.qr(preembed)[0]

net = caffe.Net(sys.argv[1])
net.set_mode_gpu()
net.set_phase_train()

momnet = caffe.Net(sys.argv[1])
momnet.set_mode_gpu()
momnet.set_phase_train()

for (layer,momlayer) in zip(net.layers,momnet.layers):
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    blob.data[:]=0.01*np.random.standard_normal(size=blob.data.shape)
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')

momembeddiff=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')

f=open(sys.argv[2],'r')

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

for ii in range(10):
  f.seek(0,0)
  for line in f:
      yx=[word for word in line.split(' ')]
      labels[bindex]=int(yx[0])-1
  
      for word in yx[1:]:
          iv=[subword for subword in word.split(':')]
          row.append(bindex)
          col.append(int(iv[0])-1)
          value.append(float(iv[1]))
      
      bindex=bindex+1
  
      if bindex >= batchsize:
          sd=csr_matrix((value, (row, col)), shape=(batchsize,invocabsize), dtype='f')
          data=sd.dot(embedding).reshape(batchsize,1,1,embeddingsize)
          net.set_input_arrays(data,labels)
          res=net.forward()
          sumloss+=res['loss'][0,0,0,0]
          sumsinceloss+=res['loss'][0,0,0,0]
          net.backward()
          data_diff=net.blobs['data'].diff.reshape(batchsize,embeddingsize)

          # y = W x
          # y_k = \sum_l W_kl x_l
          # df/dW_ij = \sum_k df/dy_k dy_k/dW_ij
          #          = \sum_k df/dy_k (\sum_l 1_{i=k} 1_{j=l} x_l)
          #          = \sum_k df/dy_k 1_{i=k} x_j
          #          = df/dy_i x_j
          # df/dW    = (df/dy)*x'

          sdtransdiff=sd.transpose().tocsr().dot(data_diff);

          momembeddiff=alpha*momembeddiff+lrs['embedding']*eta*sdtransdiff;
          embedding=embedding-momembeddiff
          embedding=(1-lrs['embedding']*weightdecay*eta)*embedding

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
              net.save(sys.argv[3]+"."+str(numupdates))
              now=time.time()
              print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
              nextprint=2*nextprint
              numsinceupdates=0
              sumsinceloss=0


now=time.time()
print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
net.save(sys.argv[3])

# GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparsemodel.py fofe_sparse_small_unigram_train fofengram9.txt
#    delta t         average           since          example        learning
#                       loss            last          counter            rate
#      2.596         11.2893         11.2893             1500        0.999999
#      4.967         11.2832         11.2772             3000        0.999998
#      9.668         11.2622         11.2412             6000        0.999996
#     19.587         11.1822         11.1022            12000        0.999992
#     38.828         10.7662         10.3502            24000        0.999984
#     77.922         10.0589          9.3516            48000        0.999968
#    154.365          9.1684          8.2779            96000        0.999936
#    302.038          8.2353          7.3023           192000        0.999872
#    597.835          7.5238          6.8124           384000        0.999744
#   1191.957          6.9111          6.2984           768000        0.999488
#   2917.859          6.4206          5.9301          1536000        0.998977
