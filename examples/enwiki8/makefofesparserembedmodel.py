#! /usr/bin/env python

import caffe
import math
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
lrs['embedding']=1
lrs[('ip1',0)]=1.5
lrs[('ip1',1)]=1.75
lrs[('ip2',0)]=1
lrs[('ip2',1)]=1.25
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

np.random.seed(8675309)

vocabsize=80000
windowsize=2
rawembeddingsize=200
batchsize=1500 
labelembeddingsize=200

embeddingsize=windowsize*rawembeddingsize
invocabsize=windowsize*(vocabsize+2)
outvocabsize=vocabsize+1

preembed=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')
preembed[:]=np.random.standard_normal(size=(invocabsize,embeddingsize))
embedding=math.sqrt(embeddingsize)*np.linalg.qr(preembed)[0]

prelabelembed=np.zeros(shape=(outvocabsize,labelembeddingsize),dtype='f')
prelabelembed[:]=np.random.standard_normal(size=(outvocabsize,labelembeddingsize))
labelembedding=np.linalg.qr(prelabelembed)[0]

net = caffe.Net(sys.argv[1])
net.set_mode_cpu()
net.set_phase_train()

momnet = caffe.Net(sys.argv[1])
momnet.set_mode_cpu()
momnet.set_phase_train()

for (layer,momlayer) in zip(net.layers,momnet.layers):
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    blob.data[:]=0.01*np.random.standard_normal(size=blob.data.shape)
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')

momembeddiff=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')

baserates=np.zeros(outvocabsize)
with open(sys.argv[2], 'r') as f:
  for line in f:
    lc=[word for word in line.split('\t')]
    baserates[int(lc[0])-2]=float(lc[1])

baserates=baserates/np.sum(baserates)

row=[]
col=[]
value=[]
labels=np.zeros(batchsize,dtype='f')
labelvalues=np.ones(batchsize,dtype='f')
bindex=0
start=time.time()
numsinceupdates=0
numupdates=0
sumloss=0
sumsinceloss=0
nextprint=1

print "%10s\t%10s\t%10s\t%11s\t%11s"%("delta t","average","since","example","learning")
print "%10s\t%10s\t%10s\t%11s\t%11s"%("","loss","last","counter","rate")

with open(sys.argv[3], 'r') as f:
  for line in f:
    yx=[word for word in line.split(' ')]
    labels[bindex]=int(yx[0])-2
    # TODO: figure out logistic reweighting
    #labelvalues[bindex]=sqrt(100*baserates[0])/sqrt(baserates[0]+99*baserates[int(yx[0])-2])

    for word in yx[1:]:
      iv=[subword for subword in word.split(':')]
      row.append(bindex)
      col.append(int(iv[0])-1)
      value.append(float(iv[1]))
    
    bindex=bindex+1

    if bindex >= batchsize:
      sd=csr_matrix((value, (row, col)),                    		\
                    shape=(batchsize,invocabsize),          		\
                    dtype='f')
      data=sd.dot(embedding).reshape(batchsize,1,1,embeddingsize)
      sl=csr_matrix((labelvalues,                                   	\
                    (np.linspace(0,batchsize-1,batchsize),          	\
                     labels)),                                      	\
                     shape=(batchsize,outvocabsize),                	\
                     dtype='f')
      projlabels=sl.dot(labelembedding).reshape(batchsize,labelembeddingsize,1,1)
      net.set_input_arrays(data,labels)
      res=net.forward()
      ip3diff=res['ip3']-projlabels
      sumloss+=np.sum(np.square(ip3diff))/batchsize
      sumsinceloss+=np.sum(np.square(ip3diff))/batchsize
      net.blobs['ip3'].diff[:]=(1.0/batchsize)*ip3diff
      net.backward()
      data_diff=net.blobs['data'].diff.reshape(batchsize,embeddingsize)

      # y = W x
      # y_k = \sum_l W_kl x_l
      # df/dW_ij = \sum_k df/dy_k dy_k/dW_ij
      #          = \sum_k df/dy_k (\sum_l 1_{i=k} 1_{j=l} x_l)
      #          = \sum_k df/dy_k 1_{i=k} x_j
      #          = df/dy_i x_j
      # df/dW    = (df/dy)*x'

      sdtransdiff=sd.transpose().tocsr().dot(data_diff)

      momembeddiff=alpha*momembeddiff+lrs['embedding']*eta*sdtransdiff
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
          net.save(sys.argv[4]+"."+str(numupdates))
          h5f=h5py.File(sys.argv[4]+"_e."+str(numupdates))
          h5f.create_dataset('embedding',data=embedding)
          h5f.create_dataset('labelembedding',data=labelembedding)
          h5f.close()
          now=time.time()
          print "%10.3f\t%10.4g\t%10.4g\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
          nextprint=2*nextprint
          numsinceupdates=0
          sumsinceloss=0

now=time.time()
print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
net.save(sys.argv[4])
h5f=h5py.File(sys.argv[4]+"_e")
h5f.create_dataset('embedding',data=embedding)
h5f.create_dataset('labelembedding',data=labelembedding)
h5f.close()

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');

#   delta t         average           since          example        learning
#                      loss            last          counter            rate
#    57.005         0.02397         0.02397             1500         0.99999
#    57.945         0.01326        0.002548             3000         0.99998
#    59.543         0.01571         0.01815             6000         0.99996
#    62.579         0.01213        0.008558            12000         0.99992
#    68.456        0.009639        0.007146            24000         0.99984
#    80.268        0.006736        0.003833            48000         0.99968
#   102.902        0.004691        0.002646            96000         0.99936
#   148.173        0.003596          0.0025           192000        0.998721
#   238.055        0.003047        0.002498           384000        0.997443
#   417.912        0.002772        0.002497           768000        0.994893
#   781.941        0.002632        0.002492          1536000        0.989812
#  1504.750        0.002555        0.002477          3072000        0.979728
#  2982.468        0.002506        0.002458          6144000        0.959867
