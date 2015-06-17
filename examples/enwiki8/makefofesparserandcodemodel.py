#! /usr/bin/env python

import caffe
import math
from math import sqrt, pow
import h5py
import numpy as np
import os
import random
import sys
import time
import warnings
from scipy.sparse import csr_matrix

#np.seterr(all='warn') 

onemalpha=1.0
alphadecay=0.999
maxalpha=0.4
etastart=1
etadecay=0.9995
weightdecay=1e-7
labelnoise=0.001
scalefac=10
maxpasses=1

vocabsize=80000
windowsize=2
rawembeddingsize=200
batchsize=12553 
labelembeddingsize=300

# UGH ... so much for DRY

lrs=dict()
lrs['embedding']=1
lrs[('ip1',0)]=1.5
lrs[('ip1',1)]=1.75
lrs[('ip2',0)]=1
lrs[('ip2',1)]=1.25
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

#-----------------------------------------------

np.random.seed(8675309)
random.seed(90210)
sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

embeddingsize=windowsize*rawembeddingsize
invocabsize=windowsize*(vocabsize+2)
outvocabsize=vocabsize+1

labelembedding=np.zeros(shape=(outvocabsize,labelembeddingsize),dtype='f')
for row in range(outvocabsize):
  labelembedding[row,:]=2*np.mod(np.random.permutation(labelembeddingsize),2)-1

baserates=np.zeros(outvocabsize)
with open(sys.argv[2], 'r') as f:
  for line in f:
    lc=[word for word in line.split('\t')]
    baserates[int(lc[0])-2]=float(lc[1])

baserates=baserates/np.sum(baserates)

expectedimportance=baserates.dot(np.divide(sqrt(scalefac*baserates[0]),np.sqrt(baserates[0]+(scalefac-1)*baserates)))

maxlinenum=int(sys.argv[3])

preembed=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')
preembed[:]=np.random.standard_normal(size=(invocabsize,embeddingsize))
embedding=np.linalg.qr(preembed)[0]

net = caffe.Net(sys.argv[1])
net.set_mode_cpu()
net.set_phase_train()

momnet = caffe.Net(sys.argv[1])
momnet.set_mode_cpu()
momnet.set_phase_train()

for (layer,momlayer) in zip(net.layers,momnet.layers):
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    if blob.data.shape[2] > 1:
      blob.data[:]=np.transpose(np.linalg.qr(np.random.standard_normal(size=(blob.data.shape[3],blob.data.shape[2])))[0])
    else:
      blob.data[:]=0
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')

momembeddiff=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')

eta=etastart
row=[]
col=[]
value=[]
labels=np.zeros(batchsize,dtype='f')
labelvalues=np.ones(batchsize,dtype='f')
importance=np.ones(batchsize,dtype='f')
bindex=0
start=time.time()
sumloss=0
sumsinceloss=0
sumimp=0
sumsinceimp=0
numupdates=0
nextprint=1
embedsum=np.zeros(labelembeddingsize,dtype='d')
embedsumsince=np.zeros(labelembeddingsize,dtype='d')

print "%4s  %8s  %8s  %8s  %8s  %12s  %8s  %8s"%("pass","delta t","average","since","best","example","eta","alpha")
print "%4s  %8s  %8s  %8s  %8s  %12s  %8s  %8s"%("","","loss","last","const","counter","","")

for passnum in range(maxpasses):
  with open(sys.argv[4], 'r') as f:
    linenum=0
    for line in f:
      linenum=linenum+1
      if linenum > maxlinenum:
        break
  
      yx=[word for word in line.split(' ')]
      labels[bindex]=int(yx[0])-2
      labelvalues[bindex]=-1 if random.random()<labelnoise else 1
      importance[bindex]=(1.0/expectedimportance)*sqrt(scalefac*baserates[0])/sqrt(baserates[0]+(scalefac-1)*baserates[int(yx[0])-2])
  
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
        projlabels=sl.dot(labelembedding)

        net.set_input_arrays(data,labels)
        res=net.forward()

	bigprojlabels=projlabels.reshape(batchsize,labelembeddingsize,1,1)
	labeltimespred=np.multiply(bigprojlabels,res['ip3'])
	labeltimespred=np.minimum(labeltimespred,20)
	labeltimespred=np.maximum(labeltimespred,-20)
	thisloss=importance.dot(np.sum(np.log1p(np.exp(-labeltimespred)),axis=1).reshape(batchsize))
        embedsum+=importance.dot(projlabels)
        embedsumsince+=importance.dot(projlabels)
        sumimp+=np.sum(importance)
        sumsinceimp+=np.sum(importance)
        sumloss+=thisloss
        sumsinceloss+=thisloss

	ip3diff=np.multiply(np.divide(bigprojlabels,1+np.exp(labeltimespred)), \
			    importance[:,np.newaxis,np.newaxis,np.newaxis])
        net.blobs['ip3'].diff[:]=-(1.0/batchsize)*ip3diff
	net.backward()
        data_diff=net.blobs['data'].diff.reshape(batchsize,embeddingsize)

	# print np.ravel(res['ip3'][0,1:5,:,:])
	# print np.ravel(bigprojlabels[0,1:5,:,:])
	# print np.ravel(ip3diff[0,1:5,:,:])

        # y = W x
        # y_k = \sum_l W_kl x_l
        # df/dW_ij = \sum_k df/dy_k dy_k/dW_ij
        #          = \sum_k df/dy_k (\sum_l 1_{i=k} 1_{j=l} x_l)
        #          = \sum_k df/dy_k 1_{i=k} x_j
        #          = df/dy_i x_j
        # df/dW    = (df/dy)*x'
  
        sdtransdiff=sd.transpose().tocsr().dot(data_diff)

        momembeddiff=min(1-onemalpha,maxalpha)*momembeddiff+lrs['embedding']*sdtransdiff
        embedding=embedding-momembeddiff
        embedding=(1-lrs['embedding']*weightdecay*eta)*embedding
  
        for (name,layer,momlayer) in zip(net._layer_names,net.layers,momnet.layers):
          blobnum=0
          for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
            layereta=lrs[(name,blobnum)]
            momblob.data[:]=min(1-onemalpha,maxalpha)*momblob.data[:]+layereta*blob.diff
            blob.data[:]-=momblob.data[:]
            blob.data[:]=(1-weightdecay*layereta)*blob.data[:]
            blobnum=blobnum+1
  
        value=[]
        row=[]
        col=[]
        labels[:]=0
        bindex=0
        numupdates=numupdates+1
        if numupdates >= nextprint:
            now=time.time()
	    bestconst=0.5+0.5*(embedsumsince/sumsinceimp)
	    bestconstloss=np.sum(np.multiply(1-bestconst,np.log1p(np.exp(bestconst)))+np.multiply(bestconst,np.log1p(np.exp(-bestconst))))
            print "%4u  %8.3f  %8.4g  %8.4g  %8.4g  %12.6g  %8.4g  %8.4g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,bestconstloss,sumimp,eta,min(1-onemalpha,maxalpha))
            nextprint=2*nextprint
            sumsinceimp=0
            sumsinceloss=0
        eta=eta*etadecay
	onemalpha=onemalpha*alphadecay
	embedsumsince[:]=0

  now=time.time()
  bestconst=0.5+0.5*(embedsum/sumimp)
  bestconstloss=np.sum(np.multiply(1-bestconst,np.log1p(np.exp(-bestconst)))+np.multiply(bestconst,np.log1p(np.exp(bestconst))))
  print "%4u  %8.3f  %8.4g  %8.4g  %8.4g  %12.6g  %8.4g  %8.4g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,bestconstloss,sumimp,eta,min(1-onemalpha,maxalpha))

  net.save(sys.argv[5]+"."+str(passnum))
  h5f=h5py.File(sys.argv[5]+"_e."+str(passnum))
  h5f.create_dataset('embedding',data=embedding)
  h5f.create_dataset('labelembedding',data=labelembedding)
  h5f.close()

os.link(sys.argv[5]+"."+str(maxpasses-1),sys.argv[5])
os.link(sys.argv[5]+"_e."+str(maxpasses-1),sys.argv[5]+"_e")

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');
# 
# GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparserandcodemodel.py fofe_sparse_rembed_small_unigram_train fofebaserates9.txt `cat numlinesfofengram9 | perl -lane 'print int(0.25*$F[0])'` fofengram9.txt fofesparserembedmodel9
# 
# pass   delta t   average     since      best       example       eta     alpha
#                     loss      last     const       counter
#    0     1.821     207.9     207.9       217       12523.7         1         0
#    0     3.537     207.8     207.6       217       25054.1    0.9995     0.001
#    0     7.046     207.6     207.5     217.2       50157.9    0.9985  0.002997
#    0    14.084     207.5     207.4     217.2        100512    0.9965  0.006979
#    0    28.076     207.3     207.2     217.2        200850    0.9925    0.0149
#    0    56.253     207.1     206.8     217.2        401328    0.9846   0.03054
#    0   111.567     206.7     206.3     217.2        803407     0.969   0.06109
#    0   222.221     205.9     205.2     217.2   1.60687e+06    0.9385    0.1193
#    0   443.273     203.8     201.6     217.2   3.21339e+06    0.8803    0.2252
#    0   884.181     198.5     193.2     217.2   6.42599e+06    0.7745       0.4
#    0  1765.189     193.6     188.8     217.2   1.28519e+07    0.5995       0.4
#    0  3533.009       190     186.4     217.2   2.57059e+07    0.3592       0.4
#    0  4273.186     189.2     185.6     217.5   3.10667e+07      0.29       0.4

