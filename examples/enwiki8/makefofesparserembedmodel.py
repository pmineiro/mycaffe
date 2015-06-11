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
etastart=1.0
etadecay=0.99999
weightdecay=1e-5

# TODO: oversampling and final SVD

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

prelabelembed=np.zeros(shape=(outvocabsize,labelembeddingsize),dtype='f')
prelabelembed[:]=np.random.standard_normal(size=(outvocabsize,labelembeddingsize))
labelembedding=np.linalg.qr(prelabelembed)[0]

baserates=np.zeros(outvocabsize)
with open(sys.argv[2], 'r') as f:
  for line in f:
    lc=[word for word in line.split('\t')]
    baserates[int(lc[0])-2]=float(lc[1])

baserates=baserates/np.sum(baserates)

scalefac=10000
expectedimportance=baserates.dot(np.divide(sqrt(scalefac*baserates[0]),np.sqrt(baserates[0]+(scalefac-1)*baserates)))

maxlinenum=int(sys.argv[3])

for passnum in range(5):

  orthopass=(passnum % 2 == 1)

  if orthopass:
    yembedhat=np.zeros(shape=(outvocabsize,labelembeddingsize),dtype='d')
  else:
    preembed=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')
    preembed[:]=np.random.standard_normal(size=(invocabsize,embeddingsize))
    embedding=sqrt(embeddingsize)*np.linalg.qr(preembed)[0]

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
  embedsumsq=np.zeros(labelembeddingsize,dtype='d')

  if orthopass:
    eta=0
  else:
    eta=etastart

  print "%4s  %10s  %8s  %8s  %10s  %10s"%("pass","delta t","average","since","example","learning")
  print "%4s  %10s  %8s  %8s  %10s  %10s"%("","","loss","last","counter","rate")

  with open(sys.argv[4], 'r') as f:
    linenum=0
    for line in f:
      linenum=linenum+1
      if linenum > maxlinenum:
        break
  
      yx=[word for word in line.split(' ')]
      labels[bindex]=int(yx[0])-2
      labelvalues[bindex]=1
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
        embedsum+=importance.dot(projlabels)
        embedsumsq+=importance.dot(np.square(projlabels))
        net.set_input_arrays(data,labels)
        res=net.forward()
        ip3diff=res['ip3']-projlabels.reshape(batchsize,labelembeddingsize,1,1)
        thisloss=importance.dot(np.sum(np.square(ip3diff),axis=1).reshape(batchsize))
        sumloss+=thisloss
        sumsinceloss+=thisloss
        sumimp+=np.sum(importance)
        sumsinceimp+=np.sum(importance)

        if orthopass:
          yembedhat+=np.transpose(sl).dot(np.multiply(res['ip3'].reshape((batchsize,labelembeddingsize)),importance[:, np.newaxis]))
        else:
          net.blobs['ip3'].diff[:]=(1.0/batchsize)*np.multiply(ip3diff,importance[:, np.newaxis, np.newaxis, np.newaxis])
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
        if numupdates >= nextprint:
            now=time.time()
            print "%4u  %10.3f  %8.4g  %8.4g  %10.6g  %10.6g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,sumimp,eta)
            nextprint=2*nextprint
            sumsinceimp=0
            sumsinceloss=0

  now=time.time()
  print "%4u  %10.3f  %8.4g  %8.4g  %10.6g  %10.6g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,sumimp,eta)
  embedsum*=1.0/sumimp
  embedsumsq*=1.0/sumimp
  print "best constant loss: %g"%(np.sum(embedsumsq-np.square(embedsum)))

  if orthopass:
    labelembedding=np.linalg.qr(yembedhat)[0].astype(dtype='f')
    h5f=h5py.File(sys.argv[5]+"_e."+str(passnum))
    h5f.create_dataset('labelembedding',data=labelembedding)
    h5f.create_dataset('yembedhat',data=yembedhat)
    h5f.close()
    del net
    del embedding
    del yembedhat
  else:
    net.save(sys.argv[5]+"."+str(passnum))
    h5f=h5py.File(sys.argv[5]+"_e."+str(passnum))
    h5f.create_dataset('embedding',data=embedding)
    h5f.create_dataset('labelembedding',data=labelembedding)
    h5f.close()
    del momnet

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');
