#! /usr/bin/env python

import caffe
import math
from math import sqrt, pow
import h5py
import numpy as np
import os
import sys
import time
import warnings
from scipy.sparse import csr_matrix

alpha=0.8
etastart=1.0 
initialt=100.0
powert=1.0
weightdecay=1e-5

# TODO: oversampling and final SVD

# UGH ... so much for DRY

lrs=dict()
lrs['embedding']=0.01
lrs[('ip1',0)]=0.01
lrs[('ip1',1)]=0.5  
lrs[('ip2',0)]=0.01
lrs[('ip2',1)]=0.5 
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

np.random.seed(8675309)

vocabsize=80000
windowsize=2
rawembeddingsize=200
batchsize=6000 
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
maxpasses=5

for passnum in range(maxpasses):
  orthopass=(passnum % 2 == 1)

  if orthopass:
    yembedhat=np.zeros(shape=(outvocabsize,labelembeddingsize),dtype='d')
  else:
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
  lastloss=0;

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

        if thisloss < lastloss:
          eta=(50.0/49.0)*eta
        else:
          eta=(49.0/50.0)*eta

        lastloss=thisloss;

        if orthopass:
          yembedhat+=np.transpose(sl).dot(np.multiply(res['ip3'].reshape((batchsize,labelembeddingsize)),importance[:, np.newaxis]))
        else:
          myeta=eta*pow(initialt/(initialt+numupdates),powert);
          xticx=np.sum(np.square(res['ip3']),axis=1).reshape(batchsize)
          impaware=-np.divide(np.expm1(-myeta*np.multiply(importance,xticx)),xticx)
          net.blobs['ip3'].diff[:]=(1.0/batchsize)*np.multiply(ip3diff,impaware[:, np.newaxis, np.newaxis, np.newaxis])
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

          momembeddiff=alpha*momembeddiff+lrs['embedding']*sdtransdiff
          embedding=embedding-momembeddiff
          embedding=(1-lrs['embedding']*weightdecay*myeta)*embedding
  
          for (name,layer,momlayer) in zip(net._layer_names,net.layers,momnet.layers):
            blobnum=0
            for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
              layereta=lrs[(name,blobnum)]
              momblob.data[:]=alpha*momblob.data[:]+layereta*blob.diff
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
            print "%4u  %10.3f  %8.4g  %8.4g  %10.6g  %10.6g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,sumimp,myeta)
            nextprint=2*nextprint
            sumsinceimp=0
            sumsinceloss=0

  now=time.time()
  print "%4u  %10.3f  %8.4g  %8.4g  %10.6g  %10.6g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,sumimp,myeta)
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

os.link(sys.argv[5]+"."+str(maxpasses-1),sys.argv[5])
os.link(sys.argv[5]+"_e."+str(maxpasses-1),sys.argv[5]+"_e")

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');
# 
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    0       0.760   0.02375   0.02375     1538.13           1
#    0       1.517    0.0132  0.002547     3060.34    0.995037
#    0       3.027   0.01347   0.01375     5994.37    0.985329
#    0       6.024  0.009409  0.005334       11973    0.966736
#    0      12.040  0.006588   0.00373       23794    0.932505
#    0      24.102  0.004608  0.002622     47501.1    0.873704
#    0      48.409  0.003555  0.002523     95997.5     0.78326
#    0      96.835  0.003038  0.002519      191765    0.663723
#    0     194.681  0.002777  0.002515      382278    0.530745
#    0     388.702  0.002644  0.002512      767336    0.404557
#    0     777.082  0.002576  0.002507  1.53489e+06    0.298408
#    0    1552.490   0.00254  0.002505  3.07183e+06    0.215816
#    0    3121.818  0.002522  0.002503   6.145e+06    0.154395
#    0    6255.429  0.002512  0.002502  1.2283e+07    0.109824
#    0   12549.258  0.002506    0.0025  2.45718e+07     0.07789
#    0   15912.412  0.002504  0.002499  3.10672e+07   0.0693125
# best constant loss: 0.00249742
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    1       0.212  0.002506  0.002506     1538.13   0.0693125
#    1       0.334  0.002496  0.002486     3060.34   0.0693125
#    1       0.556    0.0025  0.002504     5994.37   0.0693125
#    1       1.031  0.002499  0.002498       11973   0.0693125
#    1       1.931  0.002502  0.002506       23794   0.0693125
#    1       3.769    0.0025  0.002498     47501.1   0.0693125
#    1       7.356  0.002499  0.002498     95997.5   0.0693125
#    1      14.672    0.0025    0.0025      191765   0.0693125
#    1      29.179    0.0025    0.0025      382278   0.0693125
#    1      58.507    0.0025    0.0025      767336   0.0693125
#    1     117.408  0.002499  0.002498  1.53489e+06   0.0693125
#    1     235.631  0.002499  0.002499  3.07183e+06   0.0693125
#    1     472.295  0.002499  0.002499   6.145e+06   0.0693125
#    1     944.054  0.002499  0.002499  1.2283e+07   0.0693125
#    1    1871.985  0.002499  0.002499  2.45718e+07   0.0693125
#    1    2363.831  0.002499  0.002499  3.10672e+07   0.0693125
# best constant loss: 0.00249742
# 
# ss     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    2       0.750   0.07147   0.07147     1538.13           1
#    2       1.502   0.06051   0.04944     3060.34    0.995037
#    2       2.988   0.06218   0.06392     5994.37    0.985329
#    2       6.016   0.05711   0.05203       11973    0.966736
#    2      12.126   0.05458   0.05201       23794    0.932505
#    2      24.387   0.05175   0.04891     47501.1    0.873704
#    2      48.707    0.0499   0.04809     95997.5     0.78326
#    2      97.077   0.04973   0.04957      191765    0.663723
#    2     193.572   0.04967   0.04961      382278    0.530745
#    2     384.825   0.04933   0.04899      767336    0.404557
#    2     767.438    0.0492   0.04907  1.53489e+06    0.298408
#    2    1535.554   0.04905    0.0489  3.07183e+06    0.215816
#    2    3071.088   0.04889   0.04872   6.145e+06    0.154395
#    2    6168.534   0.04881   0.04872  1.2283e+07    0.109824
#    2   12406.005   0.04863   0.04845  2.45718e+07     0.07789
#    2   15725.018   0.04856   0.04833  3.10672e+07   0.0693125
# best constant loss: 0.049063
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    3       0.212   0.04451   0.04451     1538.13   0.0693125
#    3       0.329   0.04657   0.04865     3060.34   0.0693125
#    3       0.561    0.0479   0.04929     5994.37   0.0693125
#    3       1.005   0.04786   0.04782       11973   0.0693125
#    3       1.903   0.04877    0.0497       23794   0.0693125
#    3       3.687   0.04837   0.04797     47501.1   0.0693125
#    3       7.308   0.04783   0.04731     95997.5   0.0693125
#    3      14.704    0.0483   0.04876      191765   0.0693125
#    3      29.392   0.04856   0.04882      382278   0.0693125
#    3      58.141   0.04841   0.04825      767336   0.0693125
#    3     116.531   0.04838   0.04836  1.53489e+06   0.0693125
#    3     234.848   0.04833   0.04828  3.07183e+06   0.0693125
#    3     468.013   0.04828   0.04823   6.145e+06   0.0693125
#    3     935.627   0.04833   0.04839  1.2283e+07   0.0693125
#    3    1867.108   0.04831   0.04829  2.45718e+07   0.0693125
#    3    2401.905   0.04831   0.04829  3.10672e+07   0.0693125
# best constant loss: 0.049063
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    4       0.797   0.08059   0.08059     1538.13           1
#    4       1.599   0.07292   0.06518     3060.34    0.995037
#    4       3.212   0.07501   0.07719     5994.37    0.985329
#    4       6.410   0.07162   0.06822       11973    0.966736
#    4      12.888   0.07029   0.06895       23794    0.932505
#    4      25.836   0.06801   0.06572     47501.1    0.873704
#    4      51.564   0.06625   0.06453     95997.5     0.78326
#    4     103.531   0.06607   0.06588      191765    0.663723
#    4     207.801   0.06613   0.06619      382278    0.530745
#    4     417.014   0.06576   0.06539      767336    0.404557
#    4     834.298    0.0656   0.06544  1.53489e+06    0.298408
#    4    1649.537   0.06545    0.0653  3.07183e+06    0.215816
#    4    3236.426   0.06525   0.06505   6.145e+06    0.154395
#    4    6344.000   0.06516   0.06507  1.2283e+07    0.109824
#    4   12561.081   0.06494   0.06473  2.45718e+07     0.07789
#    4   15879.892   0.06488   0.06462  3.10672e+07   0.0693125
# best constant loss: 0.0654702
