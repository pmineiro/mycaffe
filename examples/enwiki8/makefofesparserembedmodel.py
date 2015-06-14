#! /usr/bin/env python

import caffe
import math
from math import sqrt
import h5py
import numpy as np
import os
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
maxpasses=5

for passnum in range(maxpasses):
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

os.link(sys.argv[5]+"."+str(maxpasses),sys.argv[5])
os.link(sys.argv[5]+"_e."+str(maxpasses),sys.argv[5]+"_e")

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');
# 
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    0       0.781   0.02375   0.02375     1538.13     0.99999
#    0       1.538   0.01324  0.002612     3060.34     0.99998
#    0       3.058   0.01561   0.01809     5994.37     0.99996
#    0       6.162   0.01181  0.007992       11973     0.99992
#    0      12.297  0.009268  0.006697       23794     0.99984
#    0      24.853  0.006453  0.003628     47501.1     0.99968
#    0      49.239  0.004545  0.002675     95997.5     0.99936
#    0      97.780  0.003539  0.002532      191765    0.998721
#    0     194.617  0.003036   0.00253      382278    0.997443
#    0     388.307  0.002782   0.00253      767336    0.994893
#    0     774.498  0.002655  0.002528  1.53489e+06    0.989812
#    0    1549.328  0.002591  0.002527  3.07183e+06    0.979728
#    0    3099.920  0.002559  0.002527   6.145e+06    0.959867
#    0    6198.609  0.002543  0.002526  1.2283e+07    0.921345
#    0   12323.411  0.002533  0.002524  2.45718e+07    0.848877
#    0   15559.550  0.002531  0.002521  3.10672e+07    0.812889
# best constant loss: 0.00249742
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    1       0.211   0.00253   0.00253     1538.13           0
#    1       0.327  0.002518  0.002506     3060.34           0
#    1       0.548  0.002523  0.002527     5994.37           0
#    1       1.006  0.002521  0.002519       11973           0
#    1       1.925  0.002525  0.002528       23794           0
#    1       3.733  0.002523  0.002521     47501.1           0
#    1       7.290  0.002522  0.002521     95997.5           0
#    1      14.420  0.002522  0.002523      191765           0
#    1      28.661  0.002522  0.002523      382278           0
#    1      57.426  0.002523  0.002523      767336           0
#    1     114.830  0.002522  0.002521  1.53489e+06           0
#    1     229.555  0.002522  0.002521  3.07183e+06           0
#    1     460.655  0.002522  0.002522   6.145e+06           0
#    1     923.599  0.002522  0.002522  1.2283e+07           0
#    1    1848.977  0.002522  0.002522  2.45718e+07           0
#    1    2338.188  0.002522  0.002521  3.10672e+07           0
# best constant loss: 0.00249742
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    2       0.770     0.054     0.054     1538.13     0.99999
#    2       1.549   0.04168   0.02922     3060.34     0.99998
#    2       3.112   0.04681   0.05217     5994.37     0.99996
#    2       6.167   0.04181    0.0368       11973     0.99992
#    2      12.274    0.0396   0.03736       23794     0.99984
#    2      24.742   0.03623   0.03285     47501.1     0.99968
#    2      49.298   0.03344   0.03071     95997.5     0.99936
#    2      98.779   0.03242   0.03139      191765    0.998721
#    2     197.338   0.03217   0.03192      382278    0.997443
#    2     393.757   0.03174   0.03131      767336    0.994893
#    2     789.314   0.03151   0.03129  1.53489e+06    0.989812
#    2    1578.830   0.03119   0.03087  3.07183e+06    0.979728
#    2    3158.645    0.0303   0.02941   6.145e+06    0.959867
#    2    6352.179   0.02806   0.02582  1.2283e+07    0.921345
#    2   12745.367   0.02595   0.02383  2.45718e+07    0.848877
#    2   16115.160   0.02537   0.02318  3.10672e+07    0.812889
# best constant loss: 0.0314169
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    3       0.233   0.02105   0.02105     1538.13           0
#    3       0.370   0.02063   0.02021     3060.34           0
#    3       0.595   0.02245   0.02435     5994.37           0
#    3       1.045   0.02178   0.02111       11973           0
#    3       1.968   0.02294    0.0241       23794           0
#    3       3.750   0.02313   0.02332     47501.1           0
#    3       7.393    0.0227   0.02228     95997.5           0
#    3      14.650   0.02282   0.02295      191765           0
#    3      29.087   0.02307   0.02331      382278           0
#    3      58.348   0.02303   0.02299      767336           0
#    3     116.742   0.02313   0.02323  1.53489e+06           0
#    3     233.349   0.02309   0.02306  3.07183e+06           0
#    3     465.474   0.02305   0.02301   6.145e+06           0
#    3     976.009    0.0231   0.02315  1.2283e+07           0
#    3    1993.849   0.02309   0.02307  2.45718e+07           0
#    3    2483.775   0.02309   0.02308  3.10672e+07           0
# best constant loss: 0.0314169
# pass     delta t   average     since     example    learning
#                       loss      last     counter        rate
#    4       0.778    0.0948    0.0948     1538.13     0.99999
#    4       1.524   0.08841   0.08194     3060.34     0.99998
#    4       3.040   0.09271   0.09721     5994.37     0.99996
#    4       6.031   0.09015   0.08757       11973     0.99992
#    4      12.056   0.09018   0.09022       23794     0.99984
#    4      24.054    0.0871     0.084     47501.1     0.99968
#    4      47.977   0.08409   0.08114     95997.5     0.99936
#    4      95.989   0.08336   0.08262      191765    0.998721
#    4     191.762   0.08325   0.08314      382278    0.997443
#    4     383.131   0.08263   0.08202      767336    0.994893
#    4     765.929   0.08225   0.08187  1.53489e+06    0.989812
#    4    1532.589   0.08161   0.08098  3.07183e+06    0.979728
#    4    3069.534   0.07886   0.07611   6.145e+06    0.959867
#    4    6158.580   0.07279    0.0667  1.2283e+07    0.921345
#    4   12261.654   0.06638   0.05998  2.45718e+07    0.848877
#    4   15471.016   0.06455   0.05761  3.10672e+07    0.812889
# best constant loss: 0.0821423
