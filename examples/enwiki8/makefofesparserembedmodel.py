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

onemalphastart=0.5
onemalphadecay=0.995
maxalpha=0.95
etastart=1e-3
etadecay=1.0
weightdecay=1e-6
epsilon=1e-12

# TODO: oversampling and final SVD

# UGH ... so much for DRY

lrs=dict()
lrs['embedding']=1
lrs[('ip1',0)]=1
lrs[('ip1',1)]=1
lrs[('ip2',0)]=1
lrs[('ip2',1)]=1
lrs[('ip3',0)]=1
lrs[('ip3',1)]=1

np.random.seed(8675309)
sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

vocabsize=80000
windowsize=2
rawembeddingsize=200
batchsize=81799
labelembeddingsize=400

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

scalefac=10
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
    net.set_mode_gpu()
    net.set_phase_train()

    momnet = caffe.Net(sys.argv[1])
    momnet.set_mode_gpu()
    momnet.set_phase_train()

    gradnet = caffe.Net(sys.argv[1])
    gradnet.set_mode_gpu()
    gradnet.set_phase_train()

    for (layer,momlayer,gradlayer) in zip(net.layers,momnet.layers,gradnet.layers):
      for (blob,momblob,gradblob) in zip(layer.blobs,momlayer.blobs,gradlayer.blobs):
        if blob.data.shape[2] > 1:
          blob.data[:]=np.transpose(np.linalg.qr(np.random.standard_normal(size=(blob.data.shape[3],blob.data.shape[2])))[0])
        else:
          blob.data[:]=0
        momblob.data[:]=np.zeros(blob.data.shape,dtype='f')
        gradblob.data[:]=np.zeros(blob.data.shape,dtype='f')

    momembeddiff=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')
    gradembeddiff=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')

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
    onemalpha=0
  else:
    eta=etastart
    onemalpha=onemalphastart

  print "%4s  %10s  %8s  %8s  %10s  %8s  %8s"%("pass","delta t","average","since","example","eta","alpha")
  print "%4s  %10s  %8s  %8s  %10s  %8s  %8s"%("","","loss","last","counter","","")

  with open(sys.argv[4], 'r') as f:
    linenum=0
    for line in f:
      linenum=linenum+1
      if linenum > maxlinenum:
        break
  
      yx=[word for word in line.split(' ')]
      labels[bindex]=int(yx[0])-2
      labelvalues[bindex]=sqrt(scalefac*baserates[0])/sqrt(baserates[0]+(scalefac-1)*baserates[int(yx[0])-2])
      importance[bindex]=1
  
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
          myeta=eta
          myalpha=min(maxalpha,1.0-onemalpha)
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
          gradembeddiff+=np.abs(sdtransdiff)

          momembeddiff=myalpha*momembeddiff+lrs['embedding']*myeta*sdtransdiff
          embedding=embedding-np.divide(momembeddiff,epsilon+gradembeddiff)
          embedding=(1-lrs['embedding']*weightdecay*myeta)*embedding

          for (name,layer,momlayer,gradlayer) in zip(net._layer_names,net.layers,momnet.layers,gradnet.layers):
            blobnum=0
            for (blob,momblob,gradblob) in zip(layer.blobs,momlayer.blobs,gradlayer.blobs):
              layereta=lrs[(name,blobnum)]*myeta
              momblob.data[:]=myalpha*momblob.data+layereta*blob.diff
              gradblob.data[:]+=np.abs(blob.diff)
              blob.data[:]-=np.divide(momblob.data,epsilon+gradblob.data)
              blob.data[:]*=(1-weightdecay*myeta)
              blobnum=blobnum+1
  
        value=[]
        row=[]
        col=[]
        labels[:]=0
        bindex=0
        eta=eta*etadecay
        onemalpha=onemalpha*onemalphadecay
        numupdates=numupdates+1
        if numupdates >= nextprint:
            now=time.time()
            print "%4u  %10.3f  %8.4g  %8.4g  %10.6g  %8.5g  %8.4g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,sumimp,myeta,myalpha)
            print "best constant loss: %g"%(np.sum(embedsumsq/sumimp-np.square(embedsum/sumimp)))
            nextprint=2*nextprint
            sumsinceimp=0
            sumsinceloss=0

  now=time.time()
  print "%4u  %10.3f  %8.4g  %8.4g  %10.6g  %8.5g  %8.4g"%(passnum,now-start,sumloss/sumimp,sumsinceloss/sumsinceimp,sumimp,myeta,myalpha)
  print "best constant loss: %g"%(np.sum(embedsumsq/sumimp-np.square(embedsum/sumimp)))

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
    del gradnet

os.link(sys.argv[5]+"."+str(maxpasses-1),sys.argv[5])
os.link(sys.argv[5]+"_e."+str(maxpasses-1),sys.argv[5]+"_e")

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');
# pass     delta t   average     since     example       eta     alpha
#                       loss      last     counter
#    0       6.741   0.03499   0.03499       81799     0.001       0.5
# best constant loss: 0.0328978
#    0      13.404   0.03411   0.03323      163598     0.001    0.5025
# best constant loss: 0.032849
#    0      26.471   0.03352   0.03293      327196     0.001    0.5075
# best constant loss: 0.0327983
#    0      53.647   0.03317   0.03282      654392     0.001    0.5172
# best constant loss: 0.0328262
#    0     107.329   0.03295   0.03272  1.30878e+06     0.001    0.5362
# best constant loss: 0.0328485
#    0     213.770   0.03275   0.03256  2.61757e+06     0.001     0.572
# best constant loss: 0.032849
#    0     427.150   0.03254   0.03232  5.23514e+06     0.001    0.6354
# best constant loss: 0.0328372
#    0     858.407   0.03229   0.03205  1.04703e+07     0.001    0.7355
# best constant loss: 0.0328371
#    0    1718.628   0.03198   0.03168  2.09405e+07     0.001    0.8607
# best constant loss: 0.0328405
#    0    3443.781   0.03154   0.03109  4.18811e+07     0.001      0.95
# best constant loss: 0.0328436
#    0    6888.679   0.03092    0.0303  8.37622e+07     0.001      0.95
# best constant loss: 0.0328442
#    0    9196.582   0.03068   0.02996  1.11819e+08     0.001      0.95
# best constant loss: 0.0328459
# pass     delta t   average     since     example       eta     alpha
#                       loss      last     counter
#    1       4.854   0.02988   0.02988       81799     0.001      0.95
# best constant loss: 0.0328978
#    1       9.630   0.02983   0.02977      163598     0.001      0.95
# best constant loss: 0.032849
#    1      19.372   0.02978   0.02974      327196     0.001      0.95
# best constant loss: 0.0327983
#    1      38.464   0.02983   0.02987      654392     0.001      0.95
# best constant loss: 0.0328262
#    1      77.056   0.02986    0.0299  1.30878e+06     0.001      0.95
# best constant loss: 0.0328485
#    1     153.686   0.02986   0.02986  2.61757e+06     0.001      0.95
# best constant loss: 0.032849
#    1     306.401   0.02985   0.02983  5.23514e+06     0.001      0.95
# best constant loss: 0.0328372
#    1     615.122   0.02985   0.02986  1.04703e+07     0.001      0.95
# best constant loss: 0.0328371
#    1    1234.527   0.02985   0.02986  2.09405e+07     0.001      0.95
# best constant loss: 0.0328405
#    1    2459.772   0.02985   0.02985  4.18811e+07     0.001      0.95
# best constant loss: 0.0328436
#    1    4864.160   0.02984   0.02982  8.37622e+07     0.001      0.95
# best constant loss: 0.0328442
#    1    6471.408   0.02983   0.02982  1.11819e+08     0.001      0.95
# best constant loss: 0.0328459
# pass     delta t   average     since     example       eta     alpha           
#                       loss      last     counter                               
#    2       7.038     2.594     2.594       81799     0.001       0.5           
# best constant loss: 2.56951                                                    
#    2      13.881     2.586     2.579      163598     0.001    0.5025           
# best constant loss: 2.56818                                                    
#    2      27.631     2.578      2.57      327196     0.001    0.5075           
# best constant loss: 2.57088                                                    
#    2      54.254     2.556     2.534      654392     0.001    0.5172           
# best constant loss: 2.56693                                                    
#    2     107.743     2.532     2.509  1.30878e+06     0.001    0.5362
# best constant loss: 2.5674
#    2     214.825     2.502     2.472  2.61757e+06     0.001     0.572
# best constant loss: 2.56747
#    2     430.358     2.468     2.434  5.23514e+06     0.001    0.6354
# best constant loss: 2.56669
#    2     860.473     2.433     2.398  1.04703e+07     0.001    0.7355
# best constant loss: 2.56623
#    2    1719.553     2.394     2.355  2.09405e+07     0.001    0.8607
# best constant loss: 2.56655
#    2    3442.367      2.34     2.286  4.18811e+07     0.001      0.95
# best constant loss: 2.56664
#    2    6930.557     2.269     2.198  8.37622e+07     0.001      0.95
# best constant loss: 2.56666
#    2    9201.379     2.241     2.158  1.11819e+08     0.001      0.95
# best constant loss: 2.56657
# pass     delta t   average     since     example       eta     alpha           
#                       loss      last     counter                               
#    3       4.711     2.148     2.148       81799     0.001      0.95           
# best constant loss: 2.56951                                                    
#    3       9.479     2.144      2.14      163598     0.001      0.95           
# best constant loss: 2.56818                                                    
#    3      18.856     2.146     2.147      327196     0.001      0.95           
# best constant loss: 2.57088                                                    
#    3      37.890     2.144     2.141      654392     0.001      0.95           
# best constant loss: 2.56693                                                    
#    3      75.520     2.146     2.148  1.30878e+06     0.001      0.95
# best constant loss: 2.5674
#    3     152.210     2.146     2.146  2.61757e+06     0.001      0.95
# best constant loss: 2.56747
#    3     305.098     2.145     2.145  5.23514e+06     0.001      0.95
# best constant loss: 2.56669
#    3     612.560     2.146     2.147  1.04703e+07     0.001      0.95
# best constant loss: 2.56623
#    3    1227.462     2.147     2.147  2.09405e+07     0.001      0.95
# best constant loss: 2.56655
#    3    2447.143     2.146     2.146  4.18811e+07     0.001      0.95
# best constant loss: 2.56664
#    3    4890.360     2.145     2.144  8.37622e+07     0.001      0.95
# best constant loss: 2.56666
#    3    6532.754     2.145     2.144  1.11819e+08     0.001      0.95
# best constant loss: 2.56657
# pass     delta t   average     since     example       eta     alpha           
#                       loss      last     counter                               
#    4       6.903     2.696     2.696       81799     0.001       0.5           
# best constant loss: 2.67061                                                    
#    4      13.980     2.688      2.68      163598     0.001    0.5025           
# best constant loss: 2.66807                                                    
#    4      27.730      2.68     2.672      327196     0.001    0.5075           
# best constant loss: 2.67067                                                    
#    4      54.169     2.658     2.636      654392     0.001    0.5172           
# best constant loss: 2.66659                                                    
#    4     106.793     2.635     2.611  1.30878e+06     0.001    0.5362          
# best constant loss: 2.66774
#    4     212.052     2.604     2.573  2.61757e+06     0.001     0.572
# best constant loss: 2.66753
#    4     422.090     2.569     2.535  5.23514e+06     0.001    0.6354
# best constant loss: 2.66672
#    4     847.322     2.534     2.499  1.04703e+07     0.001    0.7355
# best constant loss: 2.66613
#    4    1706.107     2.496     2.457  2.09405e+07     0.001    0.8607
# best constant loss: 2.6662
#    4    3436.984     2.442     2.388  4.18811e+07     0.001      0.95
# best constant loss: 2.66622
#    4    6836.399      2.37     2.299  8.37622e+07     0.001      0.95
# best constant loss: 2.66611
#    4    9121.605     2.342     2.258  1.11819e+08     0.001      0.95
# best constant loss: 2.66601

