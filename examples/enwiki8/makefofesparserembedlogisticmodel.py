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

onemalpha=0.5
onemalphadecay=0.9992
maxalpha=0.92
eta=1.0
etadecay=1.0
weightdecay=0
epsilon=1e-12

# UGH ... so much for DRY

lrs=dict()
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

np.random.seed(8675309)
sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

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
net.set_mode_gpu()
net.set_phase_train()

momnet = caffe.Net(sys.argv[4])
momnet.set_mode_gpu()
momnet.set_phase_train()

gradnet = caffe.Net(sys.argv[4])
gradnet.set_mode_gpu()
gradnet.set_phase_train()

for (layer,momlayer,gradlayer) in zip(net.layers,momnet.layers,gradnet.layers):
  for (blob,momblob,gradblob) in zip(layer.blobs,momlayer.blobs,gradlayer.blobs):
    blob.data[:]=0.01*np.random.standard_normal(size=blob.data.shape)
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')
    gradblob.data[:]=np.zeros(blob.data.shape,dtype='f')

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
    print "%4s  %10s  %8s  %8s  %10s  %8s  %8s"%("pass","delta t","average","since","example","eta","alpha")
    print "%4s  %10s  %8s  %8s  %10s  %8s  %8s"%("","","loss","last","counter","","")
    
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
  
        for (name,layer,momlayer,gradlayer) in zip(net._layer_names,net.layers,momnet.layers,gradnet.layers):
          blobnum=0
          for (blob,momblob,gradblob) in zip(layer.blobs,momlayer.blobs,gradlayer.blobs):
            myeta=lrs[(name,blobnum)]*eta
            myalpha=min(maxalpha,1.0-onemalpha)
            momblob.data[:]*=myalpha
            momblob.data[:]+=myeta*blob.diff
            gradblob.data[:]+=np.abs(blob.diff)
            blob.data[:]-=np.divide(momblob.data,epsilon+gradblob.data)
            if weightdecay > 0:
              blob.data[:]*=(1-myeta*weightdecay)
            blobnum=blobnum+1
    
        value=[]
        row=[]
        col=[]
        labels[:]=0
        bindex=0
        eta=eta*etadecay
        onemalpha=onemalpha*onemalphadecay
        numupdates=numupdates+1
        numsinceupdates=numsinceupdates+1
        if numupdates >= nextprint:
          net.save(sys.argv[7]+"."+str(numupdates))
          now=time.time()
          print "%4u  %10.3f  %8.4g  %8.4g  %10u  %8.5g  %8.5g"%(passnum,now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,myeta,myalpha)
          nextprint=2*nextprint
          numsinceupdates=0
          sumsinceloss=0
  
now=time.time()
print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
net.save(sys.argv[7])

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');
# pass     delta t   average     since     example       eta     alpha           
#                       loss      last     counter                               
#    0       2.640     11.29     11.29        1500         1       0.5           
#    0       4.930     10.26     9.241        3000         1    0.5004           
#    0       9.332      9.04     7.816        6000         1    0.5012           
#    0      17.043     7.946     6.852       12000         1   0.50279           
#    0      32.105     7.174     6.401       24000         1   0.50597           
#    0      62.384     6.691     6.209       48000         1   0.51225           
#    0     121.848     6.391     6.091       96000         1   0.52459           
#    0     240.958     6.148     5.905      192000         1   0.54832           
#    0     478.252     5.963     5.778      384000         1    0.5923           
#    0     950.847     5.846     5.728      768000         1   0.66783           
#    0    1887.179     5.745     5.645     1536000         1    0.7795           
#    0    3760.148     5.668      5.59     3072000         1   0.90284           
#    0    7466.047     5.597     5.527     6144000         1      0.92           
#    0   14895.092     5.511     5.425    12288000         1      0.92           
#    0   29752.632     5.431     5.351    24576000         1      0.92           
#    0   59788.722     5.362     5.293    49152000         1      0.92           
