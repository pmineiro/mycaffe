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
decay=0.99999

#np.seterr(all='raise')

np.random.seed(8675309)

vocabsize=80000
embeddingsize=200
batchsize=250 

invocabsize=vocabsize+2
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

print "%11s\t%11s\t%11s\t%11s\t%11s"%("delta t","average","since","example","learning")
print "%11s\t%11s\t%11s\t%11s\t%11s"%("","loss","last","counter","rate")

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

          momembeddiff=alpha*momembeddiff+eta*sdtransdiff;
          embedding=embedding-momembeddiff

          for (layer,momlayer) in zip(net.layers,momnet.layers):
            for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
              momblob.data[:]=alpha*momblob.data[:]+eta*blob.diff
              blob.data[:]-=momblob.data[:]

          eta=eta*decay
          value=[]
          row=[]
          col=[]
          labels[:]=0
          bindex=0
          numupdates=numupdates+1
          numsinceupdates=numsinceupdates+1
          if numupdates >= nextprint:
              now=time.time()
              print "%11.4f\t%11.4f\t%11.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
              nextprint=2*nextprint
              numsinceupdates=0
              sumsinceloss=0


now=time.time()
print "%11.4f\t%11.4f\t%11.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)

#    delta t         average           since         example        learning
#                       loss            last         counter            rate
#     1.0651         11.2877         11.2877             250         0.99999
#     2.0385         11.2835         11.2793             500         0.99998
#     3.9110         11.2670         11.2505            1000         0.99996
#     7.6542         11.2011         11.1351            2000         0.99992
#    14.8690         10.8806         10.5601            4000         0.99984
#    28.4091         10.1851          9.4896            8000         0.99968
#    55.4836          9.4967          8.8084           16000         0.99936
#   109.7036          8.9067          8.3167           32000        0.998721
#...
