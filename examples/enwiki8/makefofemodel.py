#! /usr/bin/env python

import caffe
import numpy as np
import sys
import time

np.random.seed(8675309)

vocabsize=80000
batchsize=10000

invocabsize=vocabsize+2
outvocabsize=vocabsize+1

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

f=open(sys.argv[2],'r')

data=np.zeros((batchsize,1,1,invocabsize),dtype='f')
labels=np.zeros(batchsize,dtype='f')
bindex=0
start=time.time()
numsinceupdates=0
numupdates=0
sumloss=0
sumsinceloss=0
nextprint=1

alpha=0.9
eta=0.4
decay=0.9999

for ii in range(10):
  f.seek(0,0)
  for line in f:
      yx=[word for word in line.split(' ')]
      labels[bindex]=int(yx[0])-1
  
      for word in yx[1:]:
          iv=[subword for subword in word.split(':')]
          data[bindex,0,0,int(iv[0])-1]=float(iv[1])
      
      bindex=bindex+1
  
      if bindex >= batchsize:
          net.set_input_arrays(data,labels)
          res=net.forward()
          sumloss+=res['loss'][0,0,0,0]
          sumsinceloss+=res['loss'][0,0,0,0]
          net.backward()
          for (layer,momlayer) in zip(net.layers,momnet.layers):
            # TODO: skip input layer (?)
            for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
              momblob.data[:]=alpha*momblob.data[:]+eta*blob.diff;
              blob.data[:]-=momblob.data[:]
          eta=eta*decay
          data[:]=0
          labels[:]=0
          bindex=0
          numupdates=numupdates+1
          numsinceupdates=numsinceupdates+1
          if numupdates >= nextprint:
              now=time.time()
              print "%10.4f\t%10.4f\t%10.4f\t%10u"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates)
              nextprint=2*nextprint
              numsinceupdates=0
              sumsinceloss=0
