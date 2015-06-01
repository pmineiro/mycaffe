#! /usr/bin/env python

import caffe
import numpy as np
import sys
import time

alpha=0.9
eta=1
decay=0.99999

np.random.seed(8675309)

vocabsize=80000
batchsize=250

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

print "%11s\t%11s\t%11s\t%11s\t%11s"%("delta t","average","since","example","learning")
print "%11s\t%11s\t%11s\t%11s\t%11s"%("","loss","last","counter","rate")

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
              print "%11.4f\t%11.4f\t%11.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
              nextprint=2*nextprint
              numsinceupdates=0
              sumsinceloss=0


now=time.time()
print "%11.4f\t%11.4f\t%11.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)

#    delta t         average           since         example        learning
#                       loss            last         counter            rate
#     1.3527         11.2905         11.2905             250         0.99999
#     2.5980         11.2852         11.2799             500         0.99998
#     5.3183         11.2681         11.2510            1000         0.99996
#    10.3841         11.2028         11.1375            2000         0.99992
#    20.4938         10.8986         10.5944            4000         0.99984
#    41.2812         10.2835          9.6685            8000         0.99968
#    79.0907          9.6025          8.9214           16000         0.99936
#   153.3155          9.0031          8.4038           32000        0.998721
#   302.6071          8.5450          8.0869           64000        0.997443
#   600.6641          8.2171          7.8892          128000        0.994893
#  1544.6629          7.9707          7.7244          256000        0.989812
#  4058.7562          7.7588          7.5468          512000        0.979728
#  8719.9198          7.5793          7.3998         1024000        0.959867
# 19143.7868          7.4235          7.2678         2048000        0.921345
# 43751.5034          7.2776          7.1317         4096000        0.848877
# 96104.8541          7.1259          6.9742         8192000        0.720592    
