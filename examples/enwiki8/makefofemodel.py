#! /usr/bin/env python

import caffe
import numpy as np
import sys
import time

alpha=0.9
eta=1.0
decay=0.99999

np.random.seed(8675309)

vocabsize=80000
batchsize=500 

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
      labels[bindex]=int(yx[0])-2
  
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

#     delta t         average           since         example        learning   
#                        loss            last         counter            rate   
#      2.0940         11.2902         11.2902             500         0.99999   
#      3.8769         11.2853         11.2804            1000         0.99998   
#      7.5326         11.2677         11.2501            2000         0.99996   
#     14.6913         11.2055         11.1433            4000         0.99992   
#     28.9187         10.8767         10.5479            8000         0.99984   
#     56.1917         10.1833          9.4900           16000         0.99968   
#    110.5309          9.5003          8.8172           32000         0.99936   
#    219.2732          8.8847          8.2691           64000        0.998721   
#    436.7229          8.4796          8.0744          128000        0.997443   
#    873.3536          8.1329          7.7863          256000        0.994893   
#   1841.8988          7.8306          7.5284          512000        0.989812   
#   3934.7942          7.5553          7.2799         1024000        0.979728   
#   8271.9772          7.3096          7.0639         2048000        0.959867   
#  16917.2846          7.0906          6.8715         4096000        0.921345   
#  34494.6173          6.8883          6.6861         8192000        0.848877
#  70322.8506          6.6791          6.4699        16384000        0.720592
