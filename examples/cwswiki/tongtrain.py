import bz2
import caffe
import hashlib
import h5py
import math
import numpy as np
import os
import random
import string
import sys
import time

import pdb

import DocGenerator
from Pretty import nicetime, nicecount
import TongNet

random.seed(8675309)
np.random.seed(90210)

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

numtokens=int(os.environ['numtokens'])
numpos=int(os.environ['numpos'])
embedd=int(os.environ['embedd'])
batchsize=int(os.environ['batchsize'])
numconvk=int(os.environ['numconvk'])
numip1=int(os.environ['numip1'])
numip2=int(os.environ['numip2'])
alpha=float(os.environ['alpha'])
eta=float(os.environ['eta'])
etamin=min(eta,float(os.environ['etamin']))
etadecay=float(os.environ['etadecay'])
weightdecay=float(os.environ['weightdecay'])
numdocs=int(os.environ['numdocs'])
maxshufbuf=int(os.environ['maxshufbuf'])

#-------------------------------------------------
# read token information
#-------------------------------------------------

tokennum=dict()
with open('tokenhisto', 'r') as f:
  for line in f:
    tc=line.split('\t')
    tokennum[tc[0]]=1+len(tokennum)
    if len(tokennum) >= numtokens:
      break

#-------------------------------------------------
# initialize net
#-------------------------------------------------

lrs=dict()
lrs['embedding']=1
lrs[('conv1',0)]=1
lrs[('conv1',1)]=2
lrs[('ip1',0)]=1
lrs[('ip1',1)]=2
lrs[('ip2',0)]=1
lrs[('ip2',1)]=2
lrs[('lastip',0)]=1
lrs[('lastip',1)]=0

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'tongnet.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'tongnet.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'tongnet.embedding.h5f')

with open(protofilename,'w') as f:
  f.write("force_backward: true\n")
  f.write(str(TongNet.net(batchsize,embedd,numpos,numdocs,numconvk,numip1,numip2)))

caffe.set_mode_gpu()
net = caffe.Net(protofilename, caffe.TRAIN)
if alpha > 0:
  momnet = caffe.Net(protofilename, caffe.TRAIN)

for layer in net.layers:
  blobnum=0 
  for blob in layer.blobs:
    if blobnum == 0 and eta > 0:
      blob.data[:]=(1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:]=0
    blobnum=blobnum+1 

if alpha > 0:
  for layer in momnet.layers:
    for blob in layer.blobs:
      blob.data[:]=0

for (name,layer) in zip(net._layer_names,net.layers):
  blobnum=0 
  for blob in layer.blobs:
    if name == "lastip" and blobnum == 1:
      blob.data[:]=0
    blobnum=blobnum+1 

embedding=np.random.standard_normal(size=(embedd,numtokens+1)).astype(float)
for n in range(numtokens+1):
  embedding[:,n]*=1.0/np.linalg.norm(embedding[:,n])

if alpha > 0:
  momembeddiff=np.zeros(shape=(embedd,numtokens+1),dtype='f')

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta","average","since","acc","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("t","loss","last","since","counter","num","rate") 

starttime=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
sumsinceacc=0 
nextprint=1 
shufbuf=[]
labels=np.zeros((batchsize,1,1,1),dtype='f')
inputs=np.zeros((batchsize,embedd,1,numpos),dtype='f')
testlabels=np.zeros((batchsize,1,1,1),dtype='f')
testinputs=np.zeros((batchsize,embedd,1,numpos),dtype='f')
bindex=0
tokens=[]

for passes in range(65536):
  keep=0

  # train
  for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
    goodparagraphs = [n for n in range (len (paragraphs))
                        if len (paragraphs[n].split ()) > 0.75 * numpos]

    if len (goodparagraphs) < 4:
      continue

    keep += 1

    if keep > numdocs:
      break

    # reserve for testing
    goodparagraphs.pop (0)

    random.shuffle (goodparagraphs)

    for n in goodparagraphs[0:1]:
      if len(shufbuf) < maxshufbuf:
        shufbuf.append((paragraphs[n],keep))
      else:
        index=random.randrange(maxshufbuf)
        dq=shufbuf[index]
        shufbuf[index]=(paragraphs[n],keep)

        inputs[bindex,:,0,:] = 0

        encoded= [ 
            tokennum[w] if w in tokennum else 0
            for w in [ t.strip (string.punctuation) for t in dq[0].split() ]
        ]

        tokens.append (encoded[0:2*numpos])

        stepsperpos=float(len(tokens[-1]))/numpos
        for p in range(numpos):
          start=stepsperpos*p
          end=min(stepsperpos*(p+1),len(tokens[-1]))
          while end > start:
            origtokenpos=int(math.floor(start))
            t=tokens[-1][origtokenpos]
            amount=min(1+math.floor(start),end)-start
            inputs[bindex,:,0,p] += (amount / stepsperpos) * embedding[:,t]
            start=min(1+math.floor(start),end)

        labels[bindex,0,0,0] = dq[1]-1

        bindex += 1

        if bindex >= batchsize:
          net.set_input_arrays (inputs, labels)
          res = net.forward ()

          sumloss+=res['loss']
          sumsinceloss+=res['loss']
          sumsinceacc+=res['acc']

          net.backward ()

          # TODO: data_diff, update embedding

          data_diff = net.blobs['features'].diff
          # (batchsize,embedd,1,numpos)
          if alpha > 0:
            momembeddiff *= alpha
            for ii in range(batchsize):
              stepsperpos=float(len(tokens[ii]))/numpos
              for p in range(numpos):
                start=stepsperpos*p
                end=min(stepsperpos*(p+1),len(tokens[ii]))
                while end > start:
                  origtokenpos=int(math.floor(start))
                  t=tokens[ii][origtokenpos]
                  amount=min(1+math.floor(start),end)-start
                  momembeddiff[:,t] += (
                    lrs['embedding'] * eta *
                    (amount / stepsperpos) * data_diff[ii,:,0,p]
                  )
                  start=min(1+math.floor(start),end)

            embedding -= momembeddiff
          else:
            for ii in range(batchsize):
              stepsperpos=float(len(tokens[ii]))/numpos
              for p in range(numpos):
                start=stepsperpos*p
                end=min(stepsperpos*(p+1),len(tokens[ii]))
                while end > start:
                  origtokenpos=int(math.floor(start))
                  t=tokens[ii][origtokenpos]
                  amount=min(1+math.floor(start),end)-start
                  embedding[:,t] -= (
                    lrs['embedding'] * eta *
                    (amount / stepsperpos) * data_diff[ii,:,0,p]
                  )
                  start=min(1+math.floor(start),end)

          if alpha > 0:
            for (name,layer,momlayer) in zip (net._layer_names,
                                              net.layers,
                                              momnet.layers):
               blobnum = 0 
               for (blob,momblob) in zip (layer.blobs,momlayer.blobs): 
                 myeta = lrs[(name,blobnum)] * eta 
                 momblob.data[:] *= alpha
                 momblob.data[:] += myeta * blob.diff
                 blob.data[:] -= momblob.data
                 if weightdecay > 0:
                   blob.data[:] *= (1.0 - weightdecay * myeta)
                 blobnum = blobnum + 1 
          else:
            for (name,layer) in zip (net._layer_names,
                                     net.layers):
               blobnum = 0 
               for blob in layer.blobs:
                 myeta = lrs[(name,blobnum)] * eta 
                 blob.data[:] -= myeta * blob.diff
                 if weightdecay > 0:
                   blob.data[:] *= (1.0 - weightdecay * myeta)
                 blobnum = blobnum + 1 

          tokens=[]
          bindex=0

          numupdates+=1
          numsinceupdates+=1
          eta=etadecay*eta+(1.0-etadecay)*etamin

          if numupdates >= nextprint:
            try:
              os.remove(netfilename+"."+str(numupdates)) 
            except:
              pass

            net.save(netfilename+"."+str(numupdates)) 
            try:
              os.remove(embeddingfilename+"."+str(numupdates)) 
            except:
              pass

            h5f=h5py.File(embeddingfilename+"."+str(numupdates)) 
            h5f.create_dataset('embedding',data=embedding)
            h5f.close() 
            now=time.time() 
            print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-starttime)),sumloss/numupdates,sumsinceloss/numsinceupdates,100*(sumsinceacc/numsinceupdates),nicecount(numupdates*batchsize),passes,eta) 
            nextprint=2*nextprint 
            numsinceupdates=0 
            sumsinceloss=0 
            sumsinceacc=0 

            if numupdates * batchsize >= numdocs:
              # test
              testsumloss=0
              testsumacc=0
              testnumupdates=0
              testbindex=0
              testkeep=0
              for testdocid, testparagraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
                goodparagraphs = [n for n in range (len (testparagraphs))
                                    if len (testparagraphs[n].split ()) > 0.75 * numpos]

                if len (goodparagraphs) < 4:
                  continue

                testkeep += 1

                if testkeep > numdocs:
                  break

                testinputs[testbindex,:,0,:] = 0

                encoded= [ 
                    tokennum[w] if w in tokennum else 0
                    for w in [ t.strip (string.punctuation) 
                               for t in testparagraphs[goodparagraphs[0]].split() ]
                ]

                encoded=encoded[0:2*numpos]

                stepsperpos=float(len(encoded))/numpos
                for p in range(numpos):
                  start=stepsperpos*p
                  end=min(stepsperpos*(p+1),len(encoded))
                  while end > start:
                    origtokenpos=int(math.floor(start))
                    t=encoded[origtokenpos]
                    amount=min(1+math.floor(start),end)-start
                    testinputs[testbindex,:,0,p] += (amount / stepsperpos) * embedding[:,t]
                    start=min(1+math.floor(start),end)

                testlabels[testbindex,0,0,0] = testkeep-1

                testbindex += 1

                if testbindex >= batchsize:
                  net.set_input_arrays (testinputs, testlabels)
                  res = net.forward ()

                  testsumloss+=res['loss']
                  testsumacc+=res['acc']
                  testnumupdates+=1
                  testbindex=0

              now=time.time() 
              print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9s"%(nicetime(float(now-starttime)),testsumloss/testnumupdates,testsumloss/testnumupdates,100*(testsumacc/testnumupdates),nicecount(testnumupdates*batchsize),passes,'(test)') 


net.save(netfilename)
try:
  os.remove(embeddingfilename)
except:
  pass
h5f=h5py.File(embeddingfilename)
h5f.create_dataset('embedding',data=embedding) 
h5f.close() 
now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-starttime)),sumloss/numupdates,sumsinceloss/numsinceupdates,100*(sumsinceacc/numsinceupdates),nicecount(numupdates*batchsize),passes,eta) 


# test
testsumloss=0
testsumacc=0
testnumupdates=0
testbindex=0
testkeep=0
for testdocid, testparagraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
  goodparagraphs = [n for n in range (len (testparagraphs))
		      if len (testparagraphs[n].split ()) > 0.75 * numpos]

  if len (goodparagraphs) < 4:
    continue

  testkeep += 1

  if testkeep > numdocs:
    break

  testinputs[testbindex,:,0,:] = 0

  encoded= [ 
      tokennum[w] if w in tokennum else 0
      for w in [ t.strip (string.punctuation) 
		 for t in testparagraphs[goodparagraphs[0]].split() ]
  ]

  encoded=encoded[0:2*numpos]

  stepsperpos=float(len(encoded))/numpos
  for p in range(numpos):
    start=stepsperpos*p
    end=min(stepsperpos*(p+1),len(encoded))
    while end > start:
      origtokenpos=int(math.floor(start))
      t=encoded[origtokenpos]
      amount=min(1+math.floor(start),end)-start
      testinputs[testbindex,:,0,p] += (amount / stepsperpos) * embedding[:,t]
      start=min(1+math.floor(start),end)

  testlabels[testbindex,0,0,0] = testkeep-1

  testbindex += 1

  if testbindex >= batchsize:
    net.set_input_arrays (testinputs, testlabels)
    res = net.forward ()

    testsumloss+=res['loss']
    testsumacc+=res['acc']
    testnumupdates+=1
    testbindex=0

now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7.3f\t%7s\t%4u\t%9s"%(nicetime(float(now-starttime)),testsumloss/testnumupdates,testsumloss/testnumupdates,100*(testsumacc/testnumupdates),nicecount(testnumupdates*batchsize),passes,'(test)') 

# still overfitting, but getting better ...
# GLOG_minloglevel=5 PYTHONPATH=../../python python ./tongtrain.py
#   delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
#  7.061s   6.920   6.920   0.500    1000   10    9.950e-03
#  8.357s   6.921   6.921   1.200    1000   10       (test)
# 10.494s   6.922   6.925   0.800    2000   11    9.900e-03
# 11.784s   6.921   6.921   1.200    1000   11       (test)
# 15.836s   6.918   6.914   0.850    4000   13    9.802e-03
# 17.147s   6.921   6.921   1.200    1000   13       (test)
# 25.288s   6.918   6.918   0.925    8000   17    9.607e-03
# 26.585s   6.920   6.920   1.200    1000   17       (test)
# 42.445s   6.916   6.915   0.925     16K   25    9.229e-03
# 43.679s   6.916   6.916   1.100    1000   25       (test)
#  1.239m   6.912   6.907   1.031     32K   41    8.518e-03
#  1.260m   6.911   6.911   1.200    1000   41       (test)
#  2.282m   6.909   6.906   1.397     64K   73    7.256e-03
#  2.303m   6.923   6.923   0.800    1000   73       (test)
#  4.383m   6.900   6.891   1.738    128K  137    5.265e-03
#  4.405m   6.925   6.925   0.900    1000  137       (test)
#  8.582m   6.880   6.860   2.744    256K  265    2.772e-03
#  8.603m   6.917   6.917   1.200    1000  265       (test)
# 16.891m   6.817   6.755   7.750    512K  521    7.690e-04
# 16.911m   6.889   6.889   1.100    1000  521       (test)
# 33.558m   6.684   6.550  19.338   1024K 1033    5.999e-05
# 33.579m   6.851   6.851   2.500    1000 1033       (test)
#  1.113h   6.554   6.424  23.010   2048K 2057    1.348e-06
#  1.113h   6.845   6.845   2.400    1000 2057       (test)
#  2.240h   6.474   6.395  23.609   4096K 4105    1.000e-06
#  2.241h   6.841   6.841   2.200    1000 4105       (test)
#  4.427h   6.399   6.325  23.926   8192K 8201    1.000e-06
#  4.428h   6.836   6.836   1.500    1000 8201       (test)
#  8.766h   6.206   6.012  27.130     16M 16393   1.000e-06
#  8.767h   6.921   6.921   3.200    1000 16393      (test)
# 17.433h   5.492   4.778  43.093     32M 32777   1.000e-06
# 17.433h   8.336   8.336   4.100    1000 32777      (test)
# 1.4484d   4.295   3.098  70.789     65M 65535   1.000e-06
# 1.4484d  23.398  23.398   6.700    1000 65535      (test)
