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
# which ids to use for training
#-------------------------------------------------

def trainfilter(idnum):
  return hashlib.md5(idnum).hexdigest()[-1] != "f"

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
bindex=0
tokens=[]

for passes in range(65536):
  keep=0

  for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
    if trainfilter(str(docid)):
      goodparagraphs = [n for n in range (len (paragraphs))
                          if len (paragraphs[n].split ()) > 0.75 * numpos]
    
      if len (goodparagraphs) < 4:
        continue

      keep += 1

      if keep > numdocs:
        break

#      random.shuffle (goodparagraphs)

      for n in goodparagraphs[0:3]:
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

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./tongtrain.py
#   delta average   since     acc example pass     learning
#       t    loss    last   since counter  num         rate
# 18.116s   9.213   9.213   0.100    1001    3    9.951e-03
# 19.984s   9.213   9.212   0.400    2002    3    9.901e-03
# 23.542s   9.213   9.213   0.100    4004    3    9.803e-03
# 30.733s   9.213   9.213   0.125    8008    3    9.611e-03
# 44.818s   9.213   9.213   0.112     16K    3    9.237e-03
#  1.214m   9.212   9.211   0.087     32K    4    8.533e-03
#  2.175m   9.212   9.211   0.084     64K    5    7.283e-03
#  4.032m   9.211   9.210   0.109    128K    7    5.312e-03
#  7.707m   9.209   9.207   0.143    256K   11    2.844e-03
# 15.045m   9.205   9.201   0.197    512K   20    8.604e-04
# 29.658m   9.197   9.190   0.231   1025K   37    1.584e-04
# 58.851m   9.184   9.171   0.332   2050K   71    1.003e-04
#  1.958h   9.100   9.015   1.040   4100K  140    1.000e-04
#  3.902h   6.292   3.485  61.519   8200K  276    1.000e-04
#  7.950h   3.158   0.023  99.991     16M  550    1.000e-04
# 
