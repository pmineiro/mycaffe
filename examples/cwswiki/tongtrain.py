import bz2
import caffe
import hashlib
import h5py
import math
import numpy as np
import os
import random
import sys
import time

import DocGenerator
from Pretty import nicetime, nicecount
import TongNet

random.seed(69)
np.random.seed(8675309)

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

tokencutoff=int(os.environ['tokencutoff'])
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
    if int(tc[1]) < tokencutoff:
      break
    tokennum[tc[0]]=1+len(tokennum)

numtokens = len(tokennum)
print 'tokencutoff = %u, numtokens = %u'%(tokencutoff,numtokens)

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

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'tongtrain.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'tongtrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'tongtrain.embedding.h5f')

with open(protofilename,'w') as f:
  f.write("force_backward: true\n")
  f.write(str(TongNet.net(batchsize,embedd,numpos,numdocs,numconvk)))

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
      blob.data[:]=math.log(1e-6/(1.0-1e-6))
    blobnum=blobnum+1 

embedding=np.random.standard_normal(size=(embedd,numtokens+1)).astype(float)
for n in range(numtokens+1):
  embedding[:,n]*=1.0/np.linalg.norm(embedding[:,n])

if alpha > 0:
  momembeddiff=np.zeros(shape=(embedd,numtokens+1),dtype='f')

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta","average","since","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("t","loss","last","counter","num","rate") 

start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
nextprint=1 
shufbuf=[]
bogus = np.zeros ((batchsize,1,1,1),dtype='f')

for passes in range(24):
  bindex=0
  keep=0
  for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
    if trainfilter(str(docid)):
      goodparagraphs = [n for n in range (len (paragraphs))
                          if len (paragraphs[n].split ()) > 20]
    
      if len (goodparagraphs) < 4:
        skip += 1
        continue

      keep += 1

      if keep > numdocs:
        break

      random.shuffle (goodparagraphs)

      for n in goodparagraphs[0:3]:
        if len(shufbuf) < maxshufbuf:
          shufbuf.append((paragraphs[n],keep))
        else:
          index=random.randrange(maxshufbuf)
          dq=shufbuf[index]
          shufbuf[index]=(paragraphs[n],keep)

          inputs[bindex,:,0,0] = 0

          tokens = [ t.strip (string.punctuation) for t in dq[0].split () ]
          stepsperpos=float(len(tokens))/numpos
          for p in range(20):
            start=stepsperpos*p
            end=stepsperpos*(p+1)
            while end > start:
              origtokenpos=math.floor(start)
              t = tokennum[origtoken] if origtoken in tokennum else 0
              amount=min(1+math.floor(start),end)-start
              inputs[bindex,p*embedd:(p+1)*embedd,0,0] += amount * embedding[:,t]
              pdb.set_trace()
              start=min(1+math.floor(start),end)

          inputs[bindex,numpos*embedd:,0,0] = 0
          inputs[bindex,(numpos*embedd)+keep,0,0] = 1

          bindex += 1

          if bindex > batchsize:
            net.set_input_arrays (inputs, bogus)
            res = net.forward ()

            sumloss+=res['loss']
            sumsinceloss+=res['loss']
    
            net.backward ()

            # TODO: data_diff, update embedding

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
                   blob.data[:] *= (1.0 - weightdecay * myeta)
                   blobnum = blobnum + 1 

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
              print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 
              nextprint=2*nextprint 
              numsinceupdates=0 
              sumsinceloss=0 

net.save(netfilename)
try:
  os.remove(embeddingfilename)
except:
  pass
h5f=h5py.File(embeddingfilename)
h5f.create_dataset('embedding',data=embedding) 
h5f.close() 
now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 

# eta = 0
# using precomputed id2cat
# using precomputed label counts
#   delta average   since example pass     learning
#       t    loss    last counter  num         rate
# 18.176s   1.703   1.703    5001    0    0.000e+00
# 20.068s   1.718   1.733     10K    0    0.000e+00
# 23.399s   1.697   1.676     20K    0    0.000e+00
# 29.803s   1.726   1.756     40K    0    0.000e+00
# 41.569s   1.744   1.762     80K    0    0.000e+00
#  1.068m   1.755   1.767    160K    0    0.000e+00
#  1.794m   1.741   1.726    320K    0    0.000e+00
#  3.291m   1.756   1.771    640K    0    0.000e+00
#  6.270m   1.732   1.709   1280K    0    0.000e+00
# 11.811m   1.755   1.777   2560K    0    0.000e+00
# 22.759m   1.767   1.779   5121K    1    0.000e+00
# 44.247m   1.766   1.766     10M    2    0.000e+00
#  1.448h   1.769   1.771     20M    4    0.000e+00
#  2.862h   1.769   1.769     40M    9    0.000e+00
# ...

# eta = 0, initialized weights
# using precomputed id2cat
# using precomputed label counts
#   delta average   since example pass     learning
#       t    loss    last counter  num         rate
# 19.302s   1.703   1.703    5001    0    0.000e+00
# 21.283s   1.718   1.733     10K    0    0.000e+00
# 24.567s   1.697   1.676     20K    0    0.000e+00
# 30.752s   1.726   1.756     40K    0    0.000e+00
# 42.395s   1.744   1.762     80K    0    0.000e+00
#  1.062m   1.756   1.767    160K    0    0.000e+00
#  1.768m   1.741   1.726    320K    0    0.000e+00
#  3.207m   1.756   1.772    640K    0    0.000e+00
#  6.064m   1.732   1.709   1280K    0    0.000e+00
# 11.393m   1.755   1.777   2560K    0    0.000e+00
# 22.187m   1.767   1.779   5121K    1    0.000e+00
# 43.429m   1.766   1.766     10M    2    0.000e+00
#  1.431h   1.769   1.771     20M    4    0.000e+00
# ...

# eta > 0
# using precomputed id2cat
# using precomputed label counts
#   delta average   since example pass     learning
#       t    loss    last counter  num         rate
# 20.653s   1.703   1.703    5001    0    9.975e-02
# 23.586s   1.718   1.733     10K    0    9.950e-02
# 28.822s   1.697   1.676     20K    0    9.900e-02
# 38.608s   1.726   1.756     40K    0    9.802e-02
# 57.327s   1.744   1.762     80K    0    9.607e-02
#  1.586m   1.755   1.766    160K    0    9.230e-02
#  2.747m   1.739   1.724    320K    0    8.520e-02
#  5.101m   1.752   1.766    640K    0    7.259e-02
#  9.827m   1.702   1.651   1280K    0    5.269e-02
# 19.180m   1.686   1.670   2560K    0    2.777e-02
# 36.967m   1.667   1.648   5121K    1    7.715e-03
#  1.199h   1.647   1.627     10M    2    6.037e-04
#  2.371h   1.638   1.629     20M    4    1.353e-05
# ...
