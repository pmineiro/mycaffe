import bz2
import caffe
from caffe import layers as L
from caffe import params as P
import exceptions
import hashlib
import h5py
import math
import numpy as np
import os
import random
import sys
import time

import CaffeFinetuner 
import DocGenerator
from Pretty import nicetime, nicecount

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

tagcutoff=int(os.environ['TAGCUTOFF'])
tokencutoff=int(os.environ['TOKENCUTOFF'])
windowsize=int(os.environ['WINDOWSIZE'])
embedd=int(os.environ['EMBEDD'])
batchsize=int(os.environ['BATCHSIZE'])
numconvk=int(os.environ['NUMCONVK'])
alpha=float(os.environ['ALPHA'])
eta=float(os.environ['ETA'])
etadecay=float(os.environ['ETADECAY'])
weightdecay=float(os.environ['WEIGHTDECAY'])
labelnoise=float(os.environ['LABELNOISE'])
numtags=int(os.environ['NUMTAGS'])
maxshufbuf=int(os.environ['MAXSHUFBUF'])

#-------------------------------------------------
# which ids to use for pretraining
#-------------------------------------------------

def pretrainfilter(idnum):
  md5 = hashlib.md5()
  md5.update (idnum)
  return md5.hexdigest ()[-1] == "a"

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

#-------------------------------------------------
# read label (tag) information
#-------------------------------------------------

tagnum=dict()
with open('taghisto', 'r') as f:
  for line in f:
    tc=line.split('\t')
    if int(tc[1]) < tagcutoff:
      break
    tagnum[tc[0]]=len(tagnum)

maxtags = len(tagnum)

#-------------------------------------------------
# read id2cat information
#-------------------------------------------------

id2cat=dict()
try:
  with open('id2cat.precomputed.txt','r') as f:
    print "using precomputed id2cat"
    for line in f:
      lc=line.split('\t')
      id2cat[int(lc[0])]=[int(c) for c in lc[1:] if not c.isspace()]
except exceptions.IOError:
  id2catstart=time.time()
  sys.stdout.write('computing id2cat ...')
  with bz2.BZ2File('enwiki-20150901.id2cat.txt.bz2', 'rb') as f:
    for line in f:
      idtags=line.split('\t')
      if pretrainfilter (idtags[0]):
        goodtags=[]
        for tag in idtags[1:]:
          if tag in tagnum:
            goodtags.append(tagnum[tag])
        id2cat[int(idtags[0])]=goodtags
  with open('id2cat.precomputed.txt','w') as f:
    for key, value in id2cat.iteritems():
      f.write('%u\t%s\n'%(key,'\t'.join([str(v) for v in value])))
  print " %g seconds.  len(id2cat)  = %d"%(float(time.time()-id2catstart), len(id2cat))

lrs=dict()
lrs['embedding']=10
lrs[('conv1',0)]=5
lrs[('conv1',1)]=10
lrs[('conv2',0)]=4
lrs[('conv2',1)]=8
lrs[('conv3',0)]=3
lrs[('conv3',1)]=6
lrs[('ip1',0)]=1
lrs[('ip1',1)]=2
lrs[('ip2',0)]=1
lrs[('ip2',1)]=2
lrs[('lastip',0)]=1
lrs[('lastip',1)]=0

random.seed(69)
np.random.seed(8675309)

#-------------------------------------------------
# define model
#-------------------------------------------------

def net(batchsize,embedd,windowsize,numtags,numconvk):
  n = caffe.NetSpec()
  n.data, n.bogus = L.MemoryData(batch_size=batchsize, channels=(windowsize*embedd+numtags), height=1, width=1, ntop=2)
  n.prefeatures, n.labels = L.Slice(n.data, slice_param=dict(axis=1,slice_point=[windowsize*embedd]),ntop=2)
  n.features = L.Reshape(n.prefeatures, reshape_param=dict(shape=dict(dim=[0,embedd,1,windowsize])))
  n.conv1 = L.Convolution(n.features, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool1 = L.Pooling(n.conv1, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.conv2 = L.Convolution(n.pool1, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool2 = L.Pooling(n.conv2, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.conv3 = L.Convolution(n.pool2, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool3 = L.Pooling(n.conv3, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.ip1 = L.InnerProduct(n.pool3, num_output=200)
  n.relu1 = L.ReLU(n.ip1, in_place=True)
  n.ip2 = L.InnerProduct(n.ip1, num_output=200)
  n.relu2 = L.ReLU(n.ip2, in_place=True)
  n.lastip = L.InnerProduct(n.relu2, num_output=numtags)
  n.loss = L.SigmoidCrossEntropyLoss(n.lastip, n.labels)
  n.silence = L.Silence(n.bogus,ntop=0)
  return n.to_proto()

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.embedding.h5f')

with open(protofilename,'w') as f:
  f.write("force_backward: true\n")
  f.write(str(net(batchsize,embedd,windowsize,numtags,numconvk)))

caffe.set_mode_gpu()
net = caffe.Net(protofilename, caffe.TRAIN)
if (alpha > 0):
  momnet = caffe.Net(protofilename, caffe.TRAIN)

#-------------------------------------------------
# compute prior for bias
#-------------------------------------------------

labelcounts=np.zeros(maxtags,dtype='d')

try:
  with open('pretrain.labelcounts.precomputed.txt', 'r') as f:
    print "using precomputed label counts"
    for line in f:
      lc=line.split('\t')
      labelcounts[int(lc[0])]=float(lc[1])
except exceptions.IOError:
  labelcountstart=time.time()
  examplecount=0
  sys.stdout.write('computing label counts ...')

  for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
    if docid in id2cat:
      labels=id2cat[docid]

      for s in DocGenerator.sentences(paragraphs):
        if len(s) < windowsize:
          continue

        examplecount+=1

        for l in labels:
          labelcounts[l]+=1

  labelcounts=(1+labelcounts)/(maxtags+examplecount)

  with open('pretrain.labelcounts.precomputed.txt', 'w') as f:
    for ii in range(maxtags):
      f.write('%u\t%g\n'%(ii,labelcounts[ii]))

  print " %g seconds.  examplecount = %d"%(float(time.time()-labelcountstart), examplecount)

#-------------------------------------------------
# initialize finetuner
#-------------------------------------------------

for layer in net.layers:
  blobnum=0 
  for blob in layer.blobs:
    if blobnum == 0:
      blob.data[:]=(1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:]=0
    blobnum=blobnum+1 

if (alpha > 0):
  for layer in momnet.layers:
    for blob in layer.blobs:
      blob.data[:]=0

for (name,layer) in zip(net._layer_names,net.layers):
  blobnum=0 
  for blob in layer.blobs:
    if name == "lastip" and blobnum == 1:
      blob.data[:]=np.log(np.divide(labelcounts[0:numtags],1-labelcounts[0:numtags]))
    blobnum=blobnum+1 

embedding=(1.0/math.sqrt(embedd))*np.random.standard_normal(size=(embedd,numtokens+1)).astype(float)
if (alpha > 0):
  momembeddiff=np.zeros(shape=(embedd,numtokens+1),dtype='f')

params = { 'lrs': lrs, 'net': net, 'embedding': embedding, 
           'windowsize': windowsize, 'embedd': embedd, 'numtags': numtags, 
           'batchsize': batchsize, 'labelnoise': labelnoise, 
           'alpha': alpha, 'eta': eta, 'weightdecay': weightdecay }

if alpha > 0:
  params['momnet'] = momnet
  params['momembeddiff'] = momembeddiff

finetuner = CaffeFinetuner.CaffeFinetuner (params)

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta t","average","since","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("","loss","last","counter","num","rate") 

start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
nextprint=1 
shufbuf=[]

for passes in range(24):
  batchnum=0
  for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
    if docid in id2cat:
      for s in DocGenerator.sentences(paragraphs):
        if len(s) < windowsize:
          continue

        if len(shufbuf) < maxshufbuf:
          shufbuf.append((s,docid))
        else:
          index=random.randrange(maxshufbuf)
          dq=shufbuf[index]
          shufbuf[index]=(s,docid)

          labels=id2cat[dq[1]]
          tokstart=random.randrange(1+len(dq[0])-windowsize)
          tokens = [ tokennum[t] if t in tokennum else 0 
                     for t in dq[0][tokstart:tokstart+windowsize] ]

          rv = finetuner.update (tokens, labels)
    
          if rv[0]:
            sumloss+=rv[1]
            sumsinceloss+=rv[1]
    
            numupdates+=1
            numsinceupdates+=1
            finetuner.eta*=etadecay
    
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
              print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 
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
print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 

# using precomputed id2cat
# using precomputed label counts
# delta t average   since example pass     learning
#            loss    last counter  num         rate
# 20.620s   1.596   1.596    5001    0    9.990e-04
# 23.365s   1.619   1.643     10K    0    9.980e-04
# 28.489s   1.589   1.559     20K    0    9.960e-04
# 38.018s   1.619   1.648     40K    0    9.920e-04
# 58.517s   1.633   1.648     80K    0    9.841e-04
#  1.597m   1.640   1.646    160K    0    9.685e-04
#  2.767m   1.599   1.558    320K    0    9.380e-04
#  5.072m   1.589   1.579    640K    0    8.798e-04
#  9.646m   1.530   1.472   1280K    0    7.740e-04
# 18.812m   1.505   1.480   2560K    0    5.991e-04
# 37.171m   1.471   1.436   5121K    1    3.590e-04
#  1.202h   1.400   1.329     10M    2    1.289e-04
#  2.352h   1.333   1.266     20M    4    1.661e-05
#  4.651h   1.292   1.251     40M    9    2.757e-07
# ...
