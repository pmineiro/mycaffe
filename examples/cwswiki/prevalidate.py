import bz2
import caffe
import exceptions
import hashlib
import h5py
import math
import md5
import numpy as np
import os
import random
import sys
import time

import pdb

import CaffeFinetuner 
import DocGenerator
from Pretty import nicetime, nicecount

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

tagcutoff=int(os.environ['tagcutoff'])
tokencutoff=int(os.environ['tokencutoff'])
windowsize=int(os.environ['windowsize'])
embedd=int(os.environ['embedd'])
batchsize=int(os.environ['batchsize'])
numtags=int(os.environ['numtags'])

def random_sublist(lst, length):
  start=random.randrange(1+len(lst)-length)
  return lst[start:start+length]

#-------------------------------------------------
# which ids to use for prevalidating
#-------------------------------------------------

def prevalidatefilter(idnum):
  return hashlib.md5(idnum).hexdigest()[-1] == "b"

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
with open('id2cat.precomputed.txt','r') as f:
  print "using precomputed id2cat"
  for line in f:
    lc=line.split('\t')
    id2cat[int(lc[0])]=[int(c) for c in lc[1:] if not c.isspace()]

random.seed(69)
np.random.seed(8675309)

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.embedding.h5f')

caffe.set_mode_gpu()
net = caffe.Net(protofilename, netfilename, caffe.TEST)
with h5py.File(embeddingfilename) as f:
  embedding=np.copy(f['embedding'])

params = { 'lrs': dict(), 'net': net, 'embedding': embedding, 
           'windowsize': windowsize, 'embedd': embedd, 'numtags': numtags, 
           'batchsize': batchsize, 'labelnoise': 0, 
           'alpha': 0, 'eta': 0, 'weightdecay': 0 }

finetuner = CaffeFinetuner.CaffeFinetuner(params)

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

for passes in range(1):
  batchnum=0
  for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
    if docid in id2cat and prevalidatefilter(str(docid)):

      scores=finetuner.predict (
        [ [ tokennum[t] if t in tokennum else 0 
            for t in random_sublist(s, windowsize) ]
          for s in DocGenerator.sentences(paragraphs) if len(s) >= windowsize ]
      )
      labels=id2cat[docid]

      loss=0
      for s in scores:
        thisloss=np.sum(np.log(1.0+np.exp(s)))-np.sum(s[ [ l for l in labels if l < numtags ] ])
        loss+=thisloss

      sumloss+=loss/batchsize
      sumsinceloss+=loss/batchsize
    
      numupdates+=float(len(scores))/batchsize
      numsinceupdates+=float(len(scores))/batchsize
      if numupdates >= nextprint:
        now=time.time() 
        print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 
        nextprint=2*nextprint 
        numsinceupdates=0 
        sumsinceloss=0 

now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 
