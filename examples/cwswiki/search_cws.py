import bz2
import caffe
import hashlib
import h5py
import math
import numpy as np
import os
import pyvw
import random
import sys
import time

import CaffeFinetuner 
import DocGenerator
import SentenceSelector

from Pretty import nicetime, nicecount

random.seed(69)
np.random.seed(8675309)

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

tagcutoff=int(os.environ['tagcutoff'])
tokencutoff=int(os.environ['tokencutoff'])
windowsize=int(os.environ['windowsize'])
embedd=int(os.environ['embedd'])
batchsize=int(os.environ['batchsize'])
searnalpha=float(os.environ['searnalpha'])
searneta=float(os.environ['searneta'])
searnetamin=min(searneta,float(os.environ['searnetamin']))
searnetadecay=float(os.environ['searnetadecay'])
searnvweta=float(os.environ['searnvweta'])
weightdecay=float(os.environ['weightdecay'])
labelnoise=float(os.environ['labelnoise'])
numtags=int(os.environ['numtags'])
searnmaxshufbuf=int(os.environ['searnmaxshufbuf'])
finetunedelay=int(os.environ['finetunedelay'])

class Example:
  def __init__ (self, partslabels, numtags):
    self.parts = partslabels[0]
    self.labels = [ l for l in partslabels[1] if l < numtags ]

  def __iter__ (self):
    yield self

def random_sublist(lst, length):
  start=random.randrange(1+len(lst)-length)
  return lst[start:start+length]

#-------------------------------------------------
# which ids to use for searn
#-------------------------------------------------

def searnfilter(idnum):
  hd=hashlib.md5(idnum).hexdigest()
  return hd[-1] != "a" and hd[-1] != "f"

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

lrs=dict()
lrs['embedding']=4
lrs[('conv1',0)]=4
lrs[('conv1',1)]=8
lrs[('conv2',0)]=3
lrs[('conv2',1)]=6
lrs[('conv3',0)]=2
lrs[('conv3',1)]=4
lrs[('ip1',0)]=1
lrs[('ip1',1)]=2
lrs[('ip2',0)]=1
lrs[('ip2',1)]=2
lrs[('lastip',0)]=1
lrs[('lastip',1)]=0

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.embedding.h5f')

caffe.set_mode_gpu()
net = caffe.Net(protofilename, netfilename, caffe.TRAIN)
if searnalpha > 0:
  momnet = caffe.Net(protofilename, netfilename, caffe.TRAIN)
with h5py.File(embeddingfilename) as f:
  embedding=np.copy(f['embedding'])

params = { 'lrs': lrs, 'net': net, 'embedding': embedding, 
           'windowsize': windowsize, 'embedd': embedd, 'numtags': numtags, 
           'batchsize': batchsize, 'labelnoise': labelnoise, 
           'alpha': searnalpha, 'eta': searneta, 'weightdecay': weightdecay }

if searnalpha > 0:
  params['momnet'] = momnet
  params['momembeddiff'] = momembeddiff

finetuner = CaffeFinetuner.CaffeFinetuner(params)

vw = pyvw.vw('--invariant --quiet --noconstant --search 2 --search_task hook -b 20 -q pq --cubic npq -l %g'%(searnvweta))

task_data = { 'test': False, 'finetuner': finetuner, 
              'finetunedelay': finetunedelay }
task = vw.init_search_task(SentenceSelector.SentenceSelector, task_data)

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s %9s %9s %9s %9s %7s %4s %9s"%("delta","excess","since","optimal","since","example","pass","learning") 
print "%7s %9s %9s %9s %9s %7s %4s %9s"%("t","loss","last","loss","last","counter","num","rate") 

start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
sumoptimalloss=0 
sumsinceoptimalloss=0 
nextprint=1 
shufbuf=[]

for passes in range(24):
  batchnum=0
  for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
    if docid in id2cat and searnfilter(str(docid)):
      parts = [
        [ tokennum[t] if t in tokennum else 0 
          for t in random_sublist(s, windowsize) ]
        for s in DocGenerator.sentences(paragraphs) if len(s) >= windowsize 
      ]

      if len(parts) < 2:
        continue

      labels = id2cat[docid]

      if len(shufbuf) < searnmaxshufbuf:
        shufbuf.append((parts,labels))
      else:
        index=random.randrange(searnmaxshufbuf)
        dq=shufbuf[index]
        shufbuf[index]=(parts,labels)
        task.learn(Example(dq,numtags))

        loss=(task.saveloss-task.saveoptimalloss)
        optimalloss=task.saveoptimalloss

        sumloss+=loss/batchsize
        sumsinceloss+=loss/batchsize
        sumoptimalloss+=optimalloss/batchsize
        sumsinceoptimalloss+=optimalloss/batchsize
    
        numupdates+=float(len(dq[0]))/batchsize
        numsinceupdates+=float(len(dq[0]))/batchsize
        if numupdates >= nextprint:
          now=time.time() 
          print "%7s %9.5f %9.5f %9.5f %9.5f %7s %4u %9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,sumoptimalloss/numupdates,sumsinceoptimalloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 
          nextprint=2*nextprint 
          numsinceupdates=0 
          sumsinceloss=0 
          sumsinceoptimalloss=0 

now=time.time() 
print "%7s %9.5f %9.5f %9.5f %9.5f %7s %4u %9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,sumoptimalloss/numupdates,sumsinceoptimalloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 

# GLOG_minloglevel=5 PYTHONPATH=../../python:../../../vowpal_wabbit/python python ./search_cws.py
# using precomputed id2cat
#   delta    excess     since   optimal     since example pass  learning
#       t      loss      last      loss      last counter  num      rate
#  2.029m   0.01867   0.01867   0.06286   0.06286    5091    0 1.000e-04
#  6.994m   0.01530   0.01181   0.05173   0.04018     10K    0 1.000e-04
# 12.543m   0.01632   0.01734   0.05954   0.06731     20K    0 1.000e-04
# 26.594m   0.01686   0.01741   0.06744   0.07539     40K    0 1.000e-04
# 52.312m   0.01775   0.01863   0.06808   0.06871     80K    0 1.000e-04
#  1.625h   0.01833   0.01890   0.07039   0.07271    160K    0 1.000e-04
# *** starting fine-tuning ***
#  3.067h   0.01826   0.01819   0.07224   0.07408    320K    0 1.000e-04
#  5.926h   0.01825   0.01824   0.07359   0.07494    640K    0 1.000e-04
# 11.999h   0.01834   0.01843   0.07262   0.07165   1280K    0 1.000e-04
# 
