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
import TheNet

from Pretty import nicetime, nicecount

random.seed(69)
np.random.seed(8675309)

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

tagcutoff=int(os.environ['tagcutoff'])
tokencutoff=int(os.environ['tokencutoff'])
windowsize=int(os.environ['windowsize'])
embedd=int(os.environ['embedd'])
batchsize=int(os.environ['searnbatchsize'])
numconvk=int(os.environ['numconvk'])
searnalpha=float(os.environ['searnalpha'])
searneta=float(os.environ['searneta'])
searnetamin=min(searneta,float(os.environ['searnetamin']))
searnetadecay=float(os.environ['searnetadecay'])
searnreplaybuf=int(os.environ['searnreplaybuf'])
searnvweta=float(os.environ['searnvweta'])
weightdecay=float(os.environ['weightdecay'])
labelnoise=float(os.environ['labelnoise'])
numtags=int(os.environ['numtags'])
searnmaxshufbuf=int(os.environ['searnmaxshufbuf'])
finetunedelay=int(os.environ['finetunedelay'])
prefixsentences=int(os.environ['prefixsentences'])

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

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'searn.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.embedding.h5f')

with open(protofilename,'w') as f:
  f.write("force_backward: true\n")
  f.write(str(TheNet.net(batchsize,embedd,windowsize,numtags,numconvk)))

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

task_data = { 'test': False, 'finetuner': finetuner, 
              'finetunedelay': finetunedelay }
vw = pyvw.vw('--invariant --quiet --search 2 --search_task hook --replay_c %u -b 20 -q pq --cubic npq -l %g'%(searnreplaybuf,searnvweta))
task = vw.init_search_task(SentenceSelector.SentenceSelector, task_data)

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s %7s %9s %9s %9s %9s %7s %4s %9s"%("delta","length","excess","since","oracle","since","example","pass","learning") 
print "%7s %7s %9s %9s %9s %9s %7s %4s %9s"%("t","since","loss","last","loss","last","counter","num","rate") 

start=time.time()
numsinceupdates=0 
numupdates=0
numsincelenupdates=0 
sumloss=0 
sumsinceloss=0 
sumoptimalloss=0 
sumsinceoptimalloss=0 
sumsincelenupdates=0
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
        shufbuf.append((parts[:prefixsentences],labels))
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
        sumsincelenupdates+=float(len(dq[0]))
    
        numupdates+=float(len(dq[0]))/batchsize
        numsinceupdates+=float(len(dq[0]))/batchsize
        numsincelenupdates+=1

        if numupdates >= nextprint:
          now=time.time() 
          print "%7s %7.3f %9.5f %9.5f %9.5f %9.5f %7s %4u %9.3e"%(nicetime(float(now-start)),sumsincelenupdates/numsincelenupdates,sumloss/numupdates,sumsinceloss/numsinceupdates,sumoptimalloss/numupdates,sumsinceoptimalloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 
          nextprint=2*nextprint 
          numsinceupdates=0 
          sumsinceloss=0 
          sumsinceoptimalloss=0 
          sumsincelenupdates=0

now=time.time() 
print "%7s %7.3f %9.5f %9.5f %9.5f %9.5f %7s %4u %9.3e"%(nicetime(float(now-start)),sumsincelenupdates/numsincelenupdates,sumloss/numupdates,sumsinceloss/numsinceupdates,sumoptimalloss/numupdates,sumsinceoptimalloss/numsinceupdates,nicecount(numupdates*batchsize),passes,finetuner.eta) 

# (using only first 20 sentences)
#   delta  length    excess     since   optimal     since example pass  learning
#       t   since      loss      last      loss      last counter  num      rate
#  9.263s  10.891   0.03892   0.03892   0.08630   0.08630    1002    0 1.000e-04
# 17.883s   5.478   0.03777   0.03662   0.10967   0.13290    2010    0 1.000e-04
# 49.541s   5.689   0.03112   0.02444   0.10695   0.10422    4007    0 1.000e-04
#  1.927m   6.127   0.02650   0.02188   0.09196   0.07699    8020    0 1.000e-04
#  5.615m   6.605   0.02295   0.01942   0.09114   0.09032     16K    0 1.000e-04
# 14.471m   7.420   0.02031   0.01765   0.08559   0.08001     32K    0 1.000e-04
# 33.275m   7.908   0.01893   0.01755   0.07882   0.07204     64K    0 1.000e-04
#  1.190h   8.461   0.01884   0.01875   0.07621   0.07361    128K    0 1.000e-04
# *** starting fine-tuning ***
#  2.322h   8.618   0.01841   0.01797   0.07471   0.07321    256K    0 1.000e-04
#  4.522h   8.729   0.01854   0.01867   0.07484   0.07498    512K    0 1.000e-04
# ...

# (using everything)
#   delta  length    excess     since   optimal     since example pass  learning
#       t   since      loss      last      loss      last counter  num      rate
# 24.716s  17.373   0.02198   0.02198   0.02775   0.02775    1025    0 1.000e-04
# 40.258s   7.792   0.02243   0.02288   0.06445   0.10158    2038    0 1.000e-04
#  1.621m   8.771   0.02380   0.02518   0.06595   0.06747    4064    0 1.000e-04
#  3.291m   8.872   0.01969   0.01547   0.06158   0.05708    8012    0 1.000e-04
# 10.848m   9.940   0.01780   0.01593   0.05782   0.05409     16K    0 1.000e-04
# 20.405m   9.207   0.01755   0.01729   0.06753   0.07731     32K    0 1.000e-04
# 41.346m   9.069   0.01687   0.01619   0.06849   0.06945     64K    0 1.000e-04
#  1.342h   8.994   0.01775   0.01864   0.07169   0.07489    128K    0 1.000e-04
# *** starting fine-tuning ***
#  2.511h   8.934   0.01770   0.01765   0.07184   0.07199    256K    0 1.000e-04
#  4.772h   8.890   0.01820   0.01869   0.07354   0.07523    512K    0 1.000e-04
# Traceback (most recent call last):
#   File "/home/pmineiro/src/vowpal_wabbit/python/pyvw.py", line 27, in <lambda>

