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
vw = pyvw.vw('--power_t 0.0 --search 2 --search_task hook --search_rollout none --replay_c %u -b 20 -q np -l %g'%(searnreplaybuf,searnvweta))
task = vw.init_search_task(SentenceSelector.SentenceSelector, task_data)

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s %7s %7s %7s %7s %7s %7s %7s %7s %4s"%("delta","pos","excess","since","simple","since","oracle","since","example","pass") 
print "%7s %7s %7s %7s %7s %7s %7s %7s %7s %4s"%("t","since","loss","last","excess","last","loss","last","counter","num") 

start=time.time()
numsinceupdates=0 
numupdates=0
numsincelenupdates=0 
numsinceposupdates=0 
sumloss=0 
sumsinceloss=0 
sumoptimalloss=0 
sumsimpleloss=0 
sumsinceoptimalloss=0 
sumsincesimpleloss=0 
sumsincelenupdates=0
sumsinceposupdates=0
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
        shufbuf[index]=(parts[:prefixsentences],labels)
        task.learn(Example(dq,numtags))

        loss=(task.saveloss-task.saveoptimalloss)
        optimalloss=task.saveoptimalloss
        simpleloss=(task.savesimpleloss-task.saveoptimalloss)

        sumloss+=loss/batchsize
        sumsinceloss+=loss/batchsize
        sumoptimalloss+=optimalloss/batchsize
        sumsimpleloss+=simpleloss/batchsize
        sumsinceoptimalloss+=optimalloss/batchsize
        sumsincesimpleloss+=simpleloss/batchsize
        sumsincelenupdates+=float(len(dq[0]))
        sumsinceposupdates+=float(task.saveoptimalpos)
    
        numupdates+=float(len(dq[0]))/batchsize
        numsinceupdates+=float(len(dq[0]))/batchsize
        numsincelenupdates+=1
        numsinceposupdates+=1

        if numupdates >= nextprint:
          now=time.time() 
          print "%7s %7.3f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7s %4u"%(nicetime(float(now-start)),sumsinceposupdates/numsinceposupdates,sumloss/numupdates,sumsinceloss/numsinceupdates,sumsimpleloss/numupdates,sumsincesimpleloss/numsinceupdates,sumoptimalloss/numupdates,sumsinceoptimalloss/numsinceupdates,nicecount(numupdates*batchsize),passes) 
          nextprint=2*nextprint 
          numsinceupdates=0 
          sumsinceloss=0 
          sumsinceoptimalloss=0 
          sumsincesimpleloss=0 
          sumsincelenupdates=0
          sumsinceposupdates=0

now=time.time() 
print "%7s %7.3f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7s %4u"%(nicetime(float(now-start)),sumsinceposupdates/numsinceposupdates,sumloss/numupdates,sumsinceloss/numsinceupdates,sumsimpleloss/numupdates,sumsincesimpleloss/numsinceupdates,sumoptimalloss/numupdates,sumsinceoptimalloss/numsinceupdates,nicecount(numupdates*batchsize),passes) 

# (using only first 20 sentences)
# (searneta=1e-8)
# GLOG_minloglevel=5 PYTHONPATH=../../python:../../../vowpal_wabbit/python python ./search_cws.py
# using precomputed id2cat
# creating quadratic features for pairs: np
# experience replay level=c, buffer=1000, replay count=1
# Enabling FTRL based optimization
# Algorithm used: Proximal-FTRL
# ftrl_alpha = 0.005
# ftrl_beta = 0.1
# Num weight bits = 20
# learning rate = 0.0001
# initial_t = 0
# power_t = 0
# using no cache
# Reading datafile =
# num sources = 0
#   delta     pos  excess   since  simple   since  oracle   since example pass
#       t   since    loss    last  excess    last    loss    last counter  num
#  1.409s   5.048 0.03306 0.03306 0.02529 0.02529 0.04929 0.04929     501    0
#  2.093s   1.967 0.05751 0.08195 0.02640 0.02751 0.08630 0.12331    1002    0
#  3.453s   1.925 0.04262 0.02789 0.02437 0.02235 0.10958 0.13260    2015    0
#  6.281s   1.977 0.03390 0.02516 0.02476 0.02516 0.11543 0.12131    4024    0
# 11.895s   2.322 0.02916 0.02440 0.02458 0.02440 0.10916 0.10285    8023    0
# 23.096s   2.033 0.02576 0.02235 0.02346 0.02235 0.11918 0.12918     16K    0
# 45.590s   2.093 0.02503 0.02429 0.02402 0.02457 0.12029 0.12140     32K    0
#  1.503m   2.160 0.02581 0.02660 0.02529 0.02657 0.12589 0.13149     64K    0
#  2.988m   2.170 0.02638 0.02695 0.02609 0.02689 0.12406 0.12223    128K    0
#  5.960m   2.121 0.02601 0.02563 0.02607 0.02604 0.12712 0.13018    256K    0
# 11.923m   2.132 0.02599 0.02597 0.02615 0.02623 0.12651 0.12591    513K    0
# 23.844m   2.149 0.02578 0.02557 0.02607 0.02600 0.12622 0.12592   1026K    0
# *** starting fine-tuning ***
# 48.478m   2.142 0.02586 0.02594 0.02630 0.02652 0.12650 0.12679   2052K    0
#  1.627h   2.150 0.02589 0.02592 0.02646 0.02663 0.12640 0.12630   4104K    0
#  

# GLOG_minloglevel=5 PYTHONPATH=../../python:../../../vowpal_wabbit/python python ./search_cws.py
# using precomputed id2cat
# creating quadratic features for pairs: np
# experience replay level=c, buffer=1000, replay count=1
# Enabling FTRL based optimization
# Algorithm used: Proximal-FTRL
# ftrl_alpha = 0.01
# ftrl_beta = 0.1
# Num weight bits = 20
# learning rate = 0.0001
# initial_t = 0
# power_t = 0
# using no cache
# Reading datafile =
# num sources = 0
#   delta     pos  excess   since  simple   since  oracle   since example pass
#       t   since    loss    last  excess    last    loss    last counter  num
#  1.453s   5.048 0.03650 0.03650 0.02529 0.02529 0.04929 0.04929     501    0
#  2.156s   1.967 0.03201 0.02751 0.02640 0.02751 0.08630 0.12331    1002    0
#  3.555s   1.925 0.02715 0.02235 0.02437 0.02235 0.10958 0.13260    2015    0
#  6.465s   1.977 0.02616 0.02516 0.02476 0.02516 0.11543 0.12131    4024    0
# 12.077s   2.322 0.02503 0.02390 0.02458 0.02440 0.10916 0.10285    8023    0
# 23.489s   2.033 0.02366 0.02229 0.02346 0.02235 0.11918 0.12918     16K    0
# 46.405s   2.093 0.02362 0.02357 0.02402 0.02457 0.12029 0.12140     32K    0
#  1.538m   2.160 0.02518 0.02674 0.02529 0.02657 0.12589 0.13149     64K    0
#  3.047m   2.170 0.02579 0.02641 0.02609 0.02689 0.12406 0.12223    128K    0
#  6.131m   2.121 0.02552 0.02526 0.02607 0.02604 0.12712 0.13018    256K    0
# ...
