import bz2
import collections
import exceptions
import hashlib
import math
import numpy as np
import os
import pyvw
import random
import string
import sys
import time

import DocGenerator

random.seed(69)
np.random.seed(8675309)

sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

numtags=int(os.environ['numtags'])
minsentlength=int(os.environ['minsentlength'])
summarylength=int(os.environ['summarylength'])
aggregatechars=int(os.environ['aggregatechars'])
labelnoise=float(os.environ['labelnoise'])

#-------------------------------------------------
# which ids to use for preval
#-------------------------------------------------

def prevalfilter(idnum):
  return hashlib.md5(idnum).hexdigest()[-1] == "b"

#-------------------------------------------------
# read label (tag) information
#-------------------------------------------------

tagnum=dict()
with open('taghisto', 'r') as f:
  for line in f:
    tc=line.split('\t')
    if "death" in tc[0] or "birth" in tc[0] or "Living people" in tc[0]:
      continue
    tagnum[tc[0]]=len(tagnum)
    if len(tagnum) >= numtags:
      break

#-------------------------------------------------
# read id2cat information
#-------------------------------------------------

id2cat=dict()
id2catstart=time.time()
with open('id2cat.precomputed.%u.txt'%numtags,'r') as f:
  sys.stderr.write('using precomputed id2cat (%u)...'%numtags)
  for line in f:
    lc=line.split('\t')
    id2cat[int(lc[0])]=[int(c) for c in lc[1:] if not c.isspace()]

sys.stderr.write(' %g seconds.  len(id2cat)  = %d\n'%(float(time.time()-id2catstart), len(id2cat)))

vw=pyvw.vw('-t -i %s'%sys.argv[1])

#-------------------------------------------------
# iterate
#-------------------------------------------------

labelcounts=collections.defaultdict(int)
labels=0
examples=0
examplessincelast=0
zeroonlyloss=0
sumlogloss=0
sumloglosssincelast=0
nextprint=1

random.seed(69)
for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
  if docid in id2cat and prevalfilter(str(docid)):
    sents = [ s for s in DocGenerator.sentences(paragraphs)
                if len(s) >= minsentlength ]
    if len(sents) < summarylength:
      continue
#     z=range(len(sents)) 
#     random.shuffle(z) 
#     sample=[sents[index] for index in 
#               sorted(z[passn*summarylength:(passn+1)*summarylength])]
    sample=sents
    words=[w.replace(':','_').replace('|','_') for s in sample for w in s]
    rawtext=' '.join(words)
    labeltext=','.join([str(l) for l in id2cat[docid]])
    zeroonlyloss+=1
    for l in id2cat[docid]:
      labelcounts[l]+=1
      labels+=1
      zeroonlyloss+=1 if l > 0 else -1
    examples+=1
    examplessincelast+=1
    truncwords=len(string.split(rawtext[:aggregatechars]))
    with vw.example("%s |w:%f %s"%(labeltext,1.0/math.sqrt(truncwords),rawtext[:aggregatechars])) as ex:
      ex.set_test_only(True)
      ex.learn()
      logloss=np.sum([math.log(1+math.exp(-ex.get_multilabel_raw_prediction(l))) for l in id2cat[docid]])+np.sum([math.log(1+math.exp(ex.get_multilabel_raw_prediction(l))) for l in range(numtags) if l not in id2cat[docid]])
      sumlogloss+=logloss
      sumloglosssincelast+=logloss

    if examples > nextprint:
      print "%g\t%g"%(sumlogloss/(examples*numtags),sumloglosssincelast/(examplessincelast*numtags))
      nextprint=2*nextprint+1
      sumloglosssincelast=0
      examplessincelast=0

sys.stderr.write('labels=%u examples=%u labels/example=%g nonzeroloss/example=%g\n'%(labels,examples,float(labels)/examples,float(zeroonlyloss)/examples))

loss=0
for l in range(numtags):
  freq=float(labelcounts[l])/examples
  freq=(1.0-labelnoise)*freq+labelnoise*(1.0-freq)
  if freq > 0 and freq < 1:
    loss-=freq*math.log(freq)+(1.0-freq)*math.log(1.0-freq)

loss=loss/numtags
sys.stderr.write("best constant loss: %g\n"%loss)

# PYTHONPATH=../../../vowpal_wabbit/python python ./prevalaggregate.py pass0aggregate.vw
# using precomputed id2cat (400)... 2.80215 seconds.  len(id2cat)  = 690906
# Generating 3-grams for all namespaces.
# Generating 1-skips for all namespaces.
# only testing
# Num weight bits = 29
# learning rate = 0.5
# initial_t = 0
# power_t = 0.5
# using no cache
# Reading datafile =
# num sources = 0
# 0.0489779       0.0489779
# 0.0301811       0.0113843
# 0.0241243       0.0180675
# 0.0233809       0.0226375
# 0.0289429       0.0345048
# 0.0271956       0.0254483
# 0.0211159       0.0150361
# 0.0228576       0.0245994
# 0.0243529       0.0258482
# 0.0242974       0.0242418
# 0.0249116       0.0255258
# 0.0247603       0.024609
# 0.0252925       0.0258248
# 0.0252128       0.025133
# labels=53278 examples=30787 labels/example=1.73054 nonzeroloss/example=2.62101
# best constant loss: 0.0267863
