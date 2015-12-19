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
    z=range(len(sents)) 
    random.shuffle(z) 
    sample=[sents[index] for index in sorted(z[0:summarylength])]
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
