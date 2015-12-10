import bz2
import collections
import exceptions
import hashlib
import math
import numpy as np
import os
import random
import re
import sys
import time

import DocGenerator

random.seed(69)
np.random.seed(8675309)

sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

numtags=int(os.environ['numtags'])
minsentlength=int(os.environ['minsentlength'])
summarylength=int(os.environ['summarylength'])
labelnoise=float(os.environ['labelnoise'])

#-------------------------------------------------
# which ids to use for pretraining
#-------------------------------------------------

def pretrainfilter(idnum):
  return hashlib.md5(idnum).hexdigest()[-1] == "a"

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
try:
  with open('id2cat.precomputed.%u.txt'%numtags,'r') as f:
    sys.stderr.write('using precomputed id2cat (%u)...'%numtags)
    for line in f:
      lc=line.split('\t')
      id2cat[int(lc[0])]=[int(c) for c in lc[1:] if not c.isspace()]
except exceptions.IOError:
  id2catstart=time.time()
  sys.stderr.write('computing id2cat (%u)...'%numtags)
  with bz2.BZ2File('enwiki-20150901.id2cat.txt.bz2', 'rb') as f:
    for line in f:
      idtags=line.split('\t')
      goodtags=[]
      for tag in idtags[1:]:
        if tag in tagnum:
          goodtags.append(tagnum[tag])
      if len(goodtags):
        id2cat[int(idtags[0])]=sorted(goodtags)
  with open('id2cat.precomputed.%u.txt'%numtags,'w') as f:
    for key, value in id2cat.iteritems():
      f.write('%u\t%s\n'%(key,'\t'.join([str(v) for v in value])))

sys.stderr.write(' %g seconds.  len(id2cat)  = %d\n'%(float(time.time()-id2catstart), len(id2cat)))

#-------------------------------------------------
# iterate
#-------------------------------------------------

labelcounts=collections.defaultdict(int)
examples=0
shufbuf=dict()

for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
 if docid in id2cat and pretrainfilter(str(docid)):
   sents = [ s for s in DocGenerator.sentences(paragraphs)
               if len(s) >= minsentlength ]
   if len(sents) < 3*summarylength:
     continue
   z=range(len(sents)) 
   random.shuffle(z) 
   labeltext=','.join([str(l) for l in sorted(id2cat[docid])])
   for l in id2cat[docid]:
     labelcounts[l]+=1
   examples+=1

   for part in range(3):
     sample=[sents[index] for index in 
               sorted(z[part*summarylength:(part+1)*summarylength])]
     words=[w.replace(':','_').replace('|','_') for s in sample for w in s]
     rawtext=' '.join(words)
     rawtext=re.sub(r'[^\x00-\x7F]+','_', rawtext)
     index=random.randint(1,1000)
     if index in shufbuf:
       dq=shufbuf[index]
       print "%s |w:%f %s"%(dq[0],dq[1],dq[2])
     shufbuf[index]=(labeltext,1.0/math.sqrt(len(words)),rawtext)

for index in shufbuf:
  dq=shufbuf[index]
  print "%s |w:%f %s"%(dq[0],dq[1],dq[2])

loss=0
for l in range(numtags):
  freq=float(labelcounts[l])/examples
  freq=(1.0-labelnoise)*freq+labelnoise*(1.0-freq)
  if freq > 0 and freq < 1:
    loss-=freq*math.log(freq)+(1.0-freq)*math.log(1.0-freq)

loss=loss/numtags
sys.stderr.write("best constant loss: %g\n"%loss)

# pmineiro@pmineiro-KGP-M-E-D16-91% make pretrainaggregate.model
# python ./pretrainaggregate.py | vw --invariant --power_t 0.5 -l 1 --multilabel_oaa 400 -b 29 --loss_noise 1e-4 --loss_function noisylogistic --ngram 3 --skips 1 --replay_m 10000
# Generating 3-grams for all namespaces.
# Generating 1-skips for all namespaces.
# experience replay level=m, buffer=10000, replay count=1
# Num weight bits = 29
# learning rate = 1
# initial_t = 1
# power_t = 0.5
# using no cache
# Reading datafile =
# num sources = 1
# average  since         example        example  current  current  current
# loss     last          counter         weight    label  predict features
# using precomputed id2cat (400)... 2.72132 seconds.  len(id2cat)  = 690906
# 0.693146 0.693146            1            1.0       11               338
# 0.689290 0.685434            2            2.0       11               326
# 0.650171 0.611051            4            4.0       96       11      326
# 0.585474 0.520778            8            8.0      320  11 96 320      464
# 0.460752 0.336030           16           16.0      380      206      284
# 0.337697 0.214642           32           32.0      140               326
# 0.207076 0.076455           64           64.0   16 103               224
# 0.136729 0.066382          128          128.0   59 350               302
# 0.088704 0.040679          256          256.0       29               356
# 0.063924 0.039144          512          512.0  28 35 114 212               332
# 0.049700 0.035476         1024         1024.0  134 269               392
# 0.041145 0.032590         2048         2048.0      363               506
# 0.035262 0.029379         4096         4096.0      388               284
# 0.031596 0.027931         8192         8192.0  14 34 101 237 258               284
# 0.028542 0.025489        16384        16384.0      109               446
# 0.026446 0.024349        32768        32768.0      230               314
# 
# best constant loss: 0.0276381
# saving regressor to pass0.vw
# 
# finished run
# number of examples per pass = 61980
# passes used = 1
# weighted example sum = 61980.000000
# weighted label sum = 0.000000
# average loss = 3.918167
# total feature number = 23344338

