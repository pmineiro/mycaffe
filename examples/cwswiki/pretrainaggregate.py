import bz2
import collections
import exceptions
import hashlib
import math
import numpy as np
import os
import random
import re
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
   if len(sents) < summarylength:
     continue
   sample=sents
   words=[w.replace(':','_').replace('|','_') for s in sample for w in s]
   rawtext=' '.join(words)
   rawtext=re.sub(r'[^\x00-\x7F]+','_', rawtext)
   labeltext=','.join([str(l) for l in id2cat[docid]])
   for l in id2cat[docid]:
     labelcounts[l]+=1
   examples+=1

   index=random.randint(1,aggregatechars)
   if index in shufbuf:
     dq=shufbuf[index]
     print "%s |w:%f %s"%(dq[0],dq[1],dq[2])
   truncwords=len(string.split(rawtext[:aggregatechars]))
   shufbuf[index]=(labeltext,1.0/math.sqrt(truncwords),rawtext[:aggregatechars])

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

# python ./pretrainaggregate.py | vw --invariant --power_t 0.25 -l 1 --multilabel_oaa 400 -b 29 --multilabel_noise 1e-4 --loss_function logistic --ngram 3 --skips 1 -f pass0aggregate.vw
# Generating 3-grams for all namespaces.
# Generating 1-skips for all namespaces.
# final_regressor = pass0aggregate.vw
# Num weight bits = 29
# learning rate = 1
# initial_t = 1
# power_t = 0.25
# using no cache
# Reading datafile =
# num sources = 1
# average  since         example        example  current  current  current
# loss     last          counter         weight    label  predict features
# using precomputed id2cat (400)... 2.80313 seconds.  len(id2cat)  = 690906
# 0.693146 0.693146            1            1.0      217               452
# 0.417598 0.142049            2            2.0  28 48 212      217     1064
# 0.326808 0.236017            4            4.0  125 326  28 48 212 213      986
# 0.232103 0.137398            8            8.0      140      222     1022
# 0.158357 0.084611           16           16.0       38  39 43 92 143 257 283 336       728
# 0.109459 0.060561           32           32.0      166  131 192      956
# 0.084635 0.059811           64           64.0  28 35 114 212               986
# 0.073600 0.062566          128          128.0       11  0 1 189 321      968
# 0.059715 0.045829          256          256.0       97              1034
# 0.051212 0.042710          512          512.0      279       14      386
# 0.046525 0.041838         1024         1024.0    84 92    56 84      614
# 0.041906 0.037287         2048         2048.0  0 1 37 194  0 1 235     1064
# 0.037525 0.033144         4096         4096.0      334              1070
# 0.034068 0.030612         8192         8192.0      128               440
# 0.031075 0.028083        16384        16384.0       18      265      572
# best constant loss: 0.0268735
# 
# finished run
# number of examples per pass = 30389
# passes used = 1
# weighted example sum = 30389.000000
# weighted label sum = 0.000000
# average loss = 0.028833
# total feature number = 25629400

