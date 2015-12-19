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
# using precomputed id2cat (400)... 2.6669 seconds.  len(id2cat)  = 690906
# 0.693146 0.693146            1            1.0      217               452
# 0.375227 0.057309            2            2.0  28 48 212      217     1070
# 0.228600 0.081974            4            4.0  118 288 291  28 48 165 212 240     1010
# 0.180287 0.131973            8            8.0      140  104 118 125 288 291      998
# 0.131238 0.082189           16           16.0   20 392       11      884
# 0.095919 0.060601           32           32.0      274   97 385      398
# 0.077209 0.058498           64           64.0     9 67  2 3 8 191 229     1004
# 0.064547 0.051886          128          128.0   32 380      224      998
# 0.055332 0.046116          256          256.0       11               968
# 0.048544 0.041755          512          512.0       53    1 257      758
# 0.044696 0.040848         1024         1024.0   16 349  103 114 170 240      938
# 0.039646 0.034595         2048         2048.0  23 333 386   23 393     1064
# 0.036094 0.032543         4096         4096.0  135 367              1028
# 0.032225 0.028357         8192         8192.0       96               926
# 0.029503 0.026780        16384        16384.0      158      169      938
# ...
