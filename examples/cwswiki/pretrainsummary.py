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

# python ./pretrainsummary.py | vw --invariant --power_t 0.25 -l 1 --multilabel_oaa 400 -b 29 --multilabel_noise 1e-4 --loss_function logistic --ngram 3 --skips 1 -f pass0summary.vw
# Generating 3-grams for all namespaces.
# Generating 1-skips for all namespaces.
# final_regressor = pass0summary.vw
# Num weight bits = 29
# learning rate = 1
# initial_t = 1
# power_t = 0.25
# using no cache
# Reading datafile =
# num sources = 1
# average  since         example        example  current  current  current
# loss     last          counter         weight    label  predict features
# using precomputed id2cat (400)... 3.05664 seconds.  len(id2cat)  = 690906
# 0.693146 0.693146            1            1.0  47 90 113               380
# 0.597301 0.501456            2            2.0      320  47 90 113      422
# 0.351716 0.106131            4            4.0      360       11      338
# 0.232462 0.113208            8            8.0       73               386
# 0.155914 0.079365           16           16.0  118 288 291       52      332
# 0.111433 0.066952           32           32.0      0 1               710
# 0.075704 0.039975           64           64.0       16      283      464
# 0.059218 0.042732          128          128.0       16               470
# 0.046813 0.034408          256          256.0      317               236
# 0.038487 0.030161          512          512.0       73               386
# 0.033972 0.029457         1024         1024.0  193 376 381               272
# 0.030358 0.026743         2048         2048.0      322               266
# 0.028096 0.025834         4096         4096.0       29               326
# 0.026018 0.023941         8192         8192.0      133        7      308
# 0.024305 0.022591        16384        16384.0  126 156               428
# 0.023048 0.021792        32768        32768.0   99 182       39      410
# best constant loss: 0.0277398
# 
# finished run
# number of examples per pass = 56589
# passes used = 1
# weighted example sum = 56589.000000
# weighted label sum = 0.000000
# average loss = 0.022120
# total feature number = 20869362
