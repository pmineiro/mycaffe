#! /usr/bin/env python

import os
import numpy as np
import pyhash
import random
import struct
import string
import sys
import time

import DocGenerator
import pdb

bits=18

def hashit(ngram):
  hv = pyhash.murmur1_32()(' '.join(ngram))
  return (1 + ((hv >> 1) % (1 << bits)), 1 - 2 * (hv % 2))

def ngrams(tokens, maxn):
  for n in range(0,maxn):
    for t in zip(*[tokens[i:] for i in range(1+n)]):
      yield t

random.seed (69)
np.random.seed (8675309)

sys.stdout = os.fdopen (sys.stdout.fileno (), 'w', 0) 
sys.stderr = os.fdopen (sys.stderr.fileno (), 'w', 0) 

start = time.time ()

skip = 0
keep = 0
exnum = 0

with open('docid2label','wb') as f, open('trainxi','wb') as trainxi, \
     open('trainxj','wb') as trainxj, open('trainxs','wb') as trainxs, \
     open('testxi','wb') as testxi, open('testxj','wb') as testxj, \
     open('testxs','wb') as testxs:
  for docid, paragraphs in DocGenerator.docs ('text/AA/wiki_00.shuf.bz2'):
    goodparagraphs = [n for n in range (len (paragraphs))
                        if len (paragraphs[n].split ()) > 20]
  
    if len (goodparagraphs) < 4:
      skip += 1
      continue
  
    keep += 1

    f.write ('%s\t%u\n'% (docid, keep))
  
    random.shuffle (goodparagraphs)
  
    for n in goodparagraphs[0:3]:
      tokens = [ t.strip (string.punctuation) for t in paragraphs[n].split () ]
      exnum += 1
      for ngram in ngrams (tokens, 3):
        (feature, value) = hashit (ngram)
        trainxi.write (struct.pack ('d', exnum))
        trainxj.write (struct.pack ('d', feature))
        trainxs.write (struct.pack ('d', value))

    for n in goodparagraphs[3:4]:
      tokens = [ t.strip (string.punctuation) for t in paragraphs[n].split () ]
      for ngram in ngrams (tokens, 3):
        (feature, value) = hashit (ngram)
        testxi.write (struct.pack ('d', keep))
        testxj.write (struct.pack ('d', feature))
        testxs.write (struct.pack ('d', value))

    if keep % 1000 == 1:
      print "%g\t%u\t%u\t%g"%(float (time.time () - start),
                              keep,
                              skip,
                              float (keep) / (keep + skip))

    if keep > 2 * 1000 * 1000:
      break
