import bz2
import collections
import operator
import os
import string
import sys

import pdb

import DocGenerator

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

tokenfreq=collections.defaultdict(int)

for docid, paragraphs in DocGenerator.docs('text/AA/wiki_00.shuf.bz2'):
  goodparagraphs = [n for n in range (len (paragraphs))
                      if len (paragraphs[n].split ()) > 20]

  for n in goodparagraphs:
    for t in [ t.strip (string.punctuation) for t in paragraphs[n].split() ]:
      tokenfreq[t] += 1

z=sorted(tokenfreq.items(),key=operator.itemgetter(1),reverse=True)

for key,value in z:
  print "%s\t%u"%(key,value)
