import bz2
import re
import string

#-------------------------------------------------
# iterate over documents
#-------------------------------------------------

def docs(filename):
  docid=None
  startdocregex=re.compile('<doc id="(\d+)"')

  with bz2.BZ2File(filename, 'rb') as f:
    for line in f:
      if docid is not None:
        if line[:6] == "</doc>":
          yield int(docid), paragraphs
          docid=None
        elif line.isspace():
          paragraphs.append(' '.join(curpara))
          curpara=[]
        else:
          curpara.append(line.rstrip('\n'))

      if docid is None:
        m=startdocregex.match (line)
        if m is not None:
          docid=m.group(1)
          paragraphs=[]
          curpara=[]

#------------------------------------------------
# extract and tokenize sentences from paragraph, 
#   and reject short paragraphs (i.e., headers)
#-------------------------------------------------

sentenceregex=re.compile('(?<!i\.e|e\.g|c\.f|.cf)\.\s(?!,)')
def sentences(paragraphs):
  for para in paragraphs:
    sents=sentenceregex.split(para)

    if len(sents) < 2:
      continue

    for s in sents:
      yield [tt for tt in [t.strip(string.punctuation) for t in s.split()] 
                if not tt.isspace() ]
