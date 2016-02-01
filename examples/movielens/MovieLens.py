import os
import random

#-------------------------------------------------
# iterate over documents
#-------------------------------------------------

# TODO: train/test

def data(path, maxshufbuf, istest):
  shufbuf=[]
  seenusers=dict()

  with open (os.path.join (path, 'ratings.dat')) as f:
    for line in f:
      (movie, user, rating, timestamp) = [ int (x) for x in line.split('::') ]

      olduser = user in seenusers
      seenusers[user] = 1

      if istest == olduser:
          continue

      seenusers[user]=1

      if maxshufbuf < 1:
        yield (movie, user, rating)
      elif len (shufbuf) < maxshufbuf:
        shufbuf.append ((movie, user, rating))
      else:
        index=random.randrange (maxshufbuf)
	dq = shufbuf[index]
	shufbuf[index] = (movie, user, rating)
	yield dq

  random.shuffle (shufbuf)
  for dq in shufbuf:
    yield dq
