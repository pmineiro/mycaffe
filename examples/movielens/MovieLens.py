import os
import random

#-------------------------------------------------
# iterate over documents
#-------------------------------------------------

# TODO: train/test

def data(path, maxshufbuf):
  shufbuf=[]

  with open (os.path.join (path, 'ratings.dat')) as f:
    for line in f:
      (user,movie,rating,timestamp) = [ int (x) for x in line.split('::') ]

      if maxshufbuf < 1:
        yield (user, movie, rating)
      elif len (shufbuf) < maxshufbuf:
        shufbuf.append ((user, movie, rating))
      else:
        index=random.randrange (maxshufbuf)
	dq = shufbuf[index]
	shufbuf[index] = (user, movie, rating)
	yield dq

  random.shuffle (shufbuf)
  for dq in shufbuf:
    yield dq
