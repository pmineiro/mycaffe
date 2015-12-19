import caffe
import gzip
import numpy as np
import struct
import os
import sys
import time

from Pretty import nicetime, nicecount

batchsize=64
eta=0
weightdecay=0

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 

with gzip.GzipFile('t10k-images-idx3-ubyte.gz') as f:
  magic, nimages, nrows, ncolumns = struct.unpack("!LLLL", f.read(16))
  assert (magic == 2051)

  traindata = np.zeros((nimages,1,nrows,ncolumns),dtype='f')

  for n in range(nimages):
    traindata[n,0,:,:] = np.array(struct.unpack("%uB"%(nrows*ncolumns), f.read(nrows*ncolumns)),dtype='f').reshape(28,28)

  traindata /= 256.0

with gzip.GzipFile('t10k-labels-idx1-ubyte.gz') as f:
  magic, nlabels = struct.unpack("!LL", f.read(8))
  assert (magic == 2049)
  assert (nlabels == nimages)

  trainlabels = np.zeros((nlabels,1,1,1),dtype='f')

  trainlabels[:,0,0,0] = np.array(struct.unpack("%uB"%nlabels, f.read(nlabels)),dtype='f')

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train.prototxt')
modelfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),sys.argv[1])

caffe.set_mode_cpu()
net = caffe.Net(protofilename, modelfilename, caffe.TEST)

#-------------------------------------------------
# iterate
#-------------------------------------------------

print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta t","average","since","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("","loss","last","counter","num","rate") 

start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
nextprint=1 

# NB: if i don't make these permanent arrays to hold the input,
# memory leaks until the script crashes
subtrain=np.zeros((batchsize,1,28,28),dtype='f')
sublabel=np.zeros((batchsize,1,1,1),dtype='f')

passes=0
for pos in range(0, nimages, batchsize):
  if pos + batchsize > nimages:
    continue

  subtrain[:,:,:,:]=traindata[pos:pos+batchsize,:,:,:]
  sublabel[:,:,:,:]=trainlabels[pos:pos+batchsize,:,:,:]
  net.set_input_arrays (subtrain, sublabel)
  res=net.forward()
  sumloss+=res['acc']
  sumsinceloss+=res['acc']
  numupdates+=1
  numsinceupdates+=1

  if numupdates >= nextprint:
    now=time.time() 
    print "%7s\t%7.4f\t%7.4f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 
    nextprint=2*nextprint 
    numsinceupdates=0 
    sumsinceloss=0 

now=time.time() 
print "%7s\t%7.4f\t%7.4f\t%7s\t%4u\t%9.3e"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicecount(numupdates*batchsize),passes,eta) 
