#! /usr/bin/env python

from enum import Enum
import caffe
import gzip
import math
import h5py
import numpy as np
import os
import sys
import time
from scipy.sparse import csr_matrix

import pdb

np.random.seed(8675309)
np.seterr(all='raise')
np.seterr(under='ignore')

eta=0.2
weightdecay=1e-5

lrs=dict()
lrs['embedding']=1
lrs[('ip1',0)]=1.5
lrs[('ip1',1)]=1.75
lrs[('ip2',0)]=1
lrs[('ip2',1)]=1.25
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

weightdecays=dict()
weightdecays['embedding']=1
weightdecays[('ip1',0)]=1
weightdecays[('ip1',1)]=1
weightdecays[('ip2',0)]=1
weightdecays[('ip2',1)]=1
weightdecays[('ip3',0)]=1
weightdecays[('ip3',1)]=1

vocabsize=80000
windowsize=2
rawembeddingsize=200
batchsize=1000
switchfac=1.05

embeddingsize=windowsize*rawembeddingsize
invocabsize=windowsize*(vocabsize+2)
outvocabsize=vocabsize+1

preembed=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')
preembed[:]=np.random.standard_normal(size=(invocabsize,embeddingsize))
embedding=math.sqrt(embeddingsize)*np.linalg.qr(preembed)[0]
oldembedding=embedding
del preembed

net = caffe.Net(sys.argv[1])
net.set_mode_gpu()
net.set_phase_train()

oldnet = caffe.Net(sys.argv[1])
oldnet.set_mode_gpu()
oldnet.set_phase_train()

batchnet = caffe.Net(sys.argv[1])
batchnet.set_mode_gpu()
batchnet.set_phase_train()

batchdata_diff=np.zeros(embeddingsize,dtype='f')

for (layer,oldlayer,batchlayer) in zip(net.layers,oldnet.layers,batchnet.layers):
  for (blob,oldblob,batchblob) in zip(layer.blobs,oldlayer.blobs,batchlayer.blobs):
    preblob=np.zeros(shape=(blob.data.shape[2],blob.data.shape[3]),dtype='f')
    preblob[:]=np.random.standard_normal(size=preblob.shape)
    blob.data[:]=np.linalg.qr(preblob)[0]
    oldblob.data[:]=blob.data
    batchblob.data[:]=0

#lrs[('ip3',1)]=0.0001
#with open(sys.argv[4],'r') as priorfile:
#    prior=np.zeros(outvocabsize,dtype='d')
#    pindex=0
#    for line in priorfile:
#        countword=[word for word in line.split(' ')]
#        prior[pindex]=math.log(float(' '.join(countword[:-1])))
#        pindex=pindex+1
#        if pindex >= outvocabsize:
#            break
#
#net.params['ip3'][1].data[:]=prior
#oldnet.params['ip3'][1].data[:]=prior

Phase=Enum('Phase','batch online sgd')

row=[]
col=[]
value=[]
labels=np.zeros(batchsize,dtype='f')
bindex=0
curbnum=0
start=time.time()
numbatch=0
numsinceupdates=0
numupdates=0
sumloss=0
sumsinceloss=0
nextprint=1
nextswitch=1
phase=Phase.sgd
phasesize=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "%9s   %8s   %8s   %9s   %7s   %7s"%("delta","average","since","example","phase","phase")
print "%9s   %8s   %8s   %9s   %7s   %7s"%("t","loss","last","counter","size","")

maxlinenum=int(sys.argv[2])

with gzip.open(sys.argv[3],'rb') as f:
  oldbnum=0
  oldlinenum=0
  oldpos=f.tell()
  curlinenum=0
  while curlinenum < maxlinenum:
    line=f.readline()
    curlinenum=curlinenum+1

    yx=[word for word in line.split(' ')]
    labels[bindex]=int(yx[0])-2

    for word in yx[1:]:
        iv=[subword for subword in word.split(':')]
        row.append(bindex)
        col.append(int(iv[0])-1)
        value.append(float(iv[1]))

    bindex=bindex+1

    if bindex >= batchsize:
        curbnum=curbnum+1

        sd=csr_matrix((value, (row, col)), shape=(batchsize,invocabsize), dtype='f')
        olddata=sd.dot(oldembedding).reshape(batchsize,1,1,embeddingsize)
        oldnet.set_input_arrays(olddata,labels)
        res=oldnet.forward()
        oldnet.backward()
        olddata_diff=oldnet.blobs['data'].diff.reshape(batchsize,embeddingsize)

        if phase == Phase.sgd:
            sumloss+=res['loss'][0,0,0,0]
            sumsinceloss+=res['loss'][0,0,0,0]
            try:
              #assert(res['loss'][0,0,0,0] < 12)
              assert(not math.isnan(sumloss))
              assert(not math.isnan(sumsinceloss))
            except:
              pdb.set_trace()

            oldsdtransdiff=sd.transpose().tocsr().dot(olddata_diff)

            myeta=lrs['embedding']*eta
            oldembedding-=myeta*oldsdtransdiff
            myweightdecay=weightdecays['embedding']*weightdecay
            oldembedding-=(myeta*myweightdecay)*oldembedding

            for (name,oldlayer) in zip(oldnet._layer_names,oldnet.layers):
              blobnum=0
              for oldblob in oldlayer.blobs:
                myeta=lrs[(name,blobnum)]*eta
                myweightdecay=weightdecays[(name,blobnum)]*weightdecay
                oldblob.data[:]-=myeta*oldblob.diff
                oldblob.data[:]-=(myeta*myweightdecay)*oldblob.data
                blobnum=blobnum+1

            numupdates=numupdates+1
            numsinceupdates=numsinceupdates+1

            if curbnum >= nextswitch:
                embedding=oldembedding
                for (layer,oldlayer) in zip(net.layers,oldnet.layers):
                  for (blob,oldblob) in zip(layer.blobs,oldlayer.blobs):
                    blob.data[:]=oldblob.data

                phase=Phase.batch
                nextswitch+=phasesize
                numbatch=0
                oldpos=f.tell()
                oldlinenum=curlinenum
                oldbnum=curbnum
        elif phase == Phase.batch:
            numbatch=numbatch+1
            # NB: batch-ing layer diff rather than parameter diff ...
            # TODO: mean or sum (?)
            batchdata_diff+=np.sum(olddata_diff,axis=0,dtype='d')

            for (oldlayer,batchlayer) in zip(oldnet.layers,batchnet.layers):
                for (oldblob,batchblob) in zip(oldlayer.blobs,batchlayer.blobs):
                    batchblob.data[:]+=oldblob.diff

            if curbnum >= nextswitch or curlinenum + batchsize >= maxlinenum:
                batchdata_diff*=1.0/numbatch
                myeta=lrs['embedding']*eta
                for (name,layer,batchlayer) in zip(net._layer_names,
                                                   net.layers,
                                                   batchnet.layers):
                    blobnum=0
                    for (blob,batchblob) in zip(layer.blobs,batchlayer.blobs):
                        batchblob.data[:]*=1.0/numbatch
                        blobnum=blobnum+1

                phase=Phase.online
                f.seek(oldpos,0)
                curlinenum=oldlinenum
                curbnum=oldbnum
        else:
            data=sd.dot(embedding).reshape(batchsize,1,1,embeddingsize)
            net.set_input_arrays(data,labels)
            res=net.forward()
            sumloss+=res['loss'][0,0,0,0]
            sumsinceloss+=res['loss'][0,0,0,0]
            try:
              #assert(res['loss'][0,0,0,0] < 12)
              assert(not math.isnan(sumloss))
              assert(not math.isnan(sumsinceloss))
            except:
              pdb.set_trace()
            net.backward()
            # cheesy stabilize embedding gradient
            # compute correlation on data_diff and then project via data
            data_diff=net.blobs['data'].diff.reshape(batchsize,embeddingsize)
            controlvar=olddata_diff-batchdata_diff
            # NB: np.linalg.norm does not take dtype argument ...
            cordenom=math.sqrt(np.sum(np.square(data_diff.flat),dtype='d'))*math.sqrt(np.sum(np.square(controlvar),dtype='d'))
            if cordenom > 0:
              cor=np.sum(np.multiply(data_diff,controlvar),dtype='d')/cordenom
            else:
              cor=0

            sdtranscsr=sd.transpose().tocsr()
            sdtransdiff=sdtranscsr.dot(data_diff)
            controlsdtransdiff=sdtranscsr.dot(controlvar)

            myeta=lrs['embedding']*eta
            embedding-=myeta*sdtransdiff
            myweightdecay=weightdecays['embedding']*weightdecay
            embedding-=(myeta*myweightdecay)*embedding
            if cor > 0:
              embedding+=(myeta*cor)*controlsdtransdiff

            for (name,layer,oldlayer,batchlayer) in zip(net._layer_names,
                                                        net.layers,
                                                        oldnet.layers,
                                                        batchnet.layers):
                blobnum=0
                for (blob,oldblob,batchblob) in zip(layer.blobs,
                                                    oldlayer.blobs,
                                                    batchlayer.blobs):
                  controlvar=oldblob.diff[:]-batchblob.data[:]
                  # NB: np.linalg.norm does not take dtype argument ...
                  cordenom=math.sqrt(np.sum(np.square(blob.diff.flat),dtype='d'))*math.sqrt(np.sum(np.square(controlvar),dtype='d'))
                  if cordenom > 0:
                    cor=np.sum(np.multiply(blob.diff[:],controlvar),dtype='d')/cordenom
                  else:
                    cor=0
                  myeta=lrs[(name,blobnum)]*eta
                  blob.data[:]-=myeta*blob.diff
                  myweightdecay=weightdecays[(name,blobnum)]*weightdecay
                  blob.data[:]-=(myeta*myweightdecay)*blob.data
                  if cor > 0:
                    blob.data[:]+=(myeta*cor)*controlvar
                  blobnum=blobnum+1

            numupdates=numupdates+1
            numsinceupdates=numsinceupdates+1

            if curbnum >= nextswitch or curlinenum + batchsize >= maxlinenum:
                batchdata_diff[:]=0
                oldembedding[:]=embedding
                for (layer,oldlayer,batchlayer) in zip(net.layers,
                                                       oldnet.layers,
                                                       batchnet.layers):
                    for (blob,oldblob,batchblob) in zip(layer.blobs,
                                                        oldlayer.blobs,
                                                        batchlayer.blobs):
                        batchblob.data[:]=0
                        oldblob.data[:]=blob.data

                phase=Phase.batch
                phasesize=switchfac*phasesize
                nextswitch+=math.floor(phasesize)
                numbatch=0
                oldpos=f.tell()
                oldlinenum=curlinenum
                oldbnum=curbnum

        value=[]
        row=[]
        col=[]
        labels[:]=0
        bindex=0

        if numupdates >= nextprint:
            net.save(sys.argv[5]+"."+str(numupdates))
            h5f=h5py.File(sys.argv[5]+"_e."+str(numupdates))
            h5f.create_dataset('embedding',data=embedding)
            h5f.close()
            now=time.time()
            print "%9.2f   %8.4f   %8.4f   %9u   %7u   %7s"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,curbnum*batchsize,phasesize,str(phase)[6:])
            nextprint*=2
            numsinceupdates=0
            sumsinceloss=0

now=time.time()
print "%9.2f   %8.4f   %8.4f   %9u   %7u   %7s"%(now-start,sumloss/numupdates,sumsinceloss/(numsinceupdates+1e-16),curbnum*batchsize,phasesize,str(phase)[6:])

net.save(sys.argv[5])
h5f=h5py.File(sys.argv[5]+"_e")
h5f.create_dataset('embedding',data=embedding)
h5f.close()

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding')

# GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparsesvrgmodel.py fofe_sparse_small_unigram_train <(head -n `wc -l fofengram9.txt | perl -lane 'print int(0.9*$F[0])'` fofengram9.txt) fofesparsesvrgmodel9
#   delta t         average           since          example        learning
#                      loss            last          counter            rate
# (flass)