#! /usr/bin/env python

import caffe
import math
import h5py
import numpy as np
import os
import sys
import time
from scipy.sparse import csr_matrix
import pdb

np.random.seed(8675309)

alpha=0.9
eta=1.0
etadecay=0.99999
weightdecay=1e-5

# UGH ... so much for DRY

lrs=dict()
lrs['embedding']=1
lrs[('ip1',0)]=1.5
lrs[('ip1',1)]=1.75
lrs[('ip2',0)]=1
lrs[('ip2',1)]=1.25
lrs[('ip3',0)]=0.75
lrs[('ip3',1)]=1

vocabsize=80000
windowsize=2
rawembeddingsize=200
batchsize=1500 

embeddingsize=windowsize*rawembeddingsize
invocabsize=windowsize*(vocabsize+2)
outvocabsize=vocabsize+1

preembed=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')
preembed[:]=np.random.standard_normal(size=(invocabsize,embeddingsize))
embedding=math.sqrt(embeddingsize)*np.linalg.qr(preembed)[0]
del preembed

net = caffe.Net(sys.argv[1])
net.set_mode_gpu()
net.set_phase_train()

momnet = caffe.Net(sys.argv[1])
momnet.set_mode_gpu()
momnet.set_phase_train()

for (layer,momlayer) in zip(net.layers,momnet.layers):
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    blob.data[:]=0.01*np.random.standard_normal(size=blob.data.shape)
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')

momembeddiff=np.zeros(shape=(invocabsize,embeddingsize),dtype='f')

#lrs[('ip3',1)]=0
#with open(sys.argv[3],'r') as priorfile:
#    prior=np.zeros(outvocabsize,dtype='i')
#    pindex=0
#    for line in priorfile:
#        countword=[word for word in line.split(' ')]
#        prior[min(pindex,outvocabsize-1)]+=int(' '.join(countword[:-1]))
#        pindex=pindex+1
#
#prior=np.log(prior.astype('f'))
#
#net.params['ip3'][1].data[:]=prior

f=open(sys.argv[2],'r')

row=[]
col=[]
value=[]
labels=np.zeros(batchsize,dtype='f')
bindex=0
start=time.time()
numsinceupdates=0
numupdates=0
sumloss=0
sumsinceloss=0
sumsigma=0
sumsincesigma=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "%10s\t%7s\t%7s\t%7s\t%7s\t%11s\t%11s"%("delta t","average","since","average","since","example","learning")
print "%10s\t%7s\t%7s\t%7s\t%7s\t%11s\t%11s"%("","loss","last","sigma","last","counter","rate")

for line in f:
    yx=[word for word in line.split(' ')]
    labels[bindex]=int(yx[0])-2

    for word in yx[1:]:
        iv=[subword for subword in word.split(':')]
        row.append(bindex)
        col.append(int(iv[0])-1)
        value.append(float(iv[1]))
    
    bindex=bindex+1

    if bindex >= batchsize:
        try:
          assert(np.min(labels) >= 0)
          assert(np.max(labels) < outvocabsize)
        except:
          pdb.set_trace()

        sd=csr_matrix((value, (row, col)), shape=(batchsize,invocabsize), dtype='f')
        data=sd.dot(embedding).reshape(batchsize,1,1,embeddingsize)
        net.set_input_arrays(data,labels)
        res=net.forward()
        sumloss+=res['loss'][0,0,0,0]
        sumsinceloss+=res['loss'][0,0,0,0]
        meanloss=res['loss'][0,0,0,0]
        meansquareloss=res['losssquared'][0,0,0,0]
        sigma=math.sqrt(meansquareloss-meanloss*meanloss)
        sumsigma+=sigma
        sumsincesigma+=sigma
        net.blobs['lossdetail'].data[:]=np.maximum(np.minimum(1+0.2*np.divide(net.blobs['lossdetail'].data - meanloss, sigma), 3), 0)
        net.backward()
        data_diff=net.blobs['data'].diff.reshape(batchsize,embeddingsize)

        # y = W x
        # y_k = \sum_l W_kl x_l
        # df/dW_ij = \sum_k df/dy_k dy_k/dW_ij
        #          = \sum_k df/dy_k (\sum_l 1_{i=k} 1_{j=l} x_l)
        #          = \sum_k df/dy_k 1_{i=k} x_j
        #          = df/dy_i x_j
        # df/dW    = (df/dy)*x'

        sdtransdiff=sd.transpose().tocsr().dot(data_diff)

        momembeddiff=alpha*momembeddiff+lrs['embedding']*eta*sdtransdiff
        embedding=embedding-momembeddiff
        embedding=(1-lrs['embedding']*weightdecay*eta)*embedding

        for (name,layer,momlayer) in zip(net._layer_names,net.layers,momnet.layers):
          blobnum=0
          for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
            myeta=lrs[(name,blobnum)]*eta
            momblob.data[:]=alpha*momblob.data[:]+myeta*blob.diff
            blob.data[:]-=momblob.data[:]
            blob.data[:]=(1-weightdecay*myeta)*blob.data[:]
            blobnum=blobnum+1

        eta=eta*etadecay
        value=[]
        row=[]
        col=[]
        labels[:]=0
        bindex=0
        numupdates=numupdates+1
        numsinceupdates=numsinceupdates+1
        if numupdates >= nextprint:
            net.save(sys.argv[4]+"."+str(numupdates))
            h5f=h5py.File(sys.argv[4]+"_e."+str(numupdates))
            h5f.create_dataset('embedding',data=embedding)
            h5f.close()
            now=time.time()
            print "%10.3f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,sumsigma/numupdates,sumsincesigma/numsinceupdates,numupdates*batchsize,eta)
            nextprint=2*nextprint
            numsinceupdates=0
            sumsinceloss=0
            sumsincesigma=0


now=time.time()
print "%10.3f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,sumsigma/numupdates,sumsincesigma/numsinceupdates,numupdates*batchsize,eta)
net.save(sys.argv[4])
h5f=h5py.File(sys.argv[4]+"_e")
h5f.create_dataset('embedding',data=embedding)
h5f.close()

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding')

# GLOG_minloglevel=5 PYTHONPATH=../../python python makefofexrmsparsemodel.py fofe_xrm_sparse_small_unigram_train <(head -n `cat numlinesfofengram9 | perl -lane 'print int(0.9*$F[0])'` fofengram9.txt) histo9 fofexrmsparsemodel9
#    delta t      average   since average   since     example        learning
#                    loss    last   sigma    last     counter            rate
#      2.437      11.2876 11.2876  0.0107  0.0107        1500         0.99999
#      4.639      11.2827 11.2779  0.0157  0.0207        3000         0.99998
#      8.373      11.2663 11.2498  0.0359  0.0562        6000         0.99996
#     14.982      11.2084 11.1505  0.1088  0.1816       12000         0.99992
#     27.307      10.9517 10.6950  0.4317  0.7546       24000         0.99984
#     55.744       9.9145  8.8774  1.8086  3.1856       48000         0.99968
#    102.500       8.9218  7.9292  2.6706  3.5326       96000         0.99936
#    198.541       8.0366  7.1513  3.1023  3.5340      192000        0.998721
#    388.289       7.4295  6.8224  3.2910  3.4798      384000        0.997443
#    767.180       6.8688  6.3081  3.4912  3.6914      768000        0.994893
#   1588.714       6.3724  5.8759  3.6531  3.8150     1536000        0.989812
#   3983.452       5.9996  5.6268  3.7345  3.8160     3072000        0.979728
#   9266.551       5.7090  5.4185  3.7602  3.7858     6144000        0.959867
#  20313.223       5.4711  5.2332  3.7465  3.7328    12288000        0.921345
#  43295.445       5.2676  5.0642  3.7090  3.6715    24576000        0.848877
#  90768.050       5.0851  4.9026  3.6571  3.6051    49152000        0.720592
# 187623.138       4.9126  4.7401  3.5969  3.5367    98304000        0.519253
