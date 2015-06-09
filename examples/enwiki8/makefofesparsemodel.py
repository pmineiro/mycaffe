#! /usr/bin/env python

import caffe
import math
import h5py
import numpy as np
import sys
import time
from scipy.sparse import csr_matrix

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
nextprint=1

print "%10s\t%10s\t%10s\t%11s\t%11s"%("delta t","average","since","example","learning")
print "%10s\t%10s\t%10s\t%11s\t%11s"%("","loss","last","counter","rate")

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
        sd=csr_matrix((value, (row, col)), shape=(batchsize,invocabsize), dtype='f')
        data=sd.dot(embedding).reshape(batchsize,1,1,embeddingsize)
        net.set_input_arrays(data,labels)
        res=net.forward()
        sumloss+=res['loss'][0,0,0,0]
        sumsinceloss+=res['loss'][0,0,0,0]
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
            net.save(sys.argv[3]+"."+str(numupdates))
            h5f=h5py.File(sys.argv[3]+"_e."+str(numupdates))
            h5f.create_dataset('embedding',data=embedding)
            h5f.close()
            now=time.time()
            print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
            nextprint=2*nextprint
            numsinceupdates=0
            sumsinceloss=0


now=time.time()
print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
net.save(sys.argv[3])
h5f=h5py.File(sys.argv[3]+"_e")
h5f.create_dataset('embedding',data=embedding)
h5f.close()

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');

# GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparsemodel.py fofe_sparse_small_unigram_train <(head -n `wc -l fofengram9.txt | perl -lane 'print int(0.9*$F[0])'` fofengram9.txt) fofesparsemodel9
#    delta t         average           since          example        learning
#                       loss            last          counter            rate
#      3.324         11.2893         11.2893             1500         0.99999
#      6.609         11.2832         11.2772             3000         0.99998
#     12.333         11.2623         11.2413             6000         0.99996
#     22.722         11.1827         11.1032            12000         0.99992
#     42.765         10.7750         10.3673            24000         0.99984
#     81.470         10.0174          9.2598            48000         0.99968
#    158.956          9.0810          8.1445            96000         0.99936
#    307.732          8.1580          7.2351           192000        0.998721
#    598.225          7.4279          6.6979           384000        0.997443
#   1180.870          6.8014          6.1749           768000        0.994893
#   2646.052          6.3289          5.8563          1536000        0.989812
#   6218.354          5.9774          5.6260          3072000        0.979728
#  14266.073          5.7044          5.4313          6144000        0.959867
#  31046.187          5.4776          5.2508         12288000        0.921345

