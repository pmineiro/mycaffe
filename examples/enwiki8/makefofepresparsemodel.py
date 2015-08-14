#! /usr/bin/env python

import caffe
import gzip
import math
import h5py
import numpy as np
import os
import sys
import time
from scipy.sparse import csr_matrix, dia_matrix
from numpy.linalg import lstsq, solve
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

#lrs[('ip3',1)]=0.001
#with open(sys.argv[4],'r') as priorfile:
#    prior=np.zeros(outvocabsize,dtype='i')
#    pindex=0
#    for line in priorfile:
#        countword=[word for word in line.split(' ')]
#        prior[min(pindex,outvocabsize-1)]+=int(' '.join(countword[:-1]))
#        pindex=pindex+1
#prior=np.log(prior.astype('f'))
#net.params['ip3'][1].data[:]=prior

maxlines=int(sys.argv[2])
f=gzip.open(sys.argv[3],'rb')

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

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "%10s\t%10s\t%10s\t%11s\t%11s"%("delta t","average","since","example","learning")
print "%10s\t%10s\t%10s\t%11s\t%11s"%("","loss","last","counter","rate")

lineno=0
for line in f:
    lineno=lineno+1
    if lineno >= maxlines:
        break
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
          assert(np.min(labels) >= 0);
          assert(np.max(labels) < outvocabsize);
        except:
          pdb.set_trace()

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
	#sdticsd=np.array(csr_matrix.sum(sd.multiply(sd),axis=0)) # diag approx
	#sdtransdiff=math.sqrt(sd.shape[0])*np.divide(sdtransdiff,1e-16+sdticsd.transpose())
	#sdtransdiff=math.sqrt(batchsize)*np.divide(sdtransdiff,1e-16+sdticsd.transpose())

        momembeddiff=alpha*momembeddiff+lrs['embedding']*eta*sdtransdiff
        embedding=embedding-momembeddiff
        embedding=(1-lrs['embedding']*weightdecay*eta)*embedding

        net.blobs['data'].data[:]+=sd.dot(momembeddiff)[:,np.newaxis,np.newaxis,:]

	bottom=dict()
	bottom['ip1']='data'
	bottom['ip2']='ip1'
	# TODO: super-slow
	bottom['ip3']='ip2'

        for (name,layer,momlayer) in zip(net._layer_names,net.layers,momnet.layers):
	  if name in bottom:
            myeta=eta
	    x=np.hstack( ( np.squeeze(net.blobs[bottom[name]].data),
	                   np.ones((batchsize,1),dtype='f') ) )
	    xticy=x.transpose().dot(np.squeeze(net.blobs[name].diff))
            #xticy=np.hstack( (np.squeeze(layer.blobs[0].diff),
            #                  np.squeeze(layer.blobs[1].diff)[:,np.newaxis] ) ).transpose()
            W=xticy
            
#	    xticx=x.transpose().dot(x)
#	    datatrace=np.trace(xticx)
#	    xticx+=1e-1*(datatrace/xticx.shape[0])*np.identity(xticx.shape[0])
#	    W=solve(xticx,xticy)

	    momlayer.blobs[0].data[:]=(alpha*momlayer.blobs[0].data
	                               +myeta*W[:-1,:].transpose())
	    momlayer.blobs[1].data[:]=(alpha*momlayer.blobs[1].data
	                               +myeta*W[-1,:])
	    layer.blobs[0].data[:]-=momlayer.blobs[0].data
	    layer.blobs[0].data[:]-=weightdecay*myeta*layer.blobs[0].data
	    layer.blobs[1].data[:]-=momlayer.blobs[1].data
	    layer.blobs[1].data[:]-=weightdecay*myeta*layer.blobs[1].data

            dW=np.squeeze(np.concatenate((np.squeeze(momlayer.blobs[0].data).transpose(),
					  np.squeeze(momlayer.blobs[1].data)[np.newaxis,:]),
					 axis=0))
#	    #net.blobs[name].data[:]+=x.dot(dW)[:,:,np.newaxis,np.newaxis]

            # TODO: sign is wrong, WTF?
	    net.blobs[name].data[:]+=np.multiply(
                                       np.greater(net.blobs[name].data,0),
                                       x.dot(dW)[:,:,np.newaxis,np.newaxis])
            net.blobs[name].data[:]=np.multiply(
                                      np.greater(net.blobs[name].data,0),
                                      net.blobs[name].data)
	  else:
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
            net.save(sys.argv[5]+"."+str(numupdates))
            h5f=h5py.File(sys.argv[5]+"_e."+str(numupdates))
            h5f.create_dataset('embedding',data=embedding)
            h5f.close()
            now=time.time()
            print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
            nextprint=2*nextprint
            numsinceupdates=0
            sumsinceloss=0


now=time.time()
print "%10.3f\t%10.4f\t%10.4f\t%11u\t%11.6g"%(now-start,sumloss/numupdates,sumsinceloss/numsinceupdates,numupdates*batchsize,eta)
net.save(sys.argv[5])
h5f=h5py.File(sys.argv[5]+"_e")
h5f.create_dataset('embedding',data=embedding)
h5f.close()

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding');
#
#GLOG_minloglevel=5 PYTHONPATH=../../python python makefofepresparsemodel.py fofe_sparse_small_unigram_train `cat numlinesfofengram9 | perl -lane 'print int(0.9*$F[0])'` fofengram9.txt.gz histo9 fofepresparsemodel9
#   delta t         average           since          example        learning
#                      loss            last          counter            rate
#     6.998         11.2876         11.2876             1500         0.99999
#    13.794         11.2819         11.2762             3000         0.99998
#    26.830         11.2611         11.2402             6000         0.99996
#    51.337         11.1823         11.1035            12000         0.99992
#   100.520         10.8143         10.4464            24000         0.99984
#   199.769          9.9396          9.0649            48000         0.99968
#   390.774          8.9433          7.9470            96000         0.99936
#   750.326          8.0424          7.1415           192000        0.998721
#  1449.740          7.3005          6.5586           384000        0.997443
#  2791.183          6.7054          6.1103           768000        0.994893
#  6317.486          6.2612          5.8170          1536000        0.989812
# 14908.916          5.9289          5.5966          3072000        0.979728
# 32218.623          5.6623          5.3957          6144000        0.959867
# 66110.485          5.4349          5.2076         12288000        0.921345
#  ...
