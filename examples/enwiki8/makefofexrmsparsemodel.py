#! /usr/bin/env python

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

alpha=0.9
eta=1.0
etadecay=0.99999
weightdecay=1e-5
kappa=0.25

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
#with open(sys.argv[4],'r') as priorfile:
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
sumsigma=0
sumsincesigma=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "%10s\t%7s\t%7s\t%7s\t%7s\t%11s\t%11s"%("delta t","average","since","average","since","example","learning")
print "%10s\t%7s\t%7s\t%7s\t%7s\t%11s\t%11s"%("","loss","last","sigma","last","counter","rate")

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
          assert(np.min(labels) >= 0)
          assert(np.max(labels) < outvocabsize)
        except:
          pdb.set_trace()

        sd=csr_matrix((value, (row, col)), shape=(batchsize,invocabsize), dtype='f')
        data=sd.dot(embedding).reshape(batchsize,1,1,embeddingsize)
        net.set_input_arrays(data,labels)
        res=net.forward()
        meanloss=np.mean(net.blobs['lossdetail'].data,dtype='d')
        sumloss+=meanloss
        sumsinceloss+=meanloss
        meansquareloss=np.mean(np.square(net.blobs['lossdetail'].data),dtype='d')
        sigma=math.sqrt(meansquareloss-meanloss*meanloss)
        sumsigma+=sigma
        sumsincesigma+=sigma
        net.blobs['lossdetail'].data[:]=1+kappa*np.divide(net.blobs['lossdetail'].data-meanloss,sigma)
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
            net.save(sys.argv[5]+"."+str(numupdates))
            h5f=h5py.File(sys.argv[5]+"_e."+str(numupdates))
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
net.save(sys.argv[5])
h5f=h5py.File(sys.argv[5]+"_e")
h5f.create_dataset('embedding',data=embedding)
h5f.close()

# import to matlab:
# >> Z=h5read('fofesparsemodel9_e','/embedding')

# GLOG_minloglevel=5 PYTHONPATH=../../python python makefofexrmsparsemodel.py fofe_xrm_sparse_small_unigram_train <(head -n `cat numlinesfofengram9 | perl -lane 'print int(0.9*$F[0])'` fofengram9.txt) histo9 fofexrmsparsemodel9

# kappa = 0
#   delta t      average   since average   since     example        learning
#                   loss    last   sigma    last     counter            rate
#     2.437      11.2876 11.2876  0.0107  0.0107        1500         0.99999
#     4.945      11.2819 11.2762  0.0178  0.0249        3000         0.99998
#     9.257      11.2610 11.2402  0.0490  0.0801        6000         0.99996
#    16.077      11.1815 11.1020  0.1699  0.2909       12000         0.99992
#    30.085      10.7707 10.3599  0.8600  1.5501       24000         0.99984
#    59.223       9.9729  9.1751  2.0945  3.3290       48000         0.99968
#   108.574       9.0664  8.1598  2.9729  3.8514       96000         0.99936
#   202.946       8.2335  7.4007  3.3921  3.8114      192000        0.998721
#   390.500       7.4495  6.6654  3.5822  3.7723      384000        0.997443
#   780.970       6.8001  6.1508  3.7509  3.9197      768000        0.994893
#  1862.865       6.3219  5.8437  3.8543  3.9577     1536000        0.989812
#  4709.038       5.9713  5.6208  3.9015  3.9486     3072000        0.979728
# 11040.083       5.6994  5.4274  3.9093  3.9171     6144000        0.959867
# 24946.610       5.4741  5.2488  3.8879  3.8665    12288000        0.921345
# 52319.283       5.2782  5.0823  3.8478  3.8078    24576000        0.848877
#107286.415       5.0992  4.9203  3.7954  3.7430    49152000        0.720592
#219665.972       4.9281  4.7570  3.7354  3.6754    98304000        0.519253
#252354.159       4.8964  4.6667  3.7236  3.6381   111871500        0.474348
# ...

# kappa = 0.2
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
# 213760.106       4.8808  4.6506  3.5850  3.4990   111871500        0.474348

# kappa = 0.25
#   delta t      average   since average   since     example        learning
#                   loss    last   sigma    last     counter            rate
#     2.592      11.2876 11.2876  0.0107  0.0107        1500         0.99999
#     4.745      11.2829 11.2783  0.0148  0.0189        3000         0.99998
#     8.340      11.2676 11.2523  0.0327  0.0506        6000         0.99996
#    14.850      11.2149 11.1622  0.0960  0.1594       12000         0.99992
#    26.999      10.9977 10.7805  0.3587  0.6214       24000         0.99984
#    50.484       9.9163  8.8349  1.7072  3.0557       48000         0.99968
#    96.601       8.9120  7.9076  2.5716  3.4360       96000         0.99936
#   187.981       8.0186  7.1252  3.0269  3.4822      192000        0.998721
#   370.063       7.3810  6.7434  3.2206  3.4143      384000        0.997443
#   737.457       6.8115  6.2420  3.4410  3.6613      768000        0.994893
#  1516.845       6.3395  5.8674  3.6103  3.7797     1536000        0.989812
#  3515.733       5.9782  5.6170  3.6981  3.7858     3072000        0.979728
#  8274.246       5.6929  5.4076  3.7268  3.7555     6144000        0.959867
# 18691.464       5.4569  5.2209  3.7145  3.7023    12288000        0.921345
# 40316.075       5.2557  5.0544  3.6777  3.6409    24576000        0.848877
# 85082.901       5.0760  4.8962  3.6266  3.5754    49152000        0.720592
#171397.618       4.9066  4.7372  3.5675  3.5084    98304000        0.519253
#195224.626       4.8754  4.6492  3.5559  3.4716   111871500        0.474348

# kappa = 0.3
#   delta t      average   since average   since     example        learning
#                   loss    last   sigma    last     counter            rate
#     2.262      11.2876 11.2876  0.0107  0.0107        1500         0.99999
#     4.483      11.2831 11.2787  0.0147  0.0187        3000         0.99998
#     7.997      11.2688 11.2544  0.0304  0.0461        6000         0.99996
#    15.195      11.2205 11.1723  0.0860  0.1416       12000         0.99992
#    27.975      11.0339 10.8474  0.3037  0.5215       24000         0.99984
#    52.158      10.0123  8.9906  1.6067  2.9097       48000         0.99968
#    97.794       9.0796  8.1470  2.4521  3.2975       96000         0.99936
#   191.882       8.1321  7.1846  2.9490  3.4460      192000        0.998721
#   372.764       7.4052  6.6783  3.1806  3.4122      384000        0.997443
#   735.480       6.8153  6.2253  3.4074  3.6342      768000        0.994893
#  1751.300       6.3434  5.8716  3.5775  3.7475     1536000        0.989812
#  4457.271       5.9834  5.6234  3.6661  3.7547     3072000        0.979728
# 10251.863       5.6987  5.4141  3.6957  3.7253     6144000        0.959867
# ...

# kappa = 0.4
#   delta t      average   since average   since     example        learning
#                   loss    last   sigma    last     counter            rate
#     2.509      11.2876 11.2876  0.0107  0.0107        1500         0.99999
#     4.953      11.2836 11.2795  0.0136  0.0166        3000         0.99998
#     8.512      11.2711 11.2587  0.0260  0.0384        6000         0.99996
#    15.115      11.2310 11.1909  0.0692  0.1124       12000         0.99992
#    27.211      11.0910 10.9509  0.2226  0.3759       24000         0.99984
#    50.754      10.1750  9.2590  1.3112  2.3999       48000         0.99968
#    96.757       9.0349  7.8948  2.2400  3.1688       96000         0.99936
#   188.029       8.0931  7.1513  2.7335  3.2269      192000        0.998721
#   369.869       7.4167  6.7403  2.9936  3.2538      384000        0.997443
#   732.889       6.8634  6.3100  3.2406  3.4876      768000        0.994893
#  1511.109       6.3903  5.9172  3.4482  3.6559     1536000        0.989812
#  3500.726       6.0181  5.6460  3.5672  3.6862     3072000        0.979728
#  8291.888       5.7257  5.4333  3.6146  3.6619     6144000        0.959867
# ...

# kappa = 0.5
#   delta t      average   since average   since     example        learning
#                   loss    last   sigma    last     counter            rate
#     2.683      11.2876 11.2876  0.0107  0.0107        1500         0.99999
#     5.196      11.2840 11.2803  0.0127  0.0146        3000         0.99998
#     9.257      11.2731 11.2622  0.0228  0.0328        6000         0.99996
#    17.124      11.2395 11.2059  0.0570  0.0912       12000         0.99992
#    30.200      11.1334 11.0273  0.1660  0.2750       24000         0.99984
#    54.298      10.3208  9.5081  1.0257  1.8854       48000         0.99968
#   102.197       9.0249  7.7291  2.0379  3.0501       96000         0.99936
#   192.121       8.0568  7.0887  2.5718  3.1056      192000        0.998721
#   371.387       7.4211  6.7853  2.8629  3.1539      384000        0.997443
#   729.432       6.8985  6.3759  3.1199  3.3770      768000        0.994893
#  1504.468       6.4300  5.9616  3.3453  3.5707     1536000        0.989812
#  3465.981       6.0481  5.6663  3.4839  3.6226     3072000        0.979728
#  8158.444       5.7428  5.4375  3.5446  3.6053     6144000        0.959867
# 18450.990       5.4926  5.2425  3.5514  3.5583    12288000        0.921345
# ...

