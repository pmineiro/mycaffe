#! /usr/bin/env python

import caffe
import math
import numpy as np
import os
import sys
import time
import pdb

np.random.seed(8675309)

alpha=0.9
eta=float(os.environ['eta'])
etadecay=0.99999
weightdecay=1e-7
maxiter=200000
maxlambda=float(os.environ['lambda'])
lambdadecay=0.99999

# UGH ... so much for DRY

lrs=dict()
lrs[('ipm1',0)]=1
lrs[('ipm1',1)]=2
lrs[('ip0',0)]=1
lrs[('ip0',1)]=2
lrs[('ip1',0)]=1
lrs[('ip1',1)]=2
lrs[('ip2',0)]=1  
lrs[('ip2',1)]=2  
lrs[('treated_prediction',0)]=1
lrs[('treated_prediction',1)]=2    
lrs[('untreated_prediction',0)]=1
lrs[('untreated_prediction',1)]=2  
lrs[('treatment_prediction',0)]=10
lrs[('treatment_prediction',1)]=20

def nicetime(dt):
  if (dt < 1):
    return "%4.0fms"%(1000.0*dt)
  elif (dt < 60):
    return "%2.3fs"%(dt)
  elif (dt < 60*60):
    return "%2.3fm"%(dt/60)
  elif (dt < 60*60*24):
    return "%2.3fh"%(dt/(60*60))
  else:
    return "%2.4fd"%(dt/(60*60*24))

def nicecount(n):
  if (n < 1000):
    return "%4u"%(n)
  elif (n < 1000*1000):
    return "%4uK"%(n/1000)
  elif (n < 1000*1000*1000):
    return "%4uM"%(n/(1000*1000))
  else:
    return "%4uB"%(n/(1000*1000*1000))

net = caffe.Net(sys.argv[1])
net.set_mode_gpu()
net.set_phase_train()

halfbatchsize=(net.blobs['data'].data.shape[0]/2)

momnet = caffe.Net(sys.argv[1])
momnet.set_mode_gpu()
momnet.set_phase_train()

for (layer,momlayer) in zip(net.layers,momnet.layers):
  blobnum=0
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    if blobnum == 0:
      fan_in = float(blob.count) / float(blob.num)
      scale = math.sqrt(3.0/fan_in)
      blob.data[:]=np.random.uniform(low=-scale,high=scale,size=blob.data.shape)
    else:
      blob.data[:]=0
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')
    blobnum=blobnum+1

# nsw (treated or control)
with open(sys.argv[2],'r') as f:
  lineno=0
  for line in f:
    lineno=lineno+1
  f.seek(0,0)
  xnsw=np.zeros((lineno,1,1,7),dtype='f')
  ynsw=np.zeros(lineno,dtype='f')
  lineno=0
  for line in f:
    cols=[word for word in line.split()]
    xnsw[lineno,0,0,:]=[float(col) for col in cols[1:-1]]
    ynsw[lineno]=float(cols[-1])/10000.0
    lineno=lineno+1

# psid
with open(sys.argv[3],'r') as f:
  lineno=0
  for line in f:
    lineno=lineno+1
  f.seek(0,0)
  xpsid=np.zeros((lineno,1,1,7),dtype='f')
  ypsid=np.zeros(lineno,dtype='f')
  lineno=0
  for line in f:
    cols=[word for word in line.split()]
    xpsid[lineno,0,0,0:6]=[float(col) for col in cols[1:-3]]
    xpsid[lineno,0,0,6]=float(cols[-2])
    ypsid[lineno]=float(cols[-1])/10000.0
    lineno=lineno+1

# sphere (all) data b/c first order learning sucks
meanx=np.mean(np.vstack((xnsw,xpsid)),axis=0,dtype='d')
xnsw=xnsw-meanx
xpsid=xpsid-meanx
varx=np.sqrt(np.mean(np.square(np.vstack((xnsw,xpsid))),axis=0,dtype='d'))
xnsw=np.divide(xnsw,varx).astype('f')
xpsid=np.divide(xpsid,varx).astype('f')

bindex=0
start=time.time()
numsinceupdates=0
numupdates=0
sumtreatedloss=0
sumsincetreatedloss=0
sumuntreatedloss=0
sumsinceuntreatedloss=0
sumtreatmentloss=0
sumsincetreatmentloss=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

meanynsw=np.mean(ynsw,dtype='d')
varynsw=np.mean(np.square(ynsw-meanynsw),dtype='d')
meanypsid=np.mean(ypsid,dtype='d')
varypsid=np.mean(np.square(ypsid-meanypsid),dtype='d')

net.params['treated_prediction'][1].data[:]=meanynsw
lrs[('treated_prediction',1)]=0
net.params['untreated_prediction'][1].data[:]=meanypsid
lrs[('untreated_prediction',1)]=0

print "mean population diff: %g"%(np.mean(ynsw,dtype='d')-np.mean(ypsid,dtype='d'))
print "constant loss: (T) %g (U) %g"%(varynsw,varypsid)

print "%7s %7s %7s %7s %7s %7s %7s %10s %9s"%("delta t","outcome","outcome","treated","tot","ate","example","learning","gradient")
print "%7s %7s %7s %7s %7s %7s %7s %10s %9s"%("","loss T","loss U","loss","","","counter","rate","scale")

maxminlambda=maxlambda
totaldata=xnsw.shape[0]+xpsid.shape[0]
for ii in range(maxiter):
    psidperm=np.random.permutation(xpsid.shape[0])
    data=np.vstack((xnsw,
                    xpsid[psidperm[0:halfbatchsize],:]))
    labels=np.hstack((ynsw,
                      ypsid[psidperm[0:halfbatchsize]]))
    net.set_input_arrays(data,labels)
    res=net.forward()
    treatedloss=res['treated_loss'][0,0,0,0]
    untreatedloss=res['untreated_loss'][0,0,0,0]
    treatmentloss=res['treatment_loss'][0,0,0,0]

    sumtreatedloss+=treatedloss
    sumsincetreatedloss+=treatedloss
    sumuntreatedloss+=untreatedloss
    sumsinceuntreatedloss+=untreatedloss
    sumtreatmentloss+=treatmentloss
    sumsincetreatmentloss+=treatmentloss

    net.blobs['treatmentpredictorscaleby'].data[:]=maxminlambda-maxlambda
    net.backward()

    for (name,layer,momlayer) in zip(net._layer_names,net.layers,momnet.layers):
      blobnum=0
      for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
        myeta=lrs[(name,blobnum)]*eta
        momblob.data[:]=alpha*momblob.data[:]+myeta*blob.diff
        blob.data[:]-=momblob.data[:]
        blob.data[:]=(1-weightdecay*myeta)*blob.data[:]
        blobnum=blobnum+1

    eta=eta*etadecay
    maxminlambda=maxminlambda*lambdadecay

    numupdates=numupdates+1
    numsinceupdates=numsinceupdates+1
    if numupdates >= nextprint:
        net.set_phase_test()

        net.save(sys.argv[4]+"."+str(numupdates))
        data=np.vstack((xnsw,xnsw))
        labels=np.hstack((ynsw,ynsw))
        net.set_input_arrays(data,labels)
        net.forward()

        ateott=np.mean(net.blobs['treated_prediction'].data-
                       net.blobs['untreated_prediction'].data,
                       dtype='d')

        tot=np.mean(net.blobs['treated_prediction'].data-
                    net.blobs['untreated_prediction'].data,
                    dtype='d')

        psidperm=np.random.permutation(xpsid.shape[0])
        data=np.vstack((xpsid[psidperm[0:halfbatchsize],:],
                        xpsid[psidperm[0:halfbatchsize],:]))
        labels=np.hstack((ypsid[psidperm[0:halfbatchsize]],
                          ypsid[psidperm[0:halfbatchsize]]))
        net.set_input_arrays(data,labels)
        net.forward()
        tou=np.mean(net.blobs['treated_prediction'].data-
                    net.blobs['untreated_prediction'].data,
                    dtype='d')
        ate=0.5*(tot+tou)

        now=time.time()
        dt=float(now-start)

        print "%7s %7.3f %7.3f %7.3f %7.4f %7.4f %7s %10.5g %9.3e"%(
            nicetime(dt),
            sumsincetreatedloss/numsinceupdates,
            sumsinceuntreatedloss/numsinceupdates,
            sumsincetreatmentloss/numsinceupdates,
            ateott,
            ate,
            nicecount(2*numupdates*halfbatchsize),
            eta,
            maxlambda-maxminlambda)
        nextprint=2*nextprint
        numsinceupdates=0
        sumsincetreatedloss=0
        sumsinceuntreatedloss=0
        sumsincetreatmentloss=0

        net.set_phase_train()


net.save(sys.argv[4])
now=time.time()
dt=float(now-start)

print "%7s %7.3f %7.3f %7.3f %7.4f %7.4f %7s %10.5g %9.3e"%(
    nicetime(dt),
    sumsincetreatedloss/numsinceupdates,
    sumsinceuntreatedloss/numsinceupdates,
    sumsincetreatmentloss/numsinceupdates,
    ateott,
    ate,
    nicecount(2*numupdates*halfbatchsize),
    eta,
    maxlambda-maxminlambda)

# lambda=0
# GLOG_minloglevel=5 lambda=0 PYTHONPATH=../../python python makenswmodel.py nsw_train nsw_control.txt psid_controls.txt nswplacebomodel
# mean population diff: -1.64639
# constant loss: (T) 0.326196 (U) 2.41872
# delta t outcome outcome treated     tot     ate example   learning  gradient
#          loss T  loss U    loss                 counter       rate     scale
#    35ms   0.163   1.227   0.693 -1.6463 -1.6463     850  0.0099999 0.000e+00
#    47ms   0.163   1.054   0.693 -1.6454 -1.6454      1K  0.0099998 0.000e+00
#    65ms   0.163   1.348   0.693 -1.6451 -1.6451      3K  0.0099996 0.000e+00
#    92ms   0.163   1.266   0.693 -1.6481 -1.6481      6K  0.0099992 0.000e+00
#   148ms   0.163   1.323   0.693 -1.6537 -1.6537     13K  0.0099984 0.000e+00
#   238ms   0.163   1.184   0.693 -1.6554 -1.6555     27K  0.0099968 0.000e+00
#   415ms   0.163   1.194   0.693 -1.6472 -1.6472     54K  0.0099936 0.000e+00
#   739ms   0.163   1.202   0.693 -1.6468 -1.6468    108K  0.0099872 0.000e+00
#  1.352s   0.163   1.224   0.693 -1.6458 -1.6458    217K  0.0099744 0.000e+00
#  2.639s   0.163   1.203   0.693 -1.6368 -1.6368    435K  0.0099489 0.000e+00
#  6.098s   0.163   1.207   0.693 -1.6415 -1.6416    870K  0.0098981 0.000e+00
# 13.599s   0.160   1.018   0.609 -0.2517 -0.7185      1M  0.0097973 0.000e+00
# 28.311s   0.146   0.513   0.412 -0.1976 -0.6088      3M  0.0095987 0.000e+00
# 56.059s   0.133   0.460   0.428 -0.2196 -0.6090      6M  0.0092135 0.000e+00
#  1.767m   0.103   0.370   0.420 -0.1332 -0.3583     13M  0.0084888 0.000e+00
#  3.233m   0.067   0.251   0.343 -0.1544 -0.3806     27M  0.0072059 0.000e+00
#  5.990m   0.038   0.131   0.247 -0.2348 -0.1821     55M  0.0051925 0.000e+00
# 11.447m   0.024   0.062   0.200  0.0769  0.2315    111M  0.0026962 0.000e+00
# 17.167m   0.021   0.042   0.192  0.0769  0.2315    170M  0.0013533 0.000e+00

# "cherry picked lambda"
#
# GLOG_minloglevel=5 lambda=1e-3 PYTHONPATH=../../python python makenswmodel.py nsw_train nsw_control.txt psid_controls.txt nswplacebomodel
# mean population diff: -1.64639
# constant loss: (T) 0.326196 (U) 2.41872
# delta t outcome outcome treated     tot     ate example   learning  gradient
#          loss T  loss U    loss                 counter       rate     scale
#    35ms   0.163   1.227   0.693 -1.6463 -1.6463     850  0.0099999 1.000e-08
#    46ms   0.163   1.054   0.693 -1.6454 -1.6454      1K  0.0099998 2.000e-08
#    64ms   0.163   1.348   0.693 -1.6451 -1.6451      3K  0.0099996 4.000e-08
#    99ms   0.163   1.266   0.693 -1.6481 -1.6481      6K  0.0099992 8.000e-08
#   148ms   0.163   1.323   0.693 -1.6537 -1.6537     13K  0.0099984 1.600e-07
#   237ms   0.163   1.184   0.693 -1.6554 -1.6555     27K  0.0099968 3.200e-07
#   419ms   0.163   1.194   0.693 -1.6472 -1.6472     54K  0.0099936 6.398e-07
#   727ms   0.163   1.202   0.693 -1.6468 -1.6469    108K  0.0099872 1.279e-06
#  1.335s   0.163   1.224   0.693 -1.6458 -1.6458    217K  0.0099744 2.557e-06
#  2.613s   0.163   1.203   0.693 -1.6368 -1.6368    435K  0.0099489 5.107e-06
#  6.091s   0.163   1.208   0.693 -1.6403 -1.6403    870K  0.0098981 1.019e-05
# 13.764s   0.163   1.123   0.654 -0.2708 -0.8176      1M  0.0097973 2.027e-05
# 28.292s   0.148   0.516   0.416 -0.2777 -0.6817      3M  0.0095987 4.013e-05
# 55.728s   0.132   0.460   0.419 -0.1648 -0.6507      6M  0.0092135 7.865e-05
#  1.762m   0.103   0.363   0.392 -0.3262 -0.5864     13M  0.0084888 1.511e-04
#  3.284m   0.070   0.236   0.375 -0.3111 -0.5696     27M  0.0072059 2.794e-04
#  6.234m   0.038   0.123   0.307 -0.0799 -0.0929     55M  0.0051925 4.807e-04
# 12.024m   0.023   0.063   0.369  0.0516  0.0259    111M  0.0026962 7.304e-04
# 17.754m   0.020   0.043   0.408  0.0516  0.0259    170M  0.0013533 8.647e-04

# unfortunately, results are very sensitive to lambda
#
# delta t outcome outcome treated     tot     ate example   learning  gradient
#          loss T  loss U    loss                 counter       rate     scale
#
# 17.421m   0.021   0.042   0.192  0.0769  0.2315    170M  0.0013533 0.000e+00
# 17.153m   0.020   0.038   0.235 -0.2317  0.1375    170M  0.0013533 8.647e-05
# 17.213m   0.020   0.043   0.408  0.0516  0.0259    170M  0.0013533 8.647e-04
# 17.363m   0.020   0.040   0.660 -0.8712 -0.8967    170M  0.0013533 8.647e-03
