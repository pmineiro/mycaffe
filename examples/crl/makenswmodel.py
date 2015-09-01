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
eta=1e-3
etadecay=0.999992
weightdecay=1e-3
kappa=0.25
maxiter=200000
maxlambda=float(os.environ['lambda'])
lambdadecay=0.99999
maximpweight=10

# UGH ... so much for DRY

lrs=dict()
lrs[('ip0',0)]=40
lrs[('ip0',1)]=40
lrs[('bn0',0)]=40
lrs[('bn0',1)]=40
lrs[('ip1',0)]=30
lrs[('ip1',1)]=30
lrs[('bn1',0)]=30
lrs[('bn1',1)]=30
lrs[('ip2',0)]=20
lrs[('ip2',1)]=20
lrs[('bn2',0)]=20
lrs[('bn2',1)]=20
lrs[('treated_prediction',0)]=1
lrs[('treated_prediction',1)]=0    # no bias weight allowed (?)
lrs[('untreated_prediction',0)]=1
lrs[('untreated_prediction',1)]=0  # no bias weight allowed (?)
lrs[('treatment_ip3',0)]=20
lrs[('treatment_ip3',1)]=20
lrs[('treatment_prediction',0)]=20
lrs[('treatment_prediction',1)]=0  # ... treatment is apriori balanced (?)

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
net.set_mode_cpu()
net.set_phase_train()

halfbatchsize=(net.blobs['data'].data.shape[0]/2)

momnet = caffe.Net(sys.argv[1])
momnet.set_mode_cpu()
momnet.set_phase_train()

for (layer,momlayer) in zip(net.layers,momnet.layers):
  for (blob,momblob) in zip(layer.blobs,momlayer.blobs):
    #blob.data[:]=0.01*np.random.standard_normal(size=blob.data.shape)
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')

net.params['treated_prediction'][1].data[:]=0
net.params['untreated_prediction'][1].data[:]=0
net.params['treatment_prediction'][1].data[:]=0

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
sumimpweight=0
sumsinceimpweight=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "mean population diff: %g"%(np.mean(ynsw,dtype='d')-np.mean(ypsid,dtype='d'))

print "%7s %7s %7s %7s %7s %6s %7s %10s %9s"%("delta t","outcome","outcome","treated","imp","ateott","example","learning","gradient")
print "%7s %7s %7s %7s %7s %6s %7s %10s %9s"%("","loss T","loss U","loss","weight","","counter","rate","scale")

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
    treatedloss=np.mean(net.blobs['treated_lossdetail'].data,dtype='d')
    untreatedloss=np.mean(net.blobs['untreated_lossdetail'].data,dtype='d')
    treatmentloss=np.mean(net.blobs['treatment_lossdetail'].data,dtype='d')

    sumtreatedloss+=treatedloss
    sumsincetreatedloss+=treatedloss
    sumuntreatedloss+=untreatedloss
    sumsinceuntreatedloss+=untreatedloss
    sumtreatmentloss+=treatmentloss
    sumsincetreatmentloss+=treatmentloss

    # TODO: importance weight, xrm, etc.
    net.blobs['treated_lossdetail'].data[:,0,0,0]=np.fmin(maximpweight,np.divide(0.5,1e-6+net.blobs['treatment_probability'].data[0:halfbatchsize,1,0,0]))
    treatedimpweight=np.mean(net.blobs['treated_lossdetail'].data,dtype='d')
    net.blobs['untreated_lossdetail'].data[:,0,0,0]=np.fmin(maximpweight,np.divide(0.5,1e-6+net.blobs['treatment_probability'].data[-halfbatchsize:,0,0,0]))
    untreatedimpweight=np.mean(net.blobs['untreated_lossdetail'].data,dtype='d')
    
    net.blobs['treated_lossdetail'].data[:,0,0,0]=np.divide(net.blobs['treated_lossdetail'].data[:,0,0,0],treatedimpweight)
    net.blobs['untreated_lossdetail'].data[:,0,0,0]=np.divide(net.blobs['untreated_lossdetail'].data[:,0,0,0],untreatedimpweight)

    impweight=0.5*(treatedimpweight+untreatedimpweight)
    sumimpweight+=impweight
    sumsinceimpweight+=impweight

    net.blobs['treatment_lossdetail'].data[:]=1
    net.blobs['scaleby'].data[:]=maxminlambda-maxlambda
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
        net.save(sys.argv[4]+"."+str(numupdates))
        data=np.vstack((xnsw,xnsw))
        labels=np.hstack((ynsw,ynsw))
        net.set_input_arrays(data,labels)
        net.forward()
        ateott=np.mean(net.blobs['treated_prediction'].data-
                       net.blobs['untreated_prediction'].data,
                       dtype='d')

#        data=np.vstack((xpsid,xpsid))
#        labels=np.hstack((ypsid,ypsid))
#        net.blobs['data'].reshape(data.shape[0],data.shape[1],data.shape[2],data.shape[3])
#        net.set_input_arrays(data,labels)
        
        now=time.time()
        dt=float(now-start)

        print "%7s %7.3f %7.3f %7.3f %7.3f %6.3f %7s %10.5g %9.3e"%(
            nicetime(dt),
            sumsincetreatedloss/numsinceupdates,
            sumsinceuntreatedloss/numsinceupdates,
            sumsincetreatmentloss/numsinceupdates,
            sumsinceimpweight/numsinceupdates,
            ateott,
            nicecount(2*numupdates*halfbatchsize),
            eta,
            maxlambda-maxminlambda)
        nextprint=2*nextprint
        numsinceupdates=0
        sumsincetreatedloss=0
        sumsinceuntreatedloss=0
        sumsincetreatmentloss=0
        sumsinceimpweight=0


net.save(sys.argv[4])
now=time.time()
dt=float(now-start)

print "%7s %7.3f %7.3f %7.3f %7.3f %6.3f %7s %10.5g %9.3e"%(
    nicetime(dt),
    sumsincetreatedloss/numsinceupdates,
    sumsinceuntreatedloss/numsinceupdates,
    sumsincetreatmentloss/numsinceupdates,
    sumsinceimpweight/numsinceupdates,
    ateott,
    nicecount(2*numupdates*halfbatchsize),
    eta,
    maxlambda-maxminlambda)

# GLOG_minloglevel=5 lambda=0.0 PYTHONPATH=../../python python makenswmodel.py nsw_train nsw_control.txt psid_controls.txt nswplacebomodel
# mean population diff: -1.64639
# delta t outcome outcome treated     imp ateott example   learning  gradient
#          loss T  loss U    loss  weight        counter       rate     scale
#    32ms   0.584   7.122   0.693   1.000 -0.011     850 0.00099999 0.000e+00
#    59ms   0.581   7.442   0.693   1.000 -0.040      1K 0.00099998 0.000e+00
#    95ms   0.567   7.690   0.693   1.000 -0.153      3K 0.00099997 0.000e+00
#   156ms   0.513   5.861   0.693   1.000 -0.518      6K 0.00099994 0.000e+00
#   280ms   0.391   3.458   0.693   1.000 -1.496     13K 0.00099987 0.000e+00
#   519ms   0.336   2.471   0.693   1.000 -1.776     27K 0.00099974 0.000e+00
#   921ms   0.327   1.761   0.688   0.997 -0.876     54K 0.00099949 0.000e+00
#  1.760s   0.311   1.162   0.594   0.932 -0.417    108K 0.00099898 0.000e+00
#  3.346s   0.296   1.068   0.428   0.950 -0.243    217K 0.00099795 0.000e+00
#  6.605s   0.290   1.060   0.392   0.941 -0.254    435K 0.00099591 0.000e+00
# 13.237s   0.273   1.031   0.387   0.932 -0.182    870K 0.00099184 0.000e+00
# 26.385s   0.256   1.002   0.356   0.900 -0.078      1M 0.00098375 0.000e+00
# 52.987s   0.232   0.931   0.268   0.814 -0.151      3M 0.00096776 0.000e+00
#  1.779m   0.192   0.821   0.189   0.757 -0.004      6M 0.00093657 0.000e+00
#  3.551m   0.141   0.622   0.126   0.666  0.100     13M 0.00087715 0.000e+00
#  7.127m   0.109   0.458   0.057   0.574  0.005     27M  0.0007694 0.000e+00
# 14.398m   0.086   0.330   0.032   0.539 -0.013     55M 0.00059198 0.000e+00
# 28.744m   0.064   0.231   0.019   0.523 -0.009    111M 0.00035043 0.000e+00
