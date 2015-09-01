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
eta=1e-2
etadecay=0.999992
weightdecay=1e-3
kappa=0.25
maxiter=200000
maxlambda=float(os.environ['lambda'])
lambdadecay=0.99999
maximpweight=10

# UGH ... so much for DRY

lrs=dict()
lrs[('ip0',0)]=4
lrs[('ip0',1)]=4
lrs[('ip1',0)]=3
lrs[('ip1',1)]=3
lrs[('ip2',0)]=2
lrs[('ip2',1)]=2
lrs[('treated_prediction',0)]=1
lrs[('treated_prediction',1)]=0    # no bias weight allowed (?)
lrs[('untreated_prediction',0)]=1
lrs[('untreated_prediction',1)]=0  # no bias weight allowed (?)
lrs[('treatment_ip3',0)]=2
lrs[('treatment_ip3',1)]=2
lrs[('treatment_prediction',0)]=2
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
sumimpweight=0
sumsinceimpweight=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "mean population diff: %g"%(np.mean(ynsw,dtype='d')-np.mean(ypsid,dtype='d'))

print "%7s %7s %7s %7s %7s %7s %7s %10s %9s"%("delta t","outcome","outcome","treated","imp","ateott","example","learning","gradient")
print "%7s %7s %7s %7s %7s %7s %7s %10s %9s"%("","loss T","loss U","loss","weight","","counter","rate","scale")

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

        print "%7s %7.3f %7.3f %7.3f %7.3f %7.4f %7s %10.5g %9.3e"%(
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

print "%7s %7.3f %7.3f %7.3f %7.3f %7.4f %7s %10.5g %9.3e"%(
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
# delta t outcome outcome treated     imp  ateott example   learning  gradient
#          loss T  loss U    loss  weight         counter       rate     scale
#    32ms   0.586   6.655   0.693   1.000 -0.0093     850  0.0099999 0.000e+00
#    54ms   0.585   6.910   0.693   1.000 -0.0301      1K  0.0099998 0.000e+00
#   102ms   0.577   7.197   0.693   1.000 -0.1075      3K  0.0099997 0.000e+00
#   170ms   0.529   5.943   0.693   1.000 -0.3976      6K  0.0099994 0.000e+00
#   286ms   0.380   3.552   0.693   1.000 -1.8953     13K  0.0099987 0.000e+00
#   496ms   0.348   2.722   0.693   1.000 -1.2550     27K  0.0099974 0.000e+00
#   907ms   0.327   1.717   0.685   0.993 -0.4703     54K  0.0099949 0.000e+00
#  1.728s   0.304   1.138   0.562   0.919 -0.3270    108K  0.0099898 0.000e+00
#  3.358s   0.297   1.070   0.428   0.947 -0.3001    217K  0.0099795 0.000e+00
#  6.616s   0.292   1.058   0.413   0.948 -0.2282    435K  0.0099591 0.000e+00
# 13.244s   0.273   1.028   0.388   0.920 -0.3314    870K  0.0099184 0.000e+00
# 26.990s   0.253   0.999   0.373   0.913 -0.0049      1M  0.0098375 0.000e+00
# 54.392s   0.221   0.916   0.298   0.850  0.0384      3M  0.0096776 0.000e+00
#  1.823m   0.175   0.769   0.213   0.755  0.0773      6M  0.0093657 0.000e+00
#  3.637m   0.132   0.597   0.133   0.662 -0.0117     13M  0.0087715 0.000e+00
#  7.415m   0.086   0.408   0.058   0.570  0.1779     27M   0.007694 0.000e+00
# 14.703m   0.059   0.249   0.029   0.533  0.0247     55M  0.0059198 0.000e+00
# 29.056m   0.048   0.165   0.022   0.524  0.0076    111M  0.0035043 0.000e+00
# 44.453m   0.044   0.126   0.021   0.522  0.0076    170M   0.002019 0.000e+00
