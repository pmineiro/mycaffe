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

# GLOG_minloglevel=5 lambda=0.0 PYTHONPATH=../../python python makenswmodel.py nsw_train nsw_control.txt psid_controls.txt nswplacebomodel
# mean population diff: -1.64639
#
# delta t outcome outcome treated     tot     ate example   learning  gradient
#          loss T  loss U    loss                 counter       rate     scale
#
# 
# 17.421m   0.021   0.042   0.192  0.0769  0.2315    170M  0.0013533 0.000e+00
# 17.365m   0.020   0.040   0.288 -0.0221  0.2299    170M  0.0013533 2.162e-04
# 17.202m   0.020   0.042   0.306  0.1223 -0.1212    170M  0.0013533 4.323e-04
# 17.067m   0.020   0.044   0.313  0.0484  0.1550    170M  0.0013533 6.485e-04
# 17.173m   0.021   0.043   0.377 -0.0728  0.8517    170M  0.0013533 6.917e-04
# 17.213m   0.020   0.043   0.408  0.0516  0.0259    170M  0.0013533 8.647e-04
# 17.281m   0.022   0.040   0.437 -0.1404  0.1548    170M  0.0013533 1.297e-03
# 16.782m   0.022   0.044   0.482 -0.3714  0.2504    170M  0.0013533 1.513e-03
# 17.316m   0.021   0.040   0.467  0.0468  1.9956    170M  0.0013533 1.729e-03
