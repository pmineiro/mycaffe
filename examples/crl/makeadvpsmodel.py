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
weightdecay=1e-7
maxiter=200000

# UGH ... so much for DRY

lrs=dict()
lrs[('scoreip0',0)]=1
lrs[('scoreip0',1)]=2
lrs[('scorelogit',0)]=1
lrs[('scorelogit',1)]=2

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

halfbatchsize=(net.blobs['covariates'].data.shape[0]/2)

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
  xnsw=np.zeros((lineno,7,1,1),dtype='f')
  ynsw=np.zeros(lineno,dtype='f')
  lineno=0
  for line in f:
    cols=[word for word in line.split()]
    xnsw[lineno,:,0,0]=[float(col) for col in cols[1:-1]]
    ynsw[lineno]=float(cols[-1])/10000.0
    lineno=lineno+1

# psid
with open(sys.argv[3],'r') as f:
  lineno=0
  for line in f:
    lineno=lineno+1
  f.seek(0,0)
  xpsid=np.zeros((lineno,7,1,1),dtype='f')
  ypsid=np.zeros(lineno,dtype='f')
  lineno=0
  for line in f:
    cols=[word for word in line.split()]
    xpsid[lineno,0:6,0,0]=[float(col) for col in cols[1:-3]]
    xpsid[lineno,6,0,0]=float(cols[-2])
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
sumbalanceloss=0
sumsincebalanceloss=0
sumtreatmentloss=0
sumsincetreatmentloss=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "%7s %7s %7s %7s %10s %9s"%("delta t","balance","log","example","learning","")
print "%7s %7s %7s %7s %10s %9s"%("","loss","loss","counter","rate","")

for ii in range(maxiter):
    psidperm=np.random.permutation(xpsid.shape[0])
    data=np.vstack((xnsw,
                    xpsid[psidperm[0:halfbatchsize],:]))
    labels=np.hstack((ynsw,
                      ypsid[psidperm[0:halfbatchsize]]))
    net.set_input_arrays(data,labels)
    res=net.forward()
    
    meantreated=np.mean(
        np.divide(net.blobs['treated_statistics'].data,
                  net.blobs['score'].data[0:halfbatchsize,1:2,:,:]),
        dtype='d',axis=0)
    meanuntreated=np.mean(
        np.divide(net.blobs['untreated_statistics'].data,
                  net.blobs['score'].data[-halfbatchsize:,0:1,:,:]),
        dtype='d',axis=0)
        
    sumbalanceloss+=res['balance_loss']
    sumsincebalanceloss+=res['balance_loss']
    sumtreatmentloss+=res['treatment_loss']
    sumsincetreatmentloss+=res['treatment_loss']

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

    numupdates=numupdates+1
    numsinceupdates=numsinceupdates+1
    if numupdates >= nextprint:
        net.save(sys.argv[4]+"."+str(numupdates))

        now=time.time()
        dt=float(now-start)

        print "%7s %7.3f %7.4f %7s %10.5g"%(
            nicetime(dt),
            sumsincebalanceloss/numsinceupdates,
            sumsincetreatmentloss/numsinceupdates,
            nicecount(2*numupdates*halfbatchsize),
            eta)
        nextprint=2*nextprint
        numsinceupdates=0
        sumsincetreatedloss=0
        sumsinceuntreatedloss=0
        sumsincetreatmentloss=0
        sumsinceimpweight=0


net.save(sys.argv[4])
now=time.time()
dt=float(now-start)

print "%7s %7.3f %7.4f %7s %10.5g"%(
    nicetime(dt),
    sumsincebalanceloss/numsinceupdates,
    sumsincetreatmentloss/numsinceupdates,
    nicecount(2*numupdates*halfbatchsize),
    eta)
