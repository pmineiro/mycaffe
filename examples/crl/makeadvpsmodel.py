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
etadecay=0.999992
weightdecay=1e-7
maxiter=200000
maxlambda=float(os.environ['lambda'])
lambdadecay=0 #0.999992

# UGH ... so much for DRY

lrs=dict()
lrs[('scoreip0',0)]=1
lrs[('scoreip0',1)]=2
lrs[('prescore',0)]=1
lrs[('prescore',1)]=2

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
sumregloss=0
sumsinceregloss=0
summaxweight=0
sumsincemaxweight=0
sumate=0
sumsinceate=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "mean population diff: %g"%(np.mean(ynsw,dtype='d')-np.mean(ypsid,dtype='d'))
print "%7s %7s %7s %7s %11s %7s %11s %9s"%("delta t","balance","max","reg","ate","example","learning","balance")
print "%7s %7s %7s %7s %11s %7s %11s %9s"%("","loss","weight","loss","","counter","rate","lambda")

maxminlambda=maxlambda
for ii in range(maxiter):
    psidperm=np.random.permutation(xpsid.shape[0])
    data=np.vstack((xnsw,
                    xpsid[psidperm[0:halfbatchsize],:]))
    labels=np.hstack((ynsw,
                      ypsid[psidperm[0:halfbatchsize]]))
    net.set_input_arrays(data,labels)
    res=net.forward()
    
    ate = ( np.dot(ynsw,net.blobs['treated_weight'].data.flat) -
            np.dot(ypsid[psidperm[0:halfbatchsize]],
                   net.blobs['untreated_weight'].data.flat) 
    )

    maxweight=max(np.max(net.blobs['treated_weight'].data),
                  np.max(net.blobs['untreated_weight'].data))

    sumbalanceloss+=res['balance_loss']
    sumsincebalanceloss+=res['balance_loss']
    summaxweight+=maxweight
    sumsincemaxweight+=maxweight
    sumregloss+=res['treated_weight_regularizer']+res['untreated_weight_regularizer']
    sumsinceregloss+=res['treated_weight_regularizer']+res['untreated_weight_regularizer']
    sumate+=ate
    sumsinceate+=ate

    net.blobs['balancelambda0'].data[:]=maxlambda-maxminlambda
    net.blobs['balancelambda1'].data[:]=maxlambda-maxminlambda

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

        now=time.time()
        dt=float(now-start)

        print "%7s %7.3f %7.3f %7.1e %11.4f %7s %11.5e %9.3e"%(
            nicetime(dt),
            sumsincebalanceloss/numsinceupdates,
            sumsincemaxweight/numsinceupdates,
            sumsinceregloss/numsinceupdates,
            sumsinceate/numsinceupdates,
            nicecount(2*numupdates*halfbatchsize),
            eta,
            net.blobs['balancelambda0'].data.flat[0])
        nextprint=2*nextprint
        numsinceupdates=0
        sumsincebalanceloss=0
        sumsincetreatedloss=0
        sumsinceuntreatedloss=0
        sumsincemaxweight=0
        sumsinceregloss=0
        sumsinceate=0

now=time.time()
dt=float(now-start)

print "%7s %7.3f %7.3f %7.1e %11.4f %7s %11.5e %9.3e"%(
    nicetime(dt),
    sumsincebalanceloss/numsinceupdates,
    sumsincemaxweight/numsinceupdates,
    sumsinceregloss/numsinceupdates,
    sumsinceate/numsinceupdates,
    nicecount(2*numupdates*halfbatchsize),
    eta,
    net.blobs['balancelambda0'].data.flat[0])
nextprint=2*nextprint
numsinceupdates=0
sumsincetreatedloss=0
sumsinceuntreatedloss=0
sumsinceregloss=0

net.set_phase_train()

# kinda flass ...
#
# GLOG_minloglevel=5 eta=2e-3 lambda=0.05 PYTHONPATH=../../python python makeadvpsmodel.py nsw_advps_train nsw_control.txt psid_controls.txt advplacebomodel
# mean population diff: -1.64639
# delta t balance     max     reg         ate example    learning   balance
#            loss  weight    loss             counter        rate    lambda
#    23ms   9.505   0.003 5.5e-06     -1.7451     850 1.99998e-03 0.000e+00
#    31ms   9.665   0.003 5.5e-06     -1.6714      1K 1.99997e-03 5.000e-02
#    46ms   8.704   0.003 5.5e-06     -1.6488      3K 1.99994e-03 5.000e-02
#    77ms   8.560   0.003 5.5e-06     -1.7360      6K 1.99987e-03 5.000e-02
#   150ms   7.924   0.003 5.5e-06     -1.6263     13K 1.99974e-03 5.000e-02
#   278ms   7.405   0.003 5.6e-06     -1.6298     27K 1.99949e-03 5.000e-02
#   543ms   6.977   0.003 5.6e-06     -1.6354     54K 1.99898e-03 5.000e-02
#  1.150s   6.026   0.004 5.7e-06     -1.5389    108K 1.99795e-03 5.000e-02
#  2.159s   4.060   0.007 6.6e-06     -1.3695    217K 1.99591e-03 5.000e-02
#  4.213s   1.015   0.029 1.6e-05     -0.9462    435K 1.99182e-03 5.000e-02
#  8.240s   0.380   0.050 2.4e-05     -0.7221    870K 1.98368e-03 5.000e-02
# 16.372s   0.217   0.061 3.0e-05     -0.6128      1M 1.96750e-03 5.000e-02
# 32.720s   0.148   0.064 3.2e-05     -0.5515      3M 1.93553e-03 5.000e-02
#  1.086m   0.107   0.061 3.2e-05     -0.5191      6M 1.87313e-03 5.000e-02
#  2.167m   0.079   0.053 3.3e-05     -0.5546     13M 1.75431e-03 5.000e-02
#  4.331m   0.043   0.091 4.7e-05     -0.6764     27M 1.53880e-03 5.000e-02
#  8.651m   0.030   0.108 5.8e-05     -0.7630     55M 1.18395e-03 5.000e-02

