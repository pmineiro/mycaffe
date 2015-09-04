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
maxlambda=float(os.environ['lambda'])
lambdadecay=0.999992

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
sumate=0
sumsinceate=0
nextprint=1

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0)

print "mean population diff: %g"%(np.mean(ynsw,dtype='d')-np.mean(ypsid,dtype='d'))
print "%7s %7s %7s %11s %7s %11s %9s"%("delta t","balance","log","ate","example","learning","balance")
print "%7s %7s %7s %11s %7s %11s %9s"%("","loss","loss","","counter","rate","lambda")

maxminlambda=maxlambda
for ii in range(maxiter):
    psidperm=np.random.permutation(xpsid.shape[0])
    data=np.vstack((xnsw,
                    xpsid[psidperm[0:halfbatchsize],:]))
    labels=np.hstack((ynsw,
                      ypsid[psidperm[0:halfbatchsize]]))
    net.set_input_arrays(data,labels)
    res=net.forward()
    
    ate = (
      ( np.sum(np.multiply(ynsw,net.blobs['inverse_treated_score'].data[:,0,0,0]),
               dtype='d') /
        np.sum(net.blobs['inverse_treated_score'].data[:,0,0,0],dtype='d') ) -
      ( np.sum(np.multiply(ypsid[psidperm[0:halfbatchsize]],
                           net.blobs['inverse_untreated_score'].data[:,0,0,0]),
               dtype='d') /
        np.sum(net.blobs['inverse_untreated_score'].data[:,0,0,0],dtype='d') )
    )
    
    sumbalanceloss+=res['balance_loss']
    sumsincebalanceloss+=res['balance_loss']
    sumtreatmentloss+=res['treatment_loss']
    sumsincetreatmentloss+=res['treatment_loss']
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

        print "%7s %7.3f %7.3f %11.4f %7s %11.5e %9.3e"%(
            nicetime(dt),
            sumsincebalanceloss/numsinceupdates,
            sumsincetreatmentloss/numsinceupdates,
            sumsinceate/numsinceupdates,
            nicecount(2*numupdates*halfbatchsize),
            eta,
            maxlambda-maxminlambda)
        nextprint=2*nextprint
        numsinceupdates=0
        sumsincetreatedloss=0
        sumsinceuntreatedloss=0
        sumsincetreatmentloss=0
        sumsinceate=0

now=time.time()
dt=float(now-start)

print "%7s %7.3f %7.3f %11.4f %7s %11.5e %9.3e"%(
    nicetime(dt),
    sumsincebalanceloss/numsinceupdates,
    sumsincetreatmentloss/numsinceupdates,
    sumsinceate/numsinceupdates,
    nicecount(2*numupdates*halfbatchsize),
    eta,
    maxlambda-maxminlambda)
nextprint=2*nextprint
numsinceupdates=0
sumsincetreatedloss=0
sumsinceuntreatedloss=0
sumsincetreatmentloss=0

net.set_phase_train()

# GLOG_minloglevel=5 lambda=0 PYTHONPATH=../../python python makeadvpsmodel.py nsw_advps_train nsw_control.txt psid_controls.txt advplacebomodel
# delta t balance     log         ate example    learning   balance
#            loss    loss             counter        rate    lambda
#    47ms  17.883   0.719  -1390.5475     850 9.99992e-04 0.000e+00
#    64ms  36.117   0.718  -1354.4636      1K 9.99984e-04 0.000e+00
#    87ms  36.235   0.718  -1415.5894      3K 9.99968e-04 0.000e+00
#   145ms  36.582   0.715  -1204.0388      6K 9.99936e-04 0.000e+00
#   249ms  35.365   0.706  -1327.2469     13K 9.99872e-04 0.000e+00
#   398ms  33.785   0.682  -1346.2931     27K 9.99744e-04 0.000e+00
#   806ms  30.164   0.637  -1318.7506     54K 9.99488e-04 0.000e+00
#  1.392s  25.070   0.560  -1214.8328    108K 9.98977e-04 0.000e+00
#  2.743s  18.539   0.427  -1020.0171    217K 9.97954e-04 0.000e+00
#  5.521s  12.396   0.289   -848.5716    435K 9.95912e-04 0.000e+00
# 10.158s   7.751   0.229   -829.6803    870K 9.91841e-04 0.000e+00
# 19.329s   4.771   0.210   -669.4616      1M 9.83749e-04 0.000e+00
# 37.553s   3.206   0.200   -654.1963      3M 9.67763e-04 0.000e+00
#  1.228m   2.483   0.191   -792.8849      6M 9.36565e-04 0.000e+00
#  2.421m   2.365   0.180   -770.2076     13M 8.77154e-04 0.000e+00
#  5.139m   2.809   0.170   -777.2445     27M 7.69399e-04 0.000e+00
#  9.967m   3.292   0.161   -927.7099     55M 5.91975e-04 0.000e+00
# 19.400m   3.783   0.153   -871.9993    111M 3.50435e-04 0.000e+00
# 29.515m   6.085   0.149   -813.8779    170M 2.01895e-04 0.000e+00

# GLOG_minloglevel=5 lambda=0.001 PYTHONPATH=../../python python makeadvpsmodel.py nsw_advps_train nsw_control.txt psid_controls.txt advplacebomodel
# delta t balance     log         ate example    learning   balance
#            loss    loss             counter        rate    lambda
#    86ms  17.883   0.719  -1390.5475     850 9.99992e-04 8.000e-09
#    97ms  36.117   0.718  -1354.4636      1K 9.99984e-04 1.600e-08
#   124ms  36.235   0.718  -1415.5894      3K 9.99968e-04 3.200e-08
#   177ms  36.582   0.715  -1204.0389      6K 9.99936e-04 6.400e-08
#   270ms  35.365   0.706  -1327.2472     13K 9.99872e-04 1.280e-07
#   408ms  33.785   0.682  -1346.2945     27K 9.99744e-04 2.560e-07
#   690ms  30.164   0.637  -1318.7551     54K 9.99488e-04 5.119e-07
#  1.230s  25.070   0.560  -1214.8415    108K 9.98977e-04 1.023e-06
#  2.437s  18.539   0.427  -1020.0281    217K 9.97954e-04 2.046e-06
#  4.690s  12.395   0.289   -848.5993    435K 9.95912e-04 4.088e-06
#  9.036s   7.750   0.229   -829.7682    870K 9.91841e-04 8.159e-06
# 17.899s   4.769   0.210   -669.5432      1M 9.83749e-04 1.625e-05
# 35.654s   3.203   0.200   -654.7145      3M 9.67763e-04 3.224e-05
#  1.174m   2.477   0.191   -793.4364      6M 9.36565e-04 6.343e-05
#  2.331m   2.309   0.180   -770.6687     13M 8.77154e-04 1.228e-04
#  4.731m   2.339   0.170   -776.5212     27M 7.69399e-04 2.306e-04
#  9.398m   2.327   0.161   -924.1202     55M 5.91975e-04 4.080e-04
# 18.701m   2.339   0.153   -860.8924    111M 3.50435e-04 6.496e-04
# 28.504m   3.398   0.149   -801.6915    170M 2.01895e-04 7.981e-04
#  
# GLOG_minloglevel=5 lambda=0.01 PYTHONPATH=../../python python makeadvpsmodel.py nsw_advps_train nsw_control.txt psid_controls.txt advplacebomodel
# delta t balance     log         ate example    learning   balance
#            loss    loss             counter        rate    lambda
#    26ms  17.883   0.719  -1390.5475     850 9.99992e-04 8.000e-08
#    40ms  36.117   0.718  -1354.4636      1K 9.99984e-04 1.600e-07
#    59ms  36.235   0.718  -1415.5894      3K 9.99968e-04 3.200e-07
#   103ms  36.582   0.715  -1204.0393      6K 9.99936e-04 6.400e-07
#   187ms  35.365   0.706  -1327.2498     13K 9.99872e-04 1.280e-06
#   326ms  33.785   0.682  -1346.3066     27K 9.99744e-04 2.560e-06
#   633ms  30.164   0.637  -1318.7957     54K 9.99488e-04 5.119e-06
#  1.201s  25.068   0.560  -1214.9210    108K 9.98977e-04 1.023e-05
#  2.378s  18.536   0.427  -1020.1257    217K 9.97954e-04 2.046e-05
#  4.680s  12.391   0.289   -848.8452    435K 9.95912e-04 4.088e-05
#  9.158s   7.741   0.229   -830.5513    870K 9.91841e-04 8.159e-05
# 18.121s   4.756   0.210   -670.1940      1M 9.83749e-04 1.625e-04
# 36.424s   3.182   0.200   -658.4283      3M 9.67763e-04 3.224e-04
#  1.209m   2.434   0.191   -798.6767      6M 9.36565e-04 6.343e-04
#  2.407m   2.123   0.180   -777.3292     13M 8.77154e-04 1.228e-03
#  4.772m   1.935   0.170   -770.3477     27M 7.69399e-04 2.306e-03
#  9.616m   1.822   0.161   -888.3036     55M 5.91975e-04 4.080e-03
# 20.091m   1.736   0.155   -818.7628    111M 3.50435e-04 6.496e-03
# 30.013m   2.459   0.151   -801.4417    170M 2.01895e-04 7.981e-03

# GLOG_minloglevel=5 lambda=0.1 PYTHONPATH=../../python python makeadvpsmodel.py nsw_advps_train nsw_control.txt psid_controls.txt advplacebomodel
# delta t balance     log         ate example    learning   balance
#            loss    loss             counter        rate    lambda
#    42ms  17.883   0.719  -1390.5475     850 9.99992e-04 8.000e-07
#    57ms  36.117   0.718  -1354.4636      1K 9.99984e-04 1.600e-06
#    81ms  36.235   0.718  -1415.5900      3K 9.99968e-04 3.200e-06
#   136ms  36.582   0.715  -1204.0434      6K 9.99936e-04 6.400e-06
#   217ms  35.364   0.706  -1327.2765     13K 9.99872e-04 1.280e-05
#   371ms  33.783   0.682  -1346.4312     27K 9.99744e-04 2.560e-05
#   670ms  30.157   0.637  -1319.1979     54K 9.99488e-04 5.119e-05
#  1.272s  25.052   0.559  -1215.7077    108K 9.98977e-04 1.023e-04
#  2.541s  18.508   0.427  -1021.0980    217K 9.97954e-04 2.046e-04
#  4.809s  12.347   0.289   -851.3000    435K 9.95912e-04 4.088e-04
#  9.302s   7.654   0.228   -837.9874    870K 9.91841e-04 8.159e-04
# 18.362s   4.646   0.210   -674.5502      1M 9.83749e-04 1.625e-03
# 36.726s   3.058   0.200   -681.4498      3M 9.67763e-04 3.224e-03
#  1.221m   2.260   0.191   -822.4619      6M 9.36565e-04 6.343e-03
#  2.423m   1.848   0.181   -803.3193     13M 8.77154e-04 1.228e-02
#  4.754m   1.627   0.172   -777.1697     27M 7.69399e-04 2.306e-02
#  9.398m   1.488   0.165   -848.7976     55M 5.91975e-04 4.080e-02
# 18.656m   1.346   0.161   -772.2984    111M 3.50435e-04 6.496e-02
# 28.566m   1.826   0.160   -812.1403    170M 2.01895e-04 7.981e-02

