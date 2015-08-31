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
weightdecay=1e-5
kappa=0.25
maxiter=200000
maxlambda=float(os.environ['lambda'])
lambdadecay=0.99999

# UGH ... so much for DRY

lrs=dict()
lrs[('ip0',0)]=40
lrs[('ip0',1)]=40
lrs[('ip1',0)]=30
lrs[('ip1',1)]=30
lrs[('ip2',0)]=20
lrs[('ip2',1)]=20
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
    blob.data[:]=0.01*np.random.standard_normal(size=blob.data.shape)
    momblob.data[:]=np.zeros(blob.data.shape,dtype='f')

net.params['treated_prediction'][1].data[:]=0
net.params['untreated_prediction'][1].data[:]=0
#net.params['treatment_prediction'][1].data[:]=0

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
    net.blobs['treated_lossdetail'].data[:,0,0,0]=np.fmin(10,np.divide(0.5,net.blobs['treatment_probability'].data[0:halfbatchsize,1,0,0]))
    net.blobs['untreated_lossdetail'].data[:,0,0,0]=np.fmin(10,np.divide(0.5,net.blobs['treatment_probability'].data[-halfbatchsize:,0,0,0]))
    impweight=np.mean(np.vstack((net.blobs['treated_lossdetail'].data,
                                 net.blobs['untreated_lossdetail'].data)),
                      dtype='d')
    normweight=impweight
    net.blobs['treated_lossdetail'].data[:,0,0,0]=np.divide(net.blobs['treated_lossdetail'].data[:,0,0,0],impweight)
    net.blobs['untreated_lossdetail'].data[:,0,0,0]=np.divide(net.blobs['untreated_lossdetail'].data[:,0,0,0],impweight)
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

# GLOG_minloglevel=5 PYTHONPATH=../../python python makenswmodel.py nsw_train nsw_control.txt psid_controls.txt nswplacebomodel   
# mean population diff: -1.64639
# delta t outcome outcome treated    ate example   learning  gradient
#          loss T  loss U    loss        counter       rate     scale
#   108ms   0.586   6.997   0.693 -0.002     850       0.05 0.000e+00
#   158ms   0.586   6.408   0.693 -0.005      1K       0.05 0.000e+00
#   204ms   0.585   6.967   0.693 -0.016      3K       0.05 0.000e+00
#   282ms   0.574   6.316   0.693 -0.222      6K       0.05 0.000e+00
#   427ms   0.415   4.095   0.693 -1.477     13K   0.049999 0.000e+00
#   695ms   0.349   2.828   0.693 -1.356     27K   0.049998 0.000e+00
#  1.243s   0.329   2.281   0.693 -0.486     54K   0.049997 0.000e+00
#  2.472s   0.312   1.338   0.545 -0.454    108K   0.049994 0.000e+00
#  4.906s   0.298   1.096   0.420 -0.198    217K   0.049987 0.000e+00
# 10.089s   0.289   1.083   0.417 -0.380    435K   0.049974 0.000e+00
# 20.263s   0.267   1.015   0.393 -0.338    870K   0.049949 0.000e+00
# 38.714s   0.220   0.889   0.300 -0.144      1M   0.049898 0.000e+00
#  1.274m   0.149   0.675   0.178 -0.268      3M   0.049796 0.000e+00
#  2.527m   0.099   0.452   0.072 -0.364      6M   0.049592 0.000e+00
#  5.071m   0.081   0.301   0.039 -0.467     13M   0.049187 0.000e+00
# 10.282m   0.064   0.215   0.026 -0.423     27M   0.048388 0.000e+00
# 21.199m   0.050   0.164   0.024 -0.476     55M   0.046828 0.000e+00
# 43.855m   0.044   0.131   0.025 -0.285    111M   0.043858 0.000e+00
#  1.530h   0.040   0.098   0.027 -0.539    222M    0.03847 0.000e+00

#mean population diff: -1.64639
#delta t outcome outcome treated ateott example   learning  gradient
#         loss T  loss U    loss        counter       rate     scale
#  126ms   0.586   6.997   0.693 -0.000     850  0.0099999 2.000e-06
#  152ms   0.586   6.414   0.693 -0.001      1K  0.0099998 4.000e-06
#  194ms   0.586   6.993   0.693 -0.003      3K  0.0099997 8.000e-06
#  268ms   0.585   6.568   0.693 -0.009      6K  0.0099994 1.600e-05
#  415ms   0.581   7.085   0.693 -0.070     13K  0.0099987 3.200e-05
#  703ms   0.455   4.596   0.693 -2.161     27K  0.0099974 6.399e-05
# 1.270s   0.337   2.647   0.693 -1.762     54K  0.0099949 1.280e-04
# 2.300s   0.326   2.385   0.693 -1.626    108K  0.0099898 2.558e-04
# 5.241s   0.306   1.223   0.624 -0.367    217K  0.0099795 5.113e-04
# 9.830s   0.298   1.091   0.418 -0.376    435K  0.0099591 1.021e-03
#19.508s   0.295   1.060   0.404 -0.364    870K  0.0099184 2.038e-03
#38.044s   0.289   1.019   0.403 -0.313      1M  0.0098375 4.054e-03
# 1.264m   0.269   0.953   0.414 -0.214      3M  0.0096776 8.027e-03
# 2.561m   0.203   0.796   0.449 -0.414      6M  0.0093657 1.573e-02
# 5.062m   0.131   0.533   0.258 -0.248     13M  0.0087715 3.022e-02
#10.415m   0.083   0.316   0.131 -0.172     27M   0.007694 5.588e-02
#21.543m   0.061   0.182   0.175 -0.475     55M  0.0059198 9.615e-02
#44.368m   0.051   0.112   0.252 -0.641    111M  0.0035043 1.461e-01
# 1.501h   0.045   0.074   0.343 -0.669    222M   0.001228 1.855e-01

