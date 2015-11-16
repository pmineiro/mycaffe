import bz2
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import math
import numpy as np
import os
import random
import sys
import time
from scipy.sparse import csr_matrix

import pdb

maxtags=13039
numtags=100
numtokens=260544
windowsize=10
embedd=300
batchsize=5001
numconvk=100

alpha=0.0
eta=1e-3
etadecay=0.9999
weightdecay=1e-6
labelnoise=1e-6

lrs=dict()
lrs['embedding']=1
lrs[('conv1',0)]=1
lrs[('conv1',1)]=2
lrs[('conv2',0)]=1
lrs[('conv2',1)]=2
lrs[('conv3',0)]=1
lrs[('conv3',1)]=2
lrs[('ip1',0)]=1
lrs[('ip1',1)]=2
lrs[('ip2',0)]=1
lrs[('ip2',1)]=2
lrs[('lastip',0)]=1
lrs[('lastip',1)]=0

random.seed(69)
np.random.seed(8675309)

#-------------------------------------------------
# beautification
#-------------------------------------------------

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
  if (n < 10*1000): 
    return "%4u"%(n) 
  elif (n < 10*1000*1000): 
    return "%4uK"%(n/1000) 
  elif (n < 10*1000*1000*1000): 
    return "%4uM"%(n/(1000*1000)) 
  else: 
    return "%4uB"%(n/(1000*1000*1000)) 

def nicediv(a,b):
  if (b == 0):
    return "%7s"%("n/a")
  else:
    return "%7.3f"%(a/b)

#-------------------------------------------------
# define model
#-------------------------------------------------

def net(batchsize,embedd,windowsize,numtags,numconvk):
  n = caffe.NetSpec()
  n.data, n.bogus = L.MemoryData(batch_size=batchsize, channels=(windowsize*embedd+numtags), height=1, width=1, ntop=2)
  n.prefeatures, n.labels = L.Slice(n.data, slice_param=dict(axis=1,slice_point=[windowsize*embedd]),ntop=2)
  n.features = L.Reshape(n.prefeatures, reshape_param=dict(shape=dict(dim=[0,embedd,1,windowsize])))
  n.conv1 = L.Convolution(n.features, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool1 = L.Pooling(n.conv1, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.conv2 = L.Convolution(n.pool1, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool2 = L.Pooling(n.conv2, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.conv3 = L.Convolution(n.pool2, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool3 = L.Pooling(n.conv3, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.ip1 = L.InnerProduct(n.pool3, num_output=200)
  n.relu1 = L.ReLU(n.ip1, in_place=True)
  n.ip2 = L.InnerProduct(n.ip1, num_output=200)
  n.relu2 = L.ReLU(n.ip2, in_place=True)
  n.lastip = L.InnerProduct(n.relu2, num_output=numtags)
  n.loss = L.SigmoidCrossEntropyLoss(n.lastip, n.labels)
  n.silence = L.Silence(n.bogus,ntop=0)
  return n.to_proto()

protofilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.prototxt')
netfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.model')
embeddingfilename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrain.embedding.h5f')

with open(protofilename,'w') as f:
  f.write("force_backward: true\n");
  f.write(str(net(batchsize,embedd,windowsize,numtags,numconvk)))

caffe.set_mode_gpu()
solver = caffe.SGDSolver('pretrain_solver.prototxt')
if (alpha > 0):
  momsolver = caffe.SGDSolver('pretrain_solver.prototxt')

#-------------------------------------------------
# initialize
#-------------------------------------------------

for layer in solver.net.layers:
  blobnum=0 
  for blob in layer.blobs:
    if blobnum == 0:
      blob.data[:]=(1.0/math.sqrt(np.prod(blob.data.shape[1:])))*np.random.standard_normal(size=blob.data.shape)
    else:
      blob.data[:]=0
    blobnum=blobnum+1 

if (alpha > 0):
  for layer in momsolver.net.layers:
    for blob in layer.blobs:
      blob.data[:]=0

# prior for bias

labelcounts=np.zeros(maxtags,dtype='d')
examplecount=0

try:
  with open('pretrain.labelcounts.txt', 'r') as f:
    print "using precomputed label counts"
    for line in f:
      lc=[word for word in line.split('\t')]
      labelcounts[int(lc[0])]=float(lc[1])
except:
  print "computing label counts"
  with bz2.BZ2File('pretrain.bz2', 'rb') as inputfile:
    for line in inputfile:
      yx=[word for word in line.split('\t')]
  
      tokens=yx[1].split(' ')
  
      if (len(tokens) < windowsize):
        continue
  
      for l in yx[0].split(','):
        labelcounts[int(l)-1]+=1
  
      examplecount+=1

  labelcounts=(1+labelcounts)/examplecount

  print "saving label counts"
  with open('pretrain.labelcounts.txt', 'w') as f:
    for ii in range(maxtags):
      f.write('%u\t%g\n'%(ii,labelcounts[ii]))

for (name,layer) in zip(solver.net._layer_names,solver.net.layers):
  blobnum=0 
  for blob in layer.blobs:
    if name == "lastip" and blobnum == 1:
      blob.data[:]=np.log(np.divide(labelcounts[0:numtags],1-labelcounts[0:numtags]))
    blobnum=blobnum+1 

embedding=(1.0/math.sqrt(embedd))*np.random.standard_normal(size=(embedd,numtokens+1)).astype(float)
if (alpha > 0):
  momembeddiff=np.zeros(shape=(embedd,numtokens+1),dtype='f')

#-------------------------------------------------
# iterate
#-------------------------------------------------

sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr=os.fdopen(sys.stderr.fileno(), 'w', 0) 
 
print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("delta t","average","since","holdout","example","pass","learning") 
print "%7s\t%7s\t%7s\t%7s\t%7s\t%4s\t%9s"%("","loss","last","loss","counter","num","rate") 

bindex=0
start=time.time()
numsinceupdates=0 
numupdates=0 
sumloss=0 
sumsinceloss=0 
nextprint=1 

comboinputs=np.zeros((batchsize,windowsize*embedd+numtags,1,1),dtype='f')
bogus=np.zeros((batchsize,1,1,1),dtype='f')
batchtokens=np.zeros((batchsize,windowsize),dtype='i')
  
for passes in range(64):
  batchnum=0
  sumholdoutloss=0
  numholdoutupdates=0
  with bz2.BZ2File('pretrain.bz2', 'rb') as inputfile:
    for line in inputfile:
      yx=[word for word in line.split('\t')]
  
      tokens=yx[1].split(' ')
  
      if (len(tokens) < windowsize):
        continue
  
      tokstart=random.randint(0,len(tokens)-windowsize)
  
      for ii in range(windowsize):
        batchtokens[bindex,ii]=int(tokens[tokstart+ii])
        comboinputs[bindex,ii*embedd:(ii+1)*embedd,0,0]=embedding[:,int(tokens[tokstart+ii])]
  
      comboinputs[bindex,(windowsize*embedd):,0,0]=labelnoise
      for l in yx[0].split(','):
        if int(l) <= numtags:
          comboinputs[bindex,(windowsize*embedd)+int(l)-1,0,0]=1.0-labelnoise
  
      bindex+=1
  
      if bindex >= batchsize:
        solver.net.set_input_arrays(comboinputs,bogus)
        res=solver.net.forward()
        sumloss+=res['loss']
        sumsinceloss+=res['loss']

        batchnum+=1

        if passes > 0 and batchnum % 16 == 0:
          sumholdoutloss+=res['loss']
          numholdoutupdates+=1
        else:
          solver.net.backward()
          data_diff=solver.net.blobs['data'].diff.reshape(batchsize,windowsize*embedd+numtags,1,1)

          # TODO: scale learning rate by token frequency ...
  
          myeta=lrs['embedding']*eta;
          if (alpha > 0):
            momembeddiff*=alpha
            for ii in range(batchsize):
              for jj in range(windowsize):
                momembeddiff[:,batchtokens[ii,jj]]+=myeta*data_diff[ii,jj*embedd:(jj+1)*embedd,0,0]
            embedding-=momembeddiff
          else:
            for ii in range(batchsize):
              for jj in range(windowsize):
                embedding[:,batchtokens[ii,jj]]-=myeta*data_diff[ii,jj*embedd:(jj+1)*embedd,0,0]

          embedding*=(1-weightdecay*myeta)
  
          if (alpha > 0):
            for (name,layer,momlayer) in zip(solver.net._layer_names,solver.net.layers,momsolver.net.layers):
               blobnum=0 
               for (blob,momblob) in zip(layer.blobs,momlayer.blobs): 
                 myeta=lrs[(name,blobnum)]*eta 
                 momblob.data[:]*=alpha
                 momblob.data[:]+=myeta*blob.diff
                 blob.data[:]-=momblob.data
                 blob.data[:]*=(1-weightdecay*myeta)
                 blobnum=blobnum+1 
          else:
            for (name,layer) in zip(solver.net._layer_names,solver.net.layers):
               blobnum=0 
               for blob in layer.blobs:
                 myeta=lrs[(name,blobnum)]*eta 
                 blob.data[:]-=myeta*blob.diff
                 blob.data[:]*=(1-weightdecay*myeta)
                 blobnum=blobnum+1 

          eta=eta*etadecay

        bindex=0
        numupdates+=1
        numsinceupdates+=1

        if numupdates >= nextprint:
          try:
            os.remove(netfilename+"."+str(numupdates)) 
          except:
            pass
            
          solver.net.save(netfilename+"."+str(numupdates)) 
          try:
            os.remove(embeddingfilename+"."+str(numupdates)) 
          except:
            pass
            
          h5f=h5py.File(embeddingfilename+"."+str(numupdates)) 
          h5f.create_dataset('embedding',data=embedding) 
          h5f.close() 
          now=time.time() 
          print "%7s\t%7.3f\t%7.3f\t%7s\t%7s\t%4u\t%9.3g"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicediv(sumholdoutloss,numholdoutupdates),nicecount(numupdates*batchsize),passes,eta) 
          nextprint=2*nextprint 
          numsinceupdates=0 
          sumsinceloss=0 


solver.net.save(netfilename)
try:
  os.remove(embeddingfilename)
except:
  pass
h5f=h5py.File(embeddingfilename);
h5f.create_dataset('embedding',data=embedding) 
h5f.close() 
now=time.time() 
print "%7s\t%7.3f\t%7.3f\t%7s\t%7s\t%4u\t%9.3g"%(nicetime(float(now-start)),sumloss/numupdates,sumsinceloss/numsinceupdates,nicediv(sumholdoutloss,numholdoutupdates),nicecount(numupdates*batchsize),passes,eta) 

# GLOG_minloglevel=5 PYTHONPATH=../../python python ./pretrain.py
# using precomputed label counts
# delta t average   since holdout example pass     learning
#            loss    last    loss counter  num         rate
#  1.979s   2.336   2.336     n/a    5001    0        0.001
#  4.078s   2.348   2.360     n/a     10K    0        0.001
#  7.262s   2.320   2.292     n/a     20K    0        0.001
# 12.931s   2.304   2.289     n/a     40K    0     0.000999
# 24.122s   2.283   2.261     n/a     80K    0     0.000998
# 45.461s   2.274   2.265     n/a    160K    0     0.000997
#  1.475m   2.268   2.263     n/a    320K    0     0.000994
#  2.829m   2.270   2.272     n/a    640K    0     0.000987
#  5.476m   2.271   2.272     n/a   1280K    0     0.000975
# 10.699m   2.271   2.272   2.271   2560K    1     0.000951
# 20.880m   2.269   2.266   2.246   5121K    2     0.000906
# 41.047m   2.253   2.237   2.218     10M    4     0.000823
#  1.370h   2.221   2.190   2.156     20M    9     0.000679
#  2.714h   2.176   2.131   2.101     40M   19     0.000462
#  5.414h   2.132   2.088   2.086     81M   38     0.000214
#  8.992h   2.106   2.067   2.071    136M   63     7.74e-05     

# numconv 100 -> 200
#GLOG_minloglevel=5 PYTHONPATH=../../python python ./pretrain.py
#using precomputed label counts
#delta t average   since holdout example pass       learning
#           loss    last    loss counter  num           rate
# 1.840s   2.327   2.327     n/a    3001    0      0.0009999
# 3.590s   2.344   2.360     n/a    6002    0      0.0009998
# 6.324s   2.350   2.356     n/a     12K    0      0.0009996
#10.914s   2.312   2.274     n/a     24K    0      0.0009992
#19.310s   2.296   2.280     n/a     48K    0      0.0009984
#35.132s   2.280   2.264     n/a     96K    0      0.0009968
# 1.101m   2.273   2.266     n/a    192K    0      0.0009936
# 2.116m   2.271   2.269     n/a    384K    0      0.0009873
# 4.132m   2.272   2.273     n/a    768K    0      0.0009747
# 8.150m   2.270   2.269     n/a   1536K    0      0.0009501
#16.111m   2.269   2.268   2.267   3073K    1      0.0009044
#31.422m   2.257   2.245   2.234   6146K    2      0.0008216
# 1.036h   2.225   2.192   2.173     12M    5       0.000678
# 2.062h   2.172   2.120   2.094     24M   11      0.0004617
# 4.113h   2.122   2.072   2.030     49M   23      0.0002141
#...

# 1000 labels
# GLOG_minloglevel=5 PYTHONPATH=../../python python ./pretrain.py
# using precomputed label counts
# delta t average   since holdout example pass       learning
#            loss    last    loss counter  num           rate
#  1.820s   7.081   7.081     n/a    3001    0      0.0009999
#  3.494s   7.164   7.247     n/a    6002    0      0.0009998
#  6.145s   7.194   7.224     n/a     12K    0      0.0009996
# 10.407s   7.151   7.108     n/a     24K    0      0.0009992
# 18.248s   7.125   7.099     n/a     48K    0      0.0009984
# 32.253s   7.095   7.066     n/a     96K    0      0.0009968
#  1.014m   7.062   7.029     n/a    192K    0      0.0009936
#  1.911m   7.063   7.064     n/a    384K    0      0.0009873
#  3.654m   7.064   7.065     n/a    768K    0      0.0009747
#  7.128m   7.060   7.055     n/a   1536K    0      0.0009501
# 13.926m   7.062   7.064   7.097   3073K    1      0.0009044
# 27.508m   7.048   7.033   7.052   6146K    2      0.0008216
# 54.369m   6.990   6.932   6.958     12M    5       0.000678
#  1.790h   6.890   6.789   6.741     24M   11      0.0004617
#  3.562h   6.784   6.678   6.564     49M   23      0.0002141
#  7.072h   6.755   6.725   6.787     98M   46      4.605e-05
# ...

# 10000 labels
# GLOG_minloglevel=5 PYTHONPATH=../../python python ./pretrain.py                  using precomputed label counts
# delta t average   since holdout example pass       learning
#            loss    last    loss counter  num           rate
#  1.688s  21.368  21.368     n/a    3001    0      0.0009999
#  3.301s  21.646  21.924     n/a    6002    0      0.0009998
#  5.951s  21.759  21.872     n/a     12K    0      0.0009996
# 10.459s  21.624  21.489     n/a     24K    0      0.0009992
# 18.728s  21.536  21.449     n/a     48K    0      0.0009984
# 35.109s  21.596  21.656     n/a     96K    0      0.0009968
#  1.115m  21.512  21.428     n/a    192K    0      0.0009936
#  2.127m  21.506  21.500     n/a    384K    0      0.0009873
#  4.156m  21.479  21.453     n/a    768K    0      0.0009747
#  8.302m  21.460  21.440     n/a   1536K    0      0.0009501
# 16.093m  21.456  21.452  21.521   3073K    1      0.0009044
# 31.490m  21.444  21.433  21.427   6146K    2      0.0008216
#  1.035h  21.389  21.333  21.311     12M    5       0.000678
#  2.063h  21.193  20.997  20.842     24M   11      0.0004617
#  4.124h  20.938  20.683  20.484     49M   23      0.0002141
# ...

# all labels
# GLOG_minloglevel=5 PYTHONPATH=../../python python ./pretrain.py
# computing label counts
# saving label counts
# delta t average   since holdout example pass     learning
#            loss    last    loss counter  num         rate
#  1.859s  24.048  24.048     n/a    3001    0        0.001
#  3.613s  24.341  24.633     n/a    6002    0        0.001
#  6.453s  24.508  24.676     n/a     12K    0        0.001
# 11.246s  24.350  24.192     n/a     24K    0     0.000999
# 20.248s  24.247  24.143     n/a     48K    0     0.000998
# 37.351s  24.313  24.379     n/a     96K    0     0.000997
#  1.182m  24.223  24.133     n/a    192K    0     0.000994
#  2.281m  24.213  24.204     n/a    384K    0     0.000987
#  4.492m  24.191  24.169     n/a    768K    0     0.000975
#  8.893m  24.173  24.154     n/a   1536K    0      0.00095
# 17.346m  24.165  24.157  24.246   3073K    1     0.000904
# 33.992m  24.149  24.134  24.135   6146K    2     0.000822
#  1.123h  24.082  24.015  24.007     12M    5     0.000678
#  2.232h  23.884  23.685  23.568     24M   11     0.000462
#  4.431h  23.648  23.412  23.292     49M   23     0.000214
#  8.821h  23.854  24.061  24.075     98M   46      4.6e-05
# 12.210h  23.938  24.156  24.089    136M   63     1.41e-05
