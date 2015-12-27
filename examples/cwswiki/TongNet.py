import caffe
from caffe import layers as L
from caffe import params as P

#-------------------------------------------------
# define model
#-------------------------------------------------

def net(batchsize,embedd,numpos,numdocs,numconvk,numip1,numip2):
  n = caffe.NetSpec()
  n.data, n.bogus = L.MemoryData(batch_size=batchsize, channels=(numpos*embedd+numdocs), height=1, width=1, ntop=2)
  n.prefeatures, n.labels = L.Slice(n.data, slice_param=dict(axis=1,slice_point=[numpos*embedd]),ntop=2)
  n.features = L.Reshape(n.prefeatures, reshape_param=dict(shape=dict(dim=[0,embedd,1,numpos])))
  n.conv1 = L.Convolution(n.features, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool1 = L.Pooling(n.conv1, kernel_h=1, kernel_w=2, stride=1, pool=P.Pooling.MAX)
  n.ip1 = L.InnerProduct(n.pool1, num_output=numip1)
  n.relu1 = L.ReLU(n.ip1, in_place=True)
  n.ip2 = L.InnerProduct(n.ip1, num_output=numip2)
  n.relu2 = L.ReLU(n.ip2, in_place=True)
  n.lastip = L.InnerProduct(n.relu2, num_output=numdocs)
  n.loss = L.SigmoidCrossEntropyLoss(n.lastip, n.labels)
  n.silence = L.Silence(n.bogus,ntop=0)
  return n.to_proto()
