import caffe
from caffe import layers as L
from caffe import params as P

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
