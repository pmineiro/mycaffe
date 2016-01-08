import caffe
from caffe import layers as L
from caffe import params as P

#-------------------------------------------------
# define model
#-------------------------------------------------

def net(batchsize,embedd,numpos,numdocs,numconvk,numip1,numip2):
  n = caffe.NetSpec()
  n.features, n.labels = L.MemoryData(batch_size=batchsize, channels=embedd, height=1, width=numpos, ntop=2)
  n.conv1 = L.Convolution(n.features, num_output=numconvk, kernel_h=1, kernel_w=3)
  n.pool1 = L.Pooling(n.conv1, kernel_h=1, kernel_w=numpos, stride=numpos, pool=P.Pooling.MAX)
#  n.ip1 = L.InnerProduct(n.pool1, num_output=numip1)
#  n.relu1 = L.ReLU(n.ip1, in_place=True)
#  n.ip2 = L.InnerProduct(n.ip1, num_output=numip2)
#  n.relu2 = L.ReLU(n.ip2, in_place=True)
#  n.lastip = L.InnerProduct(n.relu2, num_output=numdocs)
  n.lastip = L.InnerProduct(n.pool1, num_output=numdocs)
  n.loss = L.SoftmaxWithLoss(n.lastip, n.labels)
  n.acc = L.Accuracy(n.lastip, n.labels, loss_weight=0, accuracy_param={ 'top_k':10 })
  return n.to_proto()
