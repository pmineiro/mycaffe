import caffe
from caffe import layers as L
from caffe import params as P

#-------------------------------------------------
# define model
#-------------------------------------------------

#-------------------------------------------------
# full bilinear model
# 
# f (q, a) = q^\top W a, q \in \R^d, a \in \R^c, W \in \R^{d \times c}
#          = (W^\top q)^\top a
#-------------------------------------------------

def replicate(nratings, layer, orig):
  if nratings == 1:
    return layer
  else:
    return L.Concat(replicate (nratings - 1, layer, orig), 
                    orig,
                    concat_param=dict(axis=1))

def full(batchsize, nqueries, nads, nratings):
  n = caffe.NetSpec()
  n.data, n.label = L.MemoryData(batch_size=batchsize,
                                 channels=1,
                                 height=1,
                                 width=(nqueries + nads),
                                 ntop=2)
  n.queries, n.ads = L.Slice(n.data,
                             slice_param=dict(axis=3,slice_point=[nqueries]),
                             ntop=2)
  n.Wtopq = L.InnerProduct(n.queries, num_output=nratings*nads)
  n.Wtopqreshape = L.Reshape(n.Wtopq, 
                             reshape_param=dict(shape=dict(dim=[batchsize,nratings,1,nads])))
  n.bigads = replicate (nratings, n.ads, n.ads)
  n.predot = L.Eltwise(n.Wtopqreshape, n.bigads,
                       eltwise_param=dict(operation=0) # 0 = PROD
                       )
  # use identity convolution to sum
  n.dot = L.Convolution(n.predot,
                        convolution_param=dict(num_output=nratings,
                                               bias_term=False,
                                               kernel_h=1,
                                               kernel_w=nads,
                                               group=nratings))
  n.dotreshape = L.Reshape(n.dot,
                           reshape_param=dict(shape=dict(dim=[batchsize,nratings])))
  n.linearterms = L.InnerProduct(n.data, num_output=nratings)
  n.scores = L.Eltwise(n.dotreshape, n.linearterms, 
                       eltwise_param=dict(operation=1) # 1 = SUM
		       )
  n.loss = L.SoftmaxWithLoss(n.scores, n.label)
  n.acc = L.Accuracy(n.scores, n.label, loss_weight=0)
  return n.to_proto()

#-------------------------------------------------
# factorization machine
# 
# f (q, a) = \sum_k (q^\top u_k) (v_k^\top a), u_k \in R^d, v_k \in \R^c
#-------------------------------------------------
