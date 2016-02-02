import caffe
from caffe import layers as L
from caffe import params as P

#-------------------------------------------------
# define model
#-------------------------------------------------

#-------------------------------------------------
# linear model (i.e., user/movie and global bias only)
#-------------------------------------------------

def linear(batchsize, nqueries, nads, nratings):
  n = caffe.NetSpec()
  n.data, n.label = L.MemoryData(batch_size=batchsize,
                                 channels=1,
                                 height=1,
                                 width=(nqueries + nads),
                                 ntop=2)
  n.scores = L.InnerProduct(n.data, num_output=nratings)
  n.loss = L.SoftmaxWithLoss(n.scores, n.label)
  n.acc = L.Accuracy(n.scores, n.label, loss_weight=0)
  return n.to_proto()

#-------------------------------------------------
# factorization machine (+ linear)
# 
# f (q, a) = \sum_k (q^\top u_k) (v_k^\top a), u_k \in R^d, v_k \in \R^c
#-------------------------------------------------

def fm(batchsize, nqueries, nads, nratings, rank):
  n = caffe.NetSpec()
  n.data, n.label = L.MemoryData(batch_size=batchsize,
                                 channels=1,
                                 height=1,
                                 width=(nqueries + nads),
                                 ntop=2)
  n.queries, n.ads = L.Slice(n.data,
                             slice_param=dict(axis=3,slice_point=[nqueries]),
                             ntop=2)

  n.Utopq = L.InnerProduct(n.queries, num_output=nratings*rank)
  n.Utopqreshape = L.Reshape(n.Utopq, 
                             reshape_param=dict(shape=dict(dim=[batchsize,nratings,1,rank])))

  n.Vtopa = L.InnerProduct(n.ads, num_output=nratings*rank)
  n.Vtopareshape = L.Reshape(n.Vtopa, 
                             reshape_param=dict(shape=dict(dim=[batchsize,nratings,1,rank])))

  n.predot = L.Eltwise(n.Utopqreshape, n.Vtopareshape,
                       eltwise_param=dict(operation=0) # 0 = PROD
                       )
  # use identity convolution to sum
  n.dot = L.Convolution(n.predot,
                        convolution_param=dict(num_output=nratings,
                                               bias_term=False,
                                               kernel_h=1,
                                               kernel_w=rank,
                                               group=nratings))
  n.dropdot = L.Dropout(n.dot)
  n.dotreshape = L.Reshape(n.dropdot,
                           reshape_param=dict(shape=dict(dim=[batchsize,nratings])))
  n.linearterms = L.InnerProduct(n.data, num_output=nratings)
  n.scores = L.Eltwise(n.dotreshape, n.linearterms, 
                       eltwise_param=dict(operation=1) # 1 = SUM
		       )
  n.loss = L.SoftmaxWithLoss(n.scores, n.label)
  n.acc = L.Accuracy(n.scores, n.label, loss_weight=0)
  return n.to_proto()

#-------------------------------------------------
# nonlinear factorization machine (+ linear)
# 
# f (q, a) = N (q^\top U, V^\top a), U \in R^{d \times k}, V \in \R^{c \times k}
#-------------------------------------------------

def relus(layer, sizes):
  if not sizes:
    return layer
  else:
    return L.Dropout (L.ReLU (L.InnerProduct (relus (layer, sizes[:-1]), 
					      num_output=sizes[-1]),
			      in_place=True))

def crossfm(batchsize, nqueries, nads, nratings, rank, numips):
  n = caffe.NetSpec()
  n.data, n.label = L.MemoryData(batch_size=batchsize,
                                 channels=1,
                                 height=1,
                                 width=(nqueries + nads),
                                 ntop=2)
  n.queries, n.ads = L.Slice(n.data,
                             slice_param=dict(axis=3,slice_point=[nqueries]),
                             ntop=2)

  n.Utopq = L.InnerProduct(n.queries, num_output=rank)
  n.Vtopa = L.InnerProduct(n.ads, num_output=rank)

  n.combo = L.Concat (n.Utopq, n.Vtopa)

  n.iplast = relus (n.combo, numips)

  n.crossscores = L.InnerProduct (n.iplast, num_output=nratings)
  n.linearterms = L.InnerProduct(n.data, num_output=nratings)
  n.scores = L.Eltwise(n.crossscores, n.linearterms, 
                       eltwise_param=dict(operation=1) # 1 = SUM
		       )
  n.loss = L.SoftmaxWithLoss(n.scores, n.label)
  n.acc = L.Accuracy(n.scores, n.label, loss_weight=0)
  return n.to_proto()

# #-------------------------------------------------
# # full bilinear model (+ linear), fundamentally can't generalize
# # 
# # f (q, a) = q^\top W a, q \in \R^d, a \in \R^c, W \in \R^{d \times c}
# #          = (W^\top q)^\top a
# #-------------------------------------------------
# 
# def replicate(nratings, layer, orig):
#   if nratings == 1:
#     return layer
#   else:
#     return L.Concat(replicate (nratings - 1, layer, orig), 
#                     orig,
#                     concat_param=dict(axis=1))
# 
# def full(batchsize, nqueries, nads, nratings):
#   n = caffe.NetSpec()
#   n.data, n.label = L.MemoryData(batch_size=batchsize,
#                                  channels=1,
#                                  height=1,
#                                  width=(nqueries + nads),
#                                  ntop=2)
#   n.queries, n.ads = L.Slice(n.data,
#                              slice_param=dict(axis=3,slice_point=[nqueries]),
#                              ntop=2)
#   n.Wtopq = L.InnerProduct(n.queries, num_output=nratings*nads)
#   n.Wtopqreshape = L.Reshape(n.Wtopq, 
#                              reshape_param=dict(shape=dict(dim=[batchsize,nratings,1,nads])))
#   n.bigads = replicate (nratings, n.ads, n.ads)
#   n.predot = L.Eltwise(n.Wtopqreshape, n.bigads,
#                        eltwise_param=dict(operation=0) # 0 = PROD
#                        )
#   # use identity convolution to sum
#   n.dot = L.Convolution(n.predot,
#                         convolution_param=dict(num_output=nratings,
#                                                bias_term=False,
#                                                kernel_h=1,
#                                                kernel_w=nads,
#                                                group=nratings))
#   n.dotreshape = L.Reshape(n.dot,
#                            reshape_param=dict(shape=dict(dim=[batchsize,nratings])))
#   n.linearterms = L.InnerProduct(n.data, num_output=nratings)
#   n.scores = L.Eltwise(n.dotreshape, n.linearterms, 
#                        eltwise_param=dict(operation=1) # 1 = SUM
# 		       )
#   n.loss = L.SoftmaxWithLoss(n.scores, n.label)
#   n.acc = L.Accuracy(n.scores, n.label, loss_weight=0)
#   return n.to_proto()

