import numpy as np

# TODO: do the projection stuff as a python layer (?)
#       https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pyloss.py

class CaffeFinetuner:
  def __init__ (self, init_data):
    self.lrs = init_data['lrs']
    self.net = init_data['net']
    self.embedding = init_data['embedding']
    self.windowsize = init_data['windowsize']
    self.embedd = init_data['embedd']
    self.numtags = init_data['numtags']
    self.batchsize = init_data['batchsize']
    self.labelnoise = init_data['labelnoise']
    self.alpha = init_data['alpha']
    self.eta = init_data['eta']
    self.weightdecay = init_data['weightdecay']

    self.bindex=0

    if self.alpha > 0:
      self.momembeddiff = np.zeros_like (self.embedding)
      self.momnet = init_data['momnet']

    self.comboinputs = np.zeros ((self.batchsize,self.windowsize*self.embedd+self.numtags,1,1),dtype='f')
    self.bogus = np.zeros ((self.batchsize,1,1,1),dtype='f')
    self.batchtokens = np.zeros ((self.batchsize,self.windowsize),dtype='i')

    self.predcomboinputs = np.zeros ((self.batchsize,self.windowsize*self.embedd+self.numtags,1,1),dtype='f')

  def prior (self):
    return self.net.params['lastip'][1].data.reshape(self.numtags)

  def predict (self, parts):
    for n, part in enumerate (parts):
      for pos, t in enumerate (part):
        self.predcomboinputs[n,pos*self.embedd:(pos+1)*self.embedd,0,0] = self.embedding[:,t]

    self.net.set_input_arrays (self.predcomboinputs, self.bogus)
    res = self.net.forward ()

    # TODO: I'm assuming len (parts) < self.batchsize ...

    scores = [ np.copy (self.net.blobs['lastip'].data[n,:].reshape (self.numtags)) for n in range (len (parts)) ]

    return scores

  def update (self, part, labels):
    for pos, t in enumerate (part):
      self.batchtokens[self.bindex,pos] = t
      self.comboinputs[self.bindex,pos*self.embedd:(pos+1)*self.embedd,0,0] = self.embedding[:,t]

    self.comboinputs[self.bindex,(self.windowsize*self.embedd):,0,0] = self.labelnoise
    for l in labels:
      if l < self.numtags:
        self.comboinputs[self.bindex,(self.windowsize*self.embedd)+l,0,0] = 1.0-self.labelnoise

    self.bindex += 1

    if self.bindex >= self.batchsize:
      self.net.set_input_arrays (self.comboinputs, self.bogus)
      res = self.net.forward ()

      if self.eta > 0:
        self.net.backward ()
        data_diff = self.net.blobs['data'].diff.reshape (self.batchsize,self.windowsize*self.embedd+self.numtags,1,1)

        if self.alpha > 0:
          self.momembeddiff *= self.alpha
          for ii in range (self.batchsize):
            for jj in range (self.windowsize):
              self.momembeddiff[:,self.batchtokens[ii,jj]] += self.lrs['embedding'] * self.eta * data_diff[ii,jj*self.embedd:(jj+1)*self.embedd,0,0]
          self.embedding -= self.momembeddiff
        else:
          for ii in range (self.batchsize):
            for jj in range (self.windowsize):
              self.embedding[:,self.batchtokens[ii,jj]] -= self.lrs['embedding'] * self.eta * data_diff[ii,jj*self.embedd:(jj+1)*self.embedd,0,0]

        if self.weightdecay > 0:
          self.embedding *= (1.0 - self.weightdecay * self.lrs['embedding'] * self.eta)

        if self.alpha > 0:
          for (name,layer,momlayer) in zip (self.net._layer_names,
                                            self.net.layers,
                                            self.momnet.layers):
             blobnum = 0 
             for (blob,momblob) in zip (layer.blobs,momlayer.blobs): 
               myeta = self.lrs[(name,blobnum)] * self.eta 
               momblob.data[:] *= self.alpha
               momblob.data[:] += myeta * blob.diff
               blob.data[:] -= momblob.data
               if self.weightdecay > 0:
                 blob.data[:] *= (1.0 - self.weightdecay * myeta)
               blobnum = blobnum + 1 
        else:
          for (name,layer) in zip (self.net._layer_names,
                                   self.net.layers):
             blobnum = 0 
             for blob in layer.blobs:
               myeta = self.lrs[(name,blobnum)] * self.eta 
               blob.data[:] -= myeta * blob.diff
               blob.data[:] *= (1.0 - self.weightdecay * myeta)
               blobnum = blobnum + 1 

      self.bindex = 0

      return (True, res['loss']+0)
    else:
      return (False,)
