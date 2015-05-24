#! /usr/bin/env python

import caffe
import numpy as np
import sys

vocabsize=80000
maxibatchsize=20000

#------------------------------------------------

invocabsize=vocabsize+2
outvocabsize=vocabsize+1

solver=caffe.SGDSolver(sys.argv[1])
solver.net.set_mode_gpu()
solver.net.set_phase_train()
f=open(sys.argv[2],'r')

data=np.zeros((maxibatchsize,1,1,invocabsize),dtype='f')
labels=np.zeros(maxibatchsize,dtype='f')
bindex=0
for line in f:
    yx=[word for word in line.split(' ')]
    labels[bindex]=int(yx[0])-1

    for word in yx[1:]:
        iv=[subword for subword in word.split(':')]
        data[bindex,0,0,int(iv[0])-1]=float(iv[1])
    
    bindex=bindex+1

    if bindex >= maxibatchsize:
        solver.net.set_input_arrays(data,labels)
        solver.solve()
        print "yo\n"
        data[:,:,:,:]=0
        labels[:]=0
        bindex=0
