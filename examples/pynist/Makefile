SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

dataset = train-labels-idx1-ubyte.gz train-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz 

all: 

test: test.py train.model 
	@GLOG_minloglevel=5 PYTHONPATH=../../python python ./$< $(word 2, $^)

train.model: train.py $(dataset)
	@GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

trainlowrank.model: trainlowrank.py $(dataset)
	@GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

trainlowrankpair.model: trainlowrankpair.py $(dataset)
	@GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

testlowrank: test.py trainlowrank.model 
	@GLOG_minloglevel=5 PYTHONPATH=../../python python ./$< $(word 2, $^)

%.check:
	@test -x "$$(which $*)" || {				\
	  echo "ERROR: you need to install $*" 1>&2;		\
	  exit 1;						\
	}

train-%.gz: wget.check
	wget http://yann.lecun.com/exdb/mnist/train-$*.gz

t10k-%.gz: wget.check
	wget http://yann.lecun.com/exdb/mnist/t10k-$*.gz
