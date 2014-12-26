.SECONDARY:
.PHONY: all clean

numnets=6
nets=$(shell seq 1 $(numnets))

all: compiledlenet
	
#	$(patsubst %,testpreds%,$(nets))

#	$(patsubst %,lenet_solver%.prototxt,$(nets)) $(patsubst %,lenet_train_test%.prototxt,$(nets)) $(patsubst %,mnist_train_%,$(nets))

clean:
	rm -f $(wildcard *.prototxt) $(wildcard t*.dat) \
	      $(wildcard *.caffemodel) $(wildcard *.solverstate) \
	      $(wildcard lenet?) $(wildcard testpreds*)
	rm -rf $(wildcard mnist_train_*)

lenet_solver%.prototxt: make_lenet_solver
	./make_lenet_solver $* > $@

lenet_train_test%.prototxt: make_lenet_train_test
	./make_lenet_train_test $* > $@

train-images-idx3-ubyte.%.dat: ../../data/mnist/train-images-idx3-ubyte ./split-mnist-features
	cat $< | ./split-mnist-features $* $(numnets) > $@

train-labels-idx1-ubyte.%.dat: ../../data/mnist/train-labels-idx1-ubyte ./split-mnist-labels
	cat $< | ./split-mnist-labels $* $(numnets) > $@

mnist_train_%: train-images-idx3-ubyte.%.dat train-labels-idx1-ubyte.%.dat
	../../build/examples/mnist/convert_mnist_data.bin --v=0 --logtostderr $^ $@ --backend=lmdb

lenet%: lenet_solver%.prototxt lenet_train_test%.prototxt mnist_train_%
	../../build/tools/caffe train --solver=$<
	touch $@

testpreds%: lenet%
	PYTHONPATH=`pwd`/../../python ./output-preds lenet_trainpreds lenet$*_iter_5000.caffemodel > $@

mergedpreds.hdf5: merge_testpreds.py $(patsubst %,testpreds%,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_testpreds.py $@

compiledlenet: mergedpreds.hdf5
	../../build/tools/caffe train --solver=lenet_compileensemble_solver
	touch $@
