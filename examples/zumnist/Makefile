.SECONDARY:
.PHONY: all clean

numnets=6
nets=$(shell seq 1 $(numnets))

all: compiledlenet3
	
#	$(patsubst %,testpreds%,$(nets))

#	$(patsubst %,lenet_solver%.prototxt,$(nets)) $(patsubst %,lenet_train_test%.prototxt,$(nets)) $(patsubst %,mnist_train_%,$(nets))

clean:
	rm -f $(wildcard generated*/*.prototxt) $(wildcard generated*/t*.dat) \
	      $(wildcard generated*/*.caffemodel) $(wildcard generated*/*.solverstate) 	\
	      $(wildcard generated*/lenet?) $(wildcard generated*/testpreds*) \
	      $(wildcard generated*/mergedpreds.hdf5) $(wildcard compiledlenet*)
	rm -rf $(wildcard mnist_train_*)

generated/lenet_solver%.prototxt: make_lenet_solver
	./make_lenet_solver $* > $@

generated/lenet_train_test%.prototxt: make_lenet_train_test
	./make_lenet_train_test $* > $@

generated/train-images-idx3-ubyte.%.dat: ../../data/mnist/train-images-idx3-ubyte ./split-mnist-features
	cat $< | ./split-mnist-features $* $(numnets) > $@

generated/train-labels-idx1-ubyte.%.dat: ../../data/mnist/train-labels-idx1-ubyte ./split-mnist-labels
	cat $< | ./split-mnist-labels $* $(numnets) > $@

mnist_train_%: generated/train-images-idx3-ubyte.%.dat generated/train-labels-idx1-ubyte.%.dat
	../../build/examples/mnist/convert_mnist_data.bin --v=0 --logtostderr $^ $@ --backend=lmdb

generated/lenet%: generated/lenet_solver%.prototxt generated/lenet_train_test%.prototxt mnist_train_%
	../../build/tools/caffe train --solver=$<
	touch $@

generated/testpreds%: generated/lenet%
	PYTHONPATH=`pwd`/../../python ./output-preds lenet_trainpreds generated/lenet$*_iter_10000.caffemodel > $@

generated/mergedpreds.hdf5: merge_testpreds.py $(patsubst %,generated/testpreds%,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_testpreds.py $(numnets) $@

compiledlenet: generated/mergedpreds.hdf5 lenet_compileensemble_solver lenet_compileensemble_train
	../../build/tools/caffe train --solver=lenet_compileensemble_solver
	touch $@

generated2/lenet_solver%.prototxt: make_lenet_solver
	./make_lenet_solver $* 2 > $@

generated2/lenet%: generated2/lenet_solver%.prototxt generated/lenet_train_test%.prototxt mnist_train_% compiledlenet
	../../build/tools/caffe train --solver=$< --weights=lenetcompile_iter_40000.caffemodel
	touch $@

generated2/testpreds%: generated2/lenet%
	PYTHONPATH=`pwd`/../../python ./output-preds lenet_trainpreds generated2/lenet$*_iter_10000.caffemodel > $@

generated2/mergedpreds.hdf5: merge_testpreds.py $(patsubst %,generated2/testpreds%,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_testpreds.py $(numnets) $@

compiledlenet2: generated2/mergedpreds.hdf5 lenet_compileensemble_solver2 lenet_compileensemble_train2
	../../build/tools/caffe train --solver=lenet_compileensemble_solver2
	touch $@

generated3/lenet_solver%.prototxt: make_lenet_solver
	./make_lenet_solver $* 3 > $@

generated3/lenet%: generated3/lenet_solver%.prototxt generated/lenet_train_test%.prototxt mnist_train_% compiledlenet2
	../../build/tools/caffe train --solver=$< --weights=lenetcompile2_iter_40000.caffemodel
	touch $@

generated3/testpreds%: generated3/lenet%
	PYTHONPATH=`pwd`/../../python ./output-preds lenet_trainpreds generated3/lenet$*_iter_10000.caffemodel > $@

generated3/mergedpreds.hdf5: merge_testpreds.py $(patsubst %,generated3/testpreds%,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_testpreds.py $(numnets) $@

compiledlenet3: generated3/mergedpreds.hdf5 lenet_compileensemble_solver3 lenet_compileensemble_train3
	../../build/tools/caffe train --solver=lenet_compileensemble_solver3
	touch $@

generated4/lenet_solver%.prototxt: make_lenet_solver
	./make_lenet_solver $* 4 > $@

generated4/lenet%: generated4/lenet_solver%.prototxt generated/lenet_train_test%.prototxt mnist_train_% compiledlenet2
	../../build/tools/caffe train --solver=$< --weights=lenetcompile3_iter_40000.caffemodel
	touch $@

generated4/testpreds%: generated4/lenet%
	PYTHONPATH=`pwd`/../../python ./output-preds lenet_trainpreds generated4/lenet$*_iter_10000.caffemodel > $@

generated4/mergedpreds.hdf5: merge_testpreds.py $(patsubst %,generated4/testpreds%,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_testpreds.py $(numnets) $@

compiledlenet4: generated4/mergedpreds.hdf5 lenet_compileensemble_solver4 lenet_compileensemble_train4
	../../build/tools/caffe train --solver=lenet_compileensemble_solver4
	touch $@
