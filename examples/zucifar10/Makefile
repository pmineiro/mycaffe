.SECONDARY:
.PHONY: all clean

numnets=6
nets=$(shell seq 1 $(numnets))

all: compiledalexnet
	
#	$(patsubst %,trainpreds%,$(nets))

#	$(patsubst %,alexnet_solver%.prototxt,$(nets)) $(patsubst %,alexnet_train_test%.prototxt,$(nets)) $(patsubst %,mnist_train_%,$(nets))

clean:
	rm -f $(wildcard generated*)

generated/%/cifar10_full_solver.prototxt: make_alexnet_solver
	@test -d generated/$* || mkdir -p generated/$*
	./make_alexnet_solver $* > $@

generated/%/cifar10_full_solver_lr1.prototxt: make_alexnet_solver_lr1
	@test -d generated/$* || mkdir -p generated/$*
	./make_alexnet_solver_lr1 $* > $@

generated/%/cifar10_full_solver_lr2.prototxt: make_alexnet_solver_lr2
	@test -d generated/$* || mkdir -p generated/$*
	./make_alexnet_solver_lr2 $* > $@

generated/%/cifar10_full_train_test.prototxt: make_alexnet_train_test
	@test -d generated/$* || mkdir -p generated/$*
	./make_alexnet_train_test $* > $@

generated/%/data_batch_1.bin: ./split-cifar $(wildcard ../../data/cifar10/data_batch_*.bin)
	@test -d generated/$* || mkdir -p generated/$*
	cat $(wordlist 2,1000,$^) | ./split-cifar $* $(numnets) > $@

generated/%/cifar10_train_leveldb/LOCK: generated/%/data_batch_1.bin
	@rm -rf $(wildcard generated/$*/cifar10_*_leveldb)
	../../build/examples/cifar10/convert_cifar_data.bin ./generated/$* ./generated/$*

generated/%/alexnet: generated/%/cifar10_full_solver.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK
	../../build/tools/caffe train --solver=$<
	touch $@

generated/%/alexnet_lr: generated/%/cifar10_full_solver_lr1.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK generated/%/alexnet
	../../build/tools/caffe train --solver=$< --snapshot=./generated/$*/alexnet_iter_60000.solverstate
	touch $@

generated/%/alexnet_lr2: generated/%/cifar10_full_solver_lr2.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK generated/%/alexnet_lr
	../../build/tools/caffe train --solver=$< --snapshot=./generated/$*/alexnet_iter_65000.solverstate
	touch $@

generated/%/trainpreds: generated/%/alexnet_lr2
	PYTHONPATH=`pwd`/../../python ./output-preds alexnet_trainpreds generated/$*/alexnet_iter_70000.caffemodel > $@

generated/mergedpreds.hdf5: merge_trainpreds.py $(patsubst %,generated/%/trainpreds,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_trainpreds.py $(numnets) $@

compiledalexnet: cifar10_full_compile_solver cifar10_full_compile_train generated/mergedpreds.hdf5
	../../build/tools/caffe train --solver=$< --weights=generated/1/alexnet_iter_70000.caffemodel
	touch $@

compiledalexnet_lr1: cifar10_full_compile_solver_lr1 cifar10_full_compile_train generated/mergedpreds.hdf5 compiledalexnet
	../../build/tools/caffe train --solver=$< --snapshot=alexnetcompile_iter_2400.solverstate
	touch $@

compiledalexnet_lr2: cifar10_full_compile_solver_lr2 cifar10_full_compile_train generated/mergedpreds.hdf5 compiledalexnet_lr1
	../../build/tools/caffe train --solver=$< --snapshot=alexnetcompile_iter_4800.solverstate
	touch $@

# UR HERE

generated2/%/cifar10_full_solver.prototxt: make_alexnet_solver
	@test -d generated2/$* || mkdir -p generated2/$*
	./make_alexnet_solver $* 2 > $@

generated2/%/cifar10_full_solver_lr1.prototxt: make_alexnet_solver_lr1
	@test -d generated2/$* || mkdir -p generated2/$*
	./make_alexnet_solver_lr1 $* 2 > $@

generated2/%/cifar10_full_solver_lr2.prototxt: make_alexnet_solver_lr2
	@test -d generated2/$* || mkdir -p generated2/$*
	./make_alexnet_solver_lr2 $* 2 > $@

generated2/%/alexnet: generated2/%/cifar10_full_solver.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK compiledalexnet_lr2
	../../build/tools/caffe train --solver=$< --weights=alexnetcompile_iter_7200.caffemodel
	touch $@

generated2/%/alexnet_lr: generated2/%/cifar10_full_solver_lr1.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK generated2/%/alexnet
	../../build/tools/caffe train --solver=$< --snapshot=./generated2/$*/alexnet_iter_4000.solverstate
	touch $@

generated2/%/alexnet_lr2: generated2/%/cifar10_full_solver_lr2.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK generated2/%/alexnet_lr
	../../build/tools/caffe train --solver=$< --snapshot=./generated2/$*/alexnet_iter_6000.solverstate
	touch $@

generated2/%/trainpreds: generated2/%/alexnet_lr2
	PYTHONPATH=`pwd`/../../python ./output-preds alexnet_trainpreds generated2/$*/alexnet_iter_8000.caffemodel > $@

generated2/mergedpreds.hdf5: merge_trainpreds.py $(patsubst %,generated2/%/trainpreds,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_trainpreds.py $(numnets) $@

compiledalexnet2: cifar10_full_compile_solver2 cifar10_full_compile_train generated2/mergedpreds.hdf5
	../../build/tools/caffe train --solver=$< --weights=generated2/1/alexnet_iter_8000.caffemodel
	touch $@

compiledalexnet2_lr1: cifar10_full_compile_solver2_lr1 cifar10_full_compile_train generated2/mergedpreds.hdf5 compiledalexnet2
	../../build/tools/caffe train --solver=$< --snapshot=alexnetcompile2_iter_2000.solverstate
	touch $@

compiledalexnet2_lr2: cifar10_full_compile_solver2_lr2 cifar10_full_compile_train generated2/mergedpreds.hdf5 compiledalexnet2_lr1
	../../build/tools/caffe train --solver=$< --snapshot=alexnetcompile2_iter_4000.solverstate
	touch $@

generated3/%/cifar10_full_solver.prototxt: make_alexnet_solver
	@test -d generated3/$* || mkdir -p generated3/$*
	./make_alexnet_solver $* 3 > $@

generated3/%/cifar10_full_solver_lr1.prototxt: make_alexnet_solver_lr1
	@test -d generated3/$* || mkdir -p generated3/$*
	./make_alexnet_solver_lr1 $* 3 > $@

generated3/%/cifar10_full_solver_lr2.prototxt: make_alexnet_solver_lr2
	@test -d generated3/$* || mkdir -p generated3/$*
	./make_alexnet_solver_lr2 $* 3 > $@

generated3/%/alexnet: generated3/%/cifar10_full_solver.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK compiledalexnet_lr2
	../../build/tools/caffe train --solver=$< --weights=alexnetcompile2_iter_6000.caffemodel
	touch $@

generated3/%/alexnet_lr: generated3/%/cifar10_full_solver_lr1.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK generated3/%/alexnet
	../../build/tools/caffe train --solver=$< --snapshot=./generated3/$*/alexnet_iter_4000.solverstate
	touch $@

generated3/%/alexnet_lr2: generated3/%/cifar10_full_solver_lr2.prototxt generated/%/cifar10_full_train_test.prototxt generated/%/cifar10_train_leveldb/LOCK generated3/%/alexnet_lr
	../../build/tools/caffe train --solver=$< --snapshot=./generated3/$*/alexnet_iter_6000.solverstate
	touch $@

generated3/%/trainpreds: generated3/%/alexnet_lr2
	PYTHONPATH=`pwd`/../../python ./output-preds alexnet_trainpreds generated3/$*/alexnet_iter_8000.caffemodel > $@

generated3/mergedpreds.hdf5: merge_trainpreds.py $(patsubst %,generated3/%/trainpreds,$(nets)) 
	paste $(wordlist 2,1000,$^) | python ./merge_trainpreds.py $(numnets) $@

compiledalexnet3: cifar10_full_compile_solver3 cifar10_full_compile_train generated3/mergedpreds.hdf5
	../../build/tools/caffe train --solver=$< --weights=generated3/1/alexnet_iter_8000.caffemodel
	touch $@

compiledalexnet3_lr1: cifar10_full_compile_solver3_lr1 cifar10_full_compile_train generated3/mergedpreds.hdf5 compiledalexnet3
	../../build/tools/caffe train --solver=$< --snapshot=alexnetcompile3_iter_2000.solverstate
	touch $@

compiledalexnet3_lr2: cifar10_full_compile_solver3_lr2 cifar10_full_compile_train generated3/mergedpreds.hdf5 compiledalexnet3_lr1
	../../build/tools/caffe train --solver=$< --snapshot=alexnetcompile3_iter_4000.solverstate
	touch $@
