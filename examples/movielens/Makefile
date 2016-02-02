SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

export numusers = 3900
export nummovies = 6400
export numratings = 5

batchsize ?= 1000
export batchsize
maxshufbuf ?= 10000
export maxshufbuf
eta ?= 1e-2
export eta
etamin ?= 1e-7
export etamin
etadecay ?= 0.999
export etadecay
weightdecay ?= 1e-5
export weightdecay

ml-1m/README: ml-1m.zip
	unzip $<

ratelimit?=500K
ml-1m.zip:
	wget --limit-rate=$(ratelimit) http://files.grouplens.org/datasets/movielens/ml-1m.zip

linear: linear.py
	GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

full: full.py
	GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

rank ?= 10
export rank

fm: fm.py
	GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

numips ?= 40,40
export numips

crossfm: crossfm.py
	GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<
