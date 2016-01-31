SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

#export numtags=400

ratelimit?=500K

ml-1m/README: ml-1m.zip
	unzip $<

ml-1m.zip:
	wget --limit-rate=$(ratelimit) http://files.grouplens.org/datasets/movielens/ml-1m.zip

full: full.py
	GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<
