SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

all: 

clean: 

lambda=0.0
nswplacebomodel: nsw_train nsw_control.txt psid_controls.txt
	GLOG_minloglevel=5 lambda=$(lambda) PYTHONPATH=../../python python makenswmodel.py $^ $@

advplacebomodel: nsw_advps_train nsw_control.txt psid_controls.txt
	GLOG_minloglevel=5 lambda=$(lambda) PYTHONPATH=../../python python makeadvpsmodel.py $^ $@
