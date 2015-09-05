SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

all: 

clean: 

nswplacebomodel: lambda=1e-3
nswplacebomodel: nsw_train nsw_control.txt psid_controls.txt
	GLOG_minloglevel=5 lambda=$(lambda) PYTHONPATH=../../python python makenswmodel.py $^ $@

advplacebomodel: lambda=0.1
advplacebomodel: eta=1e-3
advplacebomodel: nsw_advps_train nsw_control.txt psid_controls.txt
	GLOG_minloglevel=5 eta=$(eta) lambda=$(lambda) PYTHONPATH=../../python python makeadvpsmodel.py $^ $@
