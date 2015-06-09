SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

all: 

clean: 
	rm -f $(wildcard histo*) $(wildcard fofengram*.txt)
	rm -f $(wildcard fofesparsemodel*.*) $(wildcard fofesparsemodel*_e.*)

numvocab=80000
windowsize=2
alpha=0.7
cutoff=0.01

enwik%.zip:
	wget http://mattmahoney.net/dc/enwik$*.zip

histo%: enwik%.zip wikifil.pl
	zcat $(word 1,$^) | perl $(word 2,$^) | perl -lane 'print join "\n", @F' | sort -S30% | uniq -c | sort -S30% -k1rn > $@

fofengram%.txt: enwik%.zip wikifil.pl histo% makefofengram.pl
	zcat $(word 1,$^) | perl $(word 2,$^) | perl $(word 4,$^) $(numvocab) $(windowsize) $(alpha) $(cutoff) $(word 3,$^) | perl -ne 'BEGIN { srand 69; }; print rand(),"\t",$$_;' | sort -S50% -k1,1n | cut -f2- > $@

fofemodel%: fofe_small_unigram_train fofengram%txt
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofemodel.py $(word 1,$^) <(head -n `wc -l $(word 2,$^) | perl -lane 'print int(0.9*$$F[0])'` $(word 2,$^))

fofesparsemodel%: fofe_sparse_small_unigram_train fofengram%.txt
	@rm -f $(wildcard fofesparsemodel$*.* fofesparsemodel$*_e.*)
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparsemodel.py $(word 1,$^) <(head -n `wc -l $(word 2,$^) | perl -lane 'print int(0.9*$$F[0])'` $(word 2,$^)) $@
