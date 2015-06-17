SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

all: 

clean: 
	rm -f $(wildcard histo*) $(wildcard fofengram*.txt) 
	rm -f $(wildcard fofebaserates*.txt) $(wildcard numlinesfofe*)
	rm -f $(wildcard fofesparsemodel*.*) $(wildcard fofesparsemodel*_e.*)
	rm -f $(wildcard fofesparserembedmodel*.*) $(wildcard fofesparserembedmodel*_e.*)

numvocab=80000
windowsize=2
alpha=0.7
cutoff=0.01

enwik%.zip:
	wget http://mattmahoney.net/dc/enwik$*.zip

histo%: enwik%.zip wikifil.pl
	zcat $(word 1,$^) | perl $(word 2,$^) | perl -lane 'print join "\n", @F' | sort -S30% | uniq -c | sort -S30% -k1rn > $@

fofengram%.txt: enwik%.zip wikifil.pl histo% makefofengram.pl
	zcat $(word 1,$^) | perl $(word 2,$^) | perl $(word 4,$^) $(numvocab) $(windowsize) $(alpha) $(cutoff) $(word 3,$^) | perl -ne 'BEGIN { srand 69; }; print rand(),"\t",$$_;' | sort -S80% --compress-program=lzop -k1,1n | cut -f2- > $@

fofebaserates%.txt: fofengram%.txt
	perl -lane '++$$c{$$F[0]}; } { foreach (sort { $$a<=>$$b} keys %c) { print "$$_\t$$c{$$_}"; }' $< > $@

numlinesfofengram%: fofengram%.txt
	wc -l $^ > $@

fofemodel%: fofe_small_unigram_train fofengram%txt
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofemodel.py $(word 1,$^) <(head -n `wc -l $(word 2,$^) | perl -lane 'print int(0.9*$$F[0])'` $(word 2,$^))

fofesparsemodel%: fofe_sparse_small_unigram_train numlinesfofengram% fofengram%.txt
	@rm -f $(wildcard fofesparsemodel$*.* fofesparsemodel$*_e.*)
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparsemodel.py $(word 1,$^) <(head -n `cat $(word 2,$^) | perl -lane 'print int(0.9*$$F[0])'` $(word 3,$^)) $@

fofesparserembedmodel%: fofe_sparse_rembed_small_unigram_train fofebaserates%.txt numlinesfofengram% fofengram%.txt 
	@rm -f $(wildcard fofesparserembedmodel$* fofesparserembedmodel$*_e fofesparserembedmodel$*.* fofesparserembedmodel$*_e.*)
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparserandcodemodel.py $(word 1,$^) $(word 2,$^) `cat $(word 3,$^) | perl -lane 'print int(0.9*$$F[0])'` $(word 4,$^) $@

sparserembedmodelclean%:
	@rm -f $(wildcard fofesparserembedmodel$*)

fofesparserembedlogisticmodel%: fofesparserembedmodel% fofe_sparse_rembed_small_unigram_test fofe_sparse_rembed_small_unigram_logistic_train numlinesfofengram% fofengram%.txt
	@rm -f $(wildcard fofesparserembedlogisticmodel*.* fofesparserembedlogisticmodel*_e.*)
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparserembedlogisticmodel.py $(word 1,$^)_e $(word 2,$^) $(word 1,$^) $(word 3,$^) `cat $(word 4,$^) | perl -lane 'print int(0.9*$$F[0])'` $(word 5,$^) $@

sparserembedlogisticmodelclean:
	@rm -f $(wildcard fofesparserembedlogisticmodel*.* fofesparserembedlogisticmodel*_e.*)
