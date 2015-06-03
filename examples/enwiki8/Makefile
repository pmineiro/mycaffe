.SECONDARY: 

.PHONY: 

all: 

clean: 
	rm -f $(wildcard histo8)

numvocab=80000
windowsize=1
alpha=0.7
cutoff=0.01

enwik8.zip:
	wget http://mattmahoney.net/dc/enwik8.zip

histo8: enwik8.zip wikifil.pl
	zcat $(word 1,$^) | perl $(word 2,$^) | perl -lane 'print join "\n", @F' | sort -S30% | uniq -c | sort -S30% -k1rn > $@

fofengram.txt: enwik8.zip wikifil.pl histo8 makefofengram.pl
	zcat $(word 1,$^) | perl $(word 2,$^) | perl $(word 4,$^) $(numvocab) $(windowsize) $(alpha) $(cutoff) $(word 3,$^) | perl -ne 'BEGIN { srand 69; }; print rand(),"\t",$$_;' | sort -S50% -k1,1n | cut -f2- > $@

fofemodel: fofe_small_unigram_train fofengram.txt
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofemodel.py $^

fofesparsemodel: fofe_sparse_small_unigram_train fofengram.txt
	GLOG_minloglevel=5 PYTHONPATH=../../python python makefofesparsemodel.py $^
