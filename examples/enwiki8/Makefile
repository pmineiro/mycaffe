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

fofe.train.hdf5: enwik8.zip wikifil.pl histo8 makefofengram.pl fofe_small_unigram_makedata.py
	zcat $(word 1,$^) | perl $(word 2,$^) | perl $(word 4,$^) $(numvocab) $(windowsize) $(alpha) $(cutoff) $(word 3,$^) | python $(word 5,$^) fofe

fofemodel: fofe.train.hdf5
	../../build/tools/caffe train --solver=fofe_small_unigram_solver
