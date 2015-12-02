SHELL=/bin/zsh

export tagcutoff=150
export tokencutoff=100
export windowsize=10
export embedd=300
export batchsize=5001
export numconvk=100
export alpha=0.0
export eta=1e-1
export etamin=1e-5
export etadecay=0.9975
export weightdecay=1e-6
export labelnoise=1e-6
export numtags=100
export maxshufbuf=100000

export searnalpha=0.0
export searneta=1e-4
export searnetamin=1e-4
export searnetadecay=1
export searnvweta=1e-1
export searnmaxshufbuf=1000
export finetunedelay=10000

.SECONDARY: 

.PHONY: 

all: 

pretrain.model: pretrain.py tokenhisto taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2
	@rm -f $(wildcard pretrain.model.* pretrain.embedding.h5f.*)
	@GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

prevalidate: prevalidate.py tokenhisto taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2 pretrain.model
	@GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

searn: search_cws.py pretrain.model tokenhisto taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2
	GLOG_minloglevel=5 PYTHONPATH=../../python:../../../vowpal_wabbit/python python ./$<


#------------------------------------------------
# preprocessing stuff
#------------------------------------------------

WikiExtractor.py: 
	wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py

enwiki-20150901.id2cat.txt.bz2: extractcategories.pl enwiki-20150901-pages-articles-multistream.xml.bz2
	bzcat $(word 2,$^) | perl ./$(word 1,$^) | bzip2 > $@

enwiki-20150901-pages-articles-multistream.xml.bz2:
	wget --limit-rate=500K http://dumps.wikimedia.your.org/enwiki/20150901/$@

text/AA/wiki_00.bz2: WikiExtractor.py enwiki-20150901-pages-articles-multistream.xml.bz2
	python ./WikiExtractor.py --no-templates --bytes 100G -c <(bzcat enwiki-20150901-pages-articles-multistream.xml.bz2)

text/AA/wiki_00.shuf.bz2: text/AA/wiki_00.bz2
	bzcat $< |							\
	  perl -ne 'BEGIN { srand 69; $$/="</doc>\n"; }; 		\
	            1; $$r = rand (); print "$$r\001$$_\000";' |	\
	  sort -k1n -t$$'\001' -z -S50% --compress-program=lzop | 	\
	  perl -F'\001' -ane 'BEGIN { $$/="\000"; }; 			\
	                      1; chomp($$F[1]); print $$F[1]' |		\
	  bzip2 > $@

taghisto: enwiki-20150901.id2cat.txt.bz2
	bzcat $< | \
	perl -F'\t' -lane \
	  'shift @F; foreach (@F) { ++$$c{$$_}; } } \
	   1; { foreach (sort { $$c{$$b}<=>$$c{$$a} } keys %c) { print "$$_\t$$c{$$_}"; }' > $@

tokenhisto: text/AA/wiki_00.bz2 maketokenhisto.pl
	bzcat $(word 1,$^) | perl ./$(word 2,$^) > $@
