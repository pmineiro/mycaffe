SHELL=/bin/zsh

tagcutoff=150
tokencutoff=100
windowsize=10
embedd=300
batchsize=5001
numconvk=100
alpha=0.0
eta=1e-3
etadecay=0.999
weightdecay=1e-6
labelnoise=1e-6
numtags=100
maxshufbuf=100000

.SECONDARY: 

.PHONY: 

all: 

pretrain.model: pretrain.py tokenhisto taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2
	@rm -f $(wildcard pretrain.model.* pretrain.embedding.h5f.*)
	@MAXSHUFBUF=$(maxshufbuf) NUMTAGS=$(numtags) ALPHA=$(alpha) ETA=$(eta) ETADECAY=$(etadecay) WEIGHTDECAY=$(weightdecay) LABELNOISE=$(labelnoise) WINDOWSIZE=$(windowsize) EMBEDD=$(embedd) BATCHSIZE=$(batchsize) NUMCONVK=$(numconvk) TAGCUTOFF=$(tagcutoff) TOKENCUTOFF=$(tokencutoff) GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

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
