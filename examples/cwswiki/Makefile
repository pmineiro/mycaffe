SHELL=/bin/zsh

.SECONDARY: 

.PHONY: 

export numtags=400
export vwbits=29
export minsentlength=8
export summarylength=3
export aggregatechars=1000
export labelnoise=1e-4

ratelimit?=500K

enwiki-20150901-pages-articles-multistream.xml.bz2:
	wget --limit-rate=$(ratelimit) http://dumps.wikimedia.your.org/enwiki/20150901/$@

enwiki-20150901.id2cat.txt.bz2: extractcategories.pl enwiki-20150901-pages-articles-multistream.xml.bz2
	bzcat $(word 2,$^) | perl ./$(word 1,$^) | bzip2 > $@

taghisto: enwiki-20150901.id2cat.txt.bz2
	bzcat $< | \
	perl -F'\t' -lane \
	  'shift @F; foreach (@F) { ++$$c{$$_}; } } \
	   1; { foreach (sort { $$c{$$b}<=>$$c{$$a} } keys %c) { print "$$_\t$$c{$$_}"; }' > $@

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

pretrainaggregate.model: pretrainaggregate.py taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2
	python ./$< | vw --invariant --power_t 0.25 -l 1 --multilabel_oaa $(numtags) -b $(vwbits) --multilabel_noise $(labelnoise) --loss_function logistic --ngram 3 --skips 1 -f pass0aggregate.vw

pretrainsummary.model: pretrainsummary.py taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2
	python ./$< | vw --invariant --power_t 0.25 -l 1 --multilabel_oaa $(numtags) -b $(vwbits) --multilabel_noise $(labelnoise) --loss_function logistic --ngram 3 --skips 1 -f pass0summary.vw

prevalaggregate%: prevalaggregate.py taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2 pass%aggregate.vw
	PYTHONPATH=../../../vowpal_wabbit/python python ./$< pass$*aggregate.vw

prevalsummary%: prevalsummary.py taghisto enwiki-20150901.id2cat.txt.bz2 text/AA/wiki_00.shuf.bz2 pass%summary.vw
	PYTHONPATH=../../../vowpal_wabbit/python python ./$< pass$*summary.vw

tokenhisto: maketokenhisto.py text/AA/wiki_00.shuf.bz2
	python ./$< > $@

export numtokens=100000
export numpos=30
export embedd=100
export batchsize=1000
export numconvk=30
export numip1=200
export numip2=200
export alpha=0.8
export eta=1e-2
export etamin=1e-6
export etadecay=0.995
export weightdecay=0
export numdocs=1000
export maxshufbuf=10000

tongnet.model: tongtrain.py tokenhisto text/AA/wiki_00.shuf.bz2
	rm -f $(wildcard tongnet.model.*) $(wildcard tongnet.embedding.*)
	GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<
