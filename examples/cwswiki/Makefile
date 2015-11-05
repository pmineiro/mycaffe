SHELL=/bin/zsh

tagcutoff=150
tokencutoff=100

.SECONDARY: 

.PHONY: 

all: 

pretrain.model: pretrain.py pretrain.bz2
	GLOG_minloglevel=5 PYTHONPATH=../../python python ./$<

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

taghisto: enwiki-20150901.id2cat.txt.bz2
	bzcat $< | \
	perl -F'\t' -lane \
	  'shift @F; foreach (@F) { ++$$c{$$_}; } } \
	   1; { foreach (sort { $$c{$$b}<=>$$c{$$a} } keys %c) { print "$$_\t$$c{$$_}"; }' > $@

tokenhisto: text/AA/wiki_00.bz2 maketokenhisto.pl
	bzcat $(word 1,$^) | perl ./$(word 2,$^) > $@

pretrain.bz2: seed=69
pretrain.bz2: makepretrain.pl taghisto enwiki-20150901.id2cat.txt.bz2 tokenhisto text/AA/wiki_00.bz2
	perl ./$(word 1,$^) $(word 2,$^) $(tagcutoff) <(bzcat $(word 3,$^)) $(word 4,$^) $(tokencutoff) <(bzcat $(word 5,$^)) | \
	perl -pe 'BEGIN { srand $(seed); } $$_=rand()."\t$$_"' | \
	sort -S50% -t$$'\t' -k1,1n | \
	cut -d$$'\t' -f2- | bzip2 > $@

#num tags = 13039 at ./makepretrain.pl line 39, <GEN0> line 19374.
#num tokens = 260544 at ./makepretrain.pl line 66, <GEN2> line 260545.
#nocat = 4779776 at ./makepretrain.pl line 95, <GEN3> line 65302109.
#shortpara = 1049979 at ./makepretrain.pl line 96, <GEN3> line 65302109.
#shortsentence = 102595 at ./makepretrain.pl line 97, <GEN3> line 65302109.
