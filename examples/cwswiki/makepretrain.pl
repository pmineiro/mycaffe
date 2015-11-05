#! /usr/bin/env perl

use warnings;
use strict;

use IO::File;
use Digest::MD5 qw (md5_hex);
use Data::Dumper;

local $\="\n";

my $taghistofile = shift @ARGV or die;
my $tagcutoff = shift @ARGV or die;
my $id2catfile = shift @ARGV or die;
my $tokenhistofile = shift @ARGV or die;
my $tokencutoff = shift @ARGV or die;
my $dumpfile = shift @ARGV or die;

my $taghistofh = new IO::File $taghistofile, "r" or die "$taghistofile: $!";
my $idtocatfh = new IO::File $id2catfile or die "$id2catfile: $!";
my $tokenhistofh = new IO::File $tokenhistofile, "r" or die "$tokenhistofile: $!";
my $dumpfh = new IO::File $dumpfile or die "$dumpfile: $!";

my %tags;

while (defined ($_ = <$taghistofh>))
  {
    chomp;
    my ($t, $c) = split /\t/, $_;

    last if $c < $tagcutoff;

    my $nonwords = scalar grep { !/^[A-Za-z]+$/} split /\s+/, $t;
    next if $nonwords;

    $tags{$t} = 1 + scalar keys %tags unless exists $tags{$t};
  }

warn "num tags = ", scalar keys %tags;

my %id2cat;

while (defined ($_ = <$idtocatfh>))
  {
    chomp;
    my ($id, @tags) = split /\t/, $_;
    next unless substr (md5_hex ($id), -1) eq "a";
    my @goodtags = map { $tags{$_} } grep { exists $tags{$_} } @tags;
    next unless @goodtags;

    $id2cat{$id} = \@goodtags;
  }

my %tokens;

while (defined ($_ = <$tokenhistofh>))
  {
    chomp;
    my ($t, $c) = split /\t/, $_;

    last if $c < $tokencutoff;

    $tokens{$t} = 1 + scalar keys %tokens unless exists $tokens{$t};
  }

warn "num tokens = ", scalar keys %tokens;

my $id;
my %nocat;
my $shortpara = 0;
my $shortsentence = 0;

while (defined ($_ = <$dumpfh>))
  {
    chomp;

    do { $id = $1; next } if ! defined $id && m%^<doc id="(\d+)"%;
    undef $id if m%^</doc>%;

    next unless defined $id;
    do { $nocat{$id} = 1; next } unless exists $id2cat{$id};

    my @sentences = split /(?<!i\.e|e\.g|c\.f|.cf)\.\s(?!,)/, $_;
    do { ++$shortpara; next } unless @sentences > 1;

    foreach my $s (@sentences)
      {
        my @t = grep { /\S/ } map { s/^\p{PosixPunct}+//; s/\p{PosixPunct}+$//; $_ } split /\s+/, $s;
        my @tid = map { $tokens{$_} || 0 } @t;
        do { ++$shortsentence; next } unless 3 < @tid;
        print ((join ",", @{$id2cat{$id}}), "\t", (join " ", @tid));
      }
  }

warn "nocat = ", scalar keys %nocat;
warn "shortpara = ", $shortpara;
warn "shortsentence = ", $shortsentence;
