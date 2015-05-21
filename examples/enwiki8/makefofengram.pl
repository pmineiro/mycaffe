#! /usr/bin/env perl

use warnings;
use strict;

use IO::File;
use IO::Handle;

my $vocabsize = shift @ARGV or die;
my $windowsize = shift @ARGV or die;
my $alpha = shift @ARGV or die;
my $cutoff = shift @ARGV or die;

my $histfile = shift @ARGV or die;
my $histfh = new IO::File $histfile, "r" or die "$histfile: $!";
my %vocab;
my $curvocabsize = 0;

while (defined ($_ = <$histfh>))
  {
    s/^\s+//;
    my ($cnt, $token) = split /\s+/, $_;

    $vocab{$token} = ++$curvocabsize;

    last if $curvocabsize >= $vocabsize;
  }

my @prefix = map { 1 } (2 .. $windowsize);

while (defined ($_ = <>))
  {
    chomp;
    my @words = split /\s+/, $_;
    my @enc = map { exists $vocab{$_} ? $vocab{$_}+2 : $vocabsize+1 } @words;
    unshift (@enc, @prefix);

    my %state;

    foreach my $pos (0 .. ($#enc-$windowsize))
      {
        my $offset = 0;

        print "$enc[$pos+$windowsize]\t";

        foreach my $relpos (1 .. ($windowsize-1))
          {
            my $k = $offset + $enc[$pos+$windowsize-$relpos];
            print "$k:1 ";
            $offset += $vocabsize + 2;
          }

        $state{$enc[$pos]} ||= 0;
        while (my ($k, $v) = each %state)
          {
            $state{$k} *= $alpha; $v *= $alpha;
            do { $v += 1; $state{$k} += 1; } if ($k == $enc[$pos]);
            do { $v = 0; delete $state{$k}; next } if ($state{$k} < $cutoff);
            $k += $offset;
            my $vbuf = sprintf "%3.3f", $v;
            $vbuf =~ s/\.?0*$//;
            print "$k:$vbuf ";
          }

        print "\n";
      }
  }
