#! /usr/bin/env perl

use warnings;
use strict;

use IO::File;

local $\="\n";

my $tokenhistofile = shift @ARGV or die;
my $tokencutoff = shift @ARGV or die;

my $tokenhistofh = new IO::File $tokenhistofile, "r" or die "$tokenhistofile: $!";

my %tokens;

while (defined ($_ = <$tokenhistofh>))
  {
    chomp;
    my ($t, $c) = split /\t/, $_;

    last if $c < $tokencutoff;

    $tokens{$t} = 1 + scalar keys %tokens unless exists $tokens{$t};
  }

my $numtokens = scalar keys %tokens;
warn "num tokens = ", $numtokens;

my $id;
my $shortpara = 0;

my %cooc;

while (defined ($_ = <>))
  {
    chomp;

    do { $id = $1; next } if ! defined $id && m%^<doc id="(\d+)"%;
    undef $id if m%^</doc>%;

    next unless defined $id;

    my @sentences = split /(?<!i\.e|e\.g|c\.f|.cf)\.\s(?!,)/, $_;
    do { ++$shortpara; next } unless @sentences > 1;

    foreach my $s (@sentences)
      {
        my @t = grep { /\S/ } map { s/^\p{PosixPunct}+//; s/\p{PosixPunct}+$//; $_ } split /\s+/, $s;
        my @tid = map { $tokens{$_} || 0 } @t;

        foreach my $pos (2 .. $#tid-2)
          {
            my $thist = $tid[$pos];
            next unless $thist < 10000;

            foreach my $start (-2 .. 2)
              {
                next unless $start;
                ++$cooc{$thist}->{$tid[$pos+$start]};
              }
          }
      }
  }

warn "shortpara = ", $shortpara;

foreach my $tid (0 .. 9999)
  {
    print join "\t", $tid, map { "$_:$cooc{$tid}->{$_}" } sort { $a<=>$b} keys %{$cooc{$tid}};
  }
