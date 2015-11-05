#! /usr/bin/env perl

use warnings;
use strict;

my %tc;
my $id;

while (defined ($_ = <>))
  {
    do { $id = $1; next } if ! defined $id && m%^<doc id="(\d+)"%;
    undef $id if m%^</doc>%;

    next unless defined $id;

    chomp;
    my @t = grep { /\S/ } map { s/^\p{PosixPunct}+//; s/\p{PosixPunct}+$//; $_ } split /\s+/, $_;

    foreach my $t (@t) { ++$tc{$t}; } # do { warn join "::", @t; die $_ } if $t eq '"The'; }
  }

foreach (sort { $tc{$b}<=>$tc{$a} } keys %tc)
  {
    print "$_\t$tc{$_}\n";
  }
