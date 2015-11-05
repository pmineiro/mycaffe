#! /usr/bin/env perl

use warnings;
use strict;

#    ...
#  <page>
#    <id>12</id>
#    <revision>
#      ...
#      <text xml:space="preserve">{{Redirect2|Anarchist|Anarchists|the fictional character|Anarchist (comics)|other uses|Anarchists (disambiguation)}}
#...
#[[Category:Anarchism| ]]
#[[Category:Political culture]]
#[[Category:Political ideologies]]
#[[Category:Social theories]]
#[[Category:Anti-fascism]]
#[[Category:Anti-capitalism]]
#[[Category:Far-left politics]]</text>
#    </revision>
#  </page>
#  ...

local $\="\n";

my $inpage = 0;
my $inrevision = 0;
my $intext = 0;
my $id;
my %c;

while (defined ($_ = <>))
  {
    $inpage = 1 if ! $inpage && m%^\s*<page>\s*$%;
    $id = $1 if $inpage && ! defined $id && m%^\s*<id>(\d+)</id>\s*$%;
    $inrevision = 1 if $inpage && ! $inrevision && m%^\s*<revision>\s*$%;
    $intext = 1 if $inrevision && ! $intext && m%^\s*<text xml:space="preserve">%;

    if ($intext)
      {
        while (m%\[\[Category:(.*?)\]\]%g)
          {
            my $cat = $1;
            $cat =~ s/\t/ /g;
            $c{$cat}=1;
          }
      }

    $intext = 0 if $inrevision && $intext && m%</text>\s*$%;
    do { $inrevision = 0; $intext = 0 } if $inpage && $inrevision && m%^\s*</revision>\s*$%;

    do { 
      print join "\t", $id, keys %c if defined $id && scalar keys %c;
      $inpage = 0; 
      $inrevision = 0; 
      $intext = 0; 
      undef $id; 
      undef %c; 
    } if $inpage && m%^\s*</page>\s*$%;
  }
