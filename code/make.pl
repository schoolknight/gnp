#!/usr/bin/perl -w

system("gcc -O3 -lm -o gnp gnp.c simplex_downhill.c");
print "The executable is \"gnp\"\n";

