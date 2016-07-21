#!/bin/sh

mkdir -p data

if [ ! -f data/text9 ]
then
  wget -c http://mattmahoney.net/dc/enwik9.zip -P data
  unzip data/enwik9.zip -d data
  perl wikifil.pl data/enwik9 > data/text9
fi

if [ ! -f data/rw/rw.txt ]
then
  wget -c http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip -P data
  unzip data/rw.zip -d data
fi
