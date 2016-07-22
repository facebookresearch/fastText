#!/bin/sh

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

make

export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

mkdir -p result

./fasttext -input data/text9 -output result/text9 -lr 0.025 -dim 50 \
  -ws 5 -epoch 1 -minCount 5 -neg 5 -sampling sqrt -loss ns -model sg \
  -bucket 2000000 -minn 3 -maxn 6 -onlyWord 0 -thread 8 -verbose 1000 \
  -t 1e-4

cut -f 1,2 data/rw/rw.txt | awk '{print tolower($0)}' | tr '\t' '\n' > data/queries.txt

cat data/queries.txt | ./print-vectors result/text9.bin > result/vectors.txt

python eval.py -m result/vectors.txt -d data/rw/rw.txt
