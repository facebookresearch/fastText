#!/bin/sh
# 
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# 

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

make

export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

mkdir -p result

./fasttext -input data/text9 -output result/text9 -lr 0.025 -dim 300 \
  -ws 5 -epoch 1 -minCount 5 -neg 5 -sampling sqrt -loss ns -model sg \
  -bucket 2000000 -minn 3 -maxn 6 -onlyWord 0 -thread 8 -verbose 1000 \
  -t 1e-4

cut -f 1,2 data/rw/rw.txt | awk '{print tolower($0)}' | tr '\t' '\n' > data/queries.txt

cat data/queries.txt | ./print-vectors result/text9.bin > result/vectors.txt

python eval.py -m result/vectors.txt -d data/rw/rw.txt
