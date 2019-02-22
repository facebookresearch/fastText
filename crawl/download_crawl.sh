#!/bin/usr/env sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Set this variable to the crawl you want to process.
WET_PATHS_URL="https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2018-34/wet.paths.gz"

# Set NUM_LANGID and NUM_DEDUP according to the capacity of your machine.
# Please note that each dedup process uses 2GB of RAM, while langid is
# mostly limited by cpu usage.
NUM_LANGID=12
NUM_DEDUP=8
URL="https://commoncrawl.s3.amazonaws.com/"

if [ ! -d fastText ]; then
    git clone https://github.com/facebookresearch/fastText.git
fi

if [ ! -f fastText/fasttext ]; then
    cd fastText
    make
    cd ..
fi

if [ ! -f lid.176.bin ]; then
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi

if [ ! -d tmp ]; then
    mkdir tmp
fi

if [ ! -d shard ]; then
    mkdir shard
fi

if [ ! -f wet.paths ]; then
    wget "${WET_PATHS_URL}"
    gunzip wet.paths.gz
fi

## Language identification
cat wet.paths | xargs -n 1 -P "${NUM_LANGID}" -I '{}' sh process_wet_file.sh "${URL}{}"

## Deduplication
g++ -std=c++11 -O3 -o dedup dedup.cc
g++ -std=c++11 -O3 -o filter_utf8 filter_utf8.cc
find shard -name '*.txt' | xargs -n 1 -P "${NUM_DEDUP}" -I '{}' sh filter_dedup.sh "{}"

## Example of data filtering + tokenization
git clone https://github.com/moses-smt/mosesdecoder.git
perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l es < shard/es.dedup > shard/es.tok
