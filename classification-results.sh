#!/bin/sh
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

make

DATASET=( ag_news sogou_news dbpedia yelp_review_polarity \
  yelp_review_full yahoo_answers amazon_review_full amazon_review_polarity )

LR=( 0.25 0.5 0.5 0.1 0.1 0.1 0.05 0.05 )

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"

for i in {0..1}
do
  echo "Working on dataset ${DATASET[i]}"
  ./fasttext supervised -input "${DATADIR}/${DATASET[i]}.train" \
    -output "${RESULTDIR}/${DATASET[i]}" -dim 10 -lr "${LR[i]}" -wordNgrams 2 \
    -minCount 1 -bucket 10000000 -epoch 5 -thread 4 > /dev/null
  ./fasttext test "${RESULTDIR}/${DATASET[i]}.bin" \
    "${DATADIR}/${DATASET[i]}.test"
done
