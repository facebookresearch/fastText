#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

# This script applies quantization to the models from Table 1 in:
# Bag of Tricks for Efficient Text Classification, arXiv 1607.01759, 2016

set -e

DATASET=(
  ag_news
  sogou_news
  dbpedia
  yelp_review_polarity
  yelp_review_full
  yahoo_answers
  amazon_review_full
  amazon_review_polarity
)

# These learning rates were chosen by validation on a subset of the training set.
LR=( 0.25 0.5 0.5 0.1 0.1 0.1 0.05 0.05 )

RESULTDIR=result
DATADIR=data

echo 'Warning! Make sure you run the classification-results.sh script before this one'
echo 'Otherwise you can expect the commands in this script to fail'

for i in {0..7}
do
  echo "Working on dataset ${DATASET[i]}"
  ./fasttext quantize -input "${DATADIR}/${DATASET[i]}.train" \
    -output "${RESULTDIR}/${DATASET[i]}" -lr "${LR[i]}" \
    -thread 4 -qnorm -retrain -epoch 5 -cutoff 100000 > /dev/null
  ./fasttext test "${RESULTDIR}/${DATASET[i]}.ftz" \
    "${DATADIR}/${DATASET[i]}.test"
done
