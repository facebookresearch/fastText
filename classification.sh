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

export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"

./fasttext supervised -input "${DATADIR}/yelp_review_full.train" -output "${RESULTDIR}/yelp_review_full" -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4

./fasttext test "${RESULTDIR}/yelp_review_full.bin" "${DATADIR}/yelp_review_full.test"

./fasttext predict "${RESULTDIR}/yelp_review_full.bin" "${DATADIR}/yelp_review_full.test" > "${RESULTDIR}/yelp_review_full.test.predict"
