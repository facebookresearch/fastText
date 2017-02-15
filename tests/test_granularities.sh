#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

RESULTDIR=result
DATADIR=.

DATAFILE=data0

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

make -C ..

GRANULARITIES=3

../fasttext supervised -input "${DATADIR}/${DATAFILE}.train" -output "${RESULTDIR}/${DATAFILE}" -dim 10 -granularities ${GRANULARITIES} -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4

../fasttext test "${RESULTDIR}/${DATAFILE}.bin" "${DATADIR}/${DATAFILE}.test" ${GRANULARITIES}

../fasttext predict "${RESULTDIR}/${DATAFILE}.bin" "${DATADIR}/${DATAFILE}.test" > "${RESULTDIR}/${DATAFILE}.test.predict" ${GRANULARITIES}
