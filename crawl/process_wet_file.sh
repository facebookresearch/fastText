#!/bin/usr/env sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

URL=$1

FILENAME=$(basename --suffix=".warc.wet.gz" "${URL}")

echo "Processing ${FILENAME}."

wget -q -P tmp "${URL}"

#echo "Extracting ${FILENAME}.warc.wet.gz"
gunzip "tmp/${FILENAME}.warc.wet.gz"

#echo "Language identification for ${FILENAME}.warc.wet"
fastText/fasttext predict-prob lid.176.bin "tmp/${FILENAME}.warc.wet" > "tmp/${FILENAME}.lid"

#echo "Splitting ${FILENAME}.warc.wet per language"
paste "tmp/${FILENAME}.lid" "tmp/${FILENAME}.warc.wet" | \
    awk '($2 > 0.8 || ($1=="__label__hr" && $2 > 0.4)) && length() > 100 {lang = substr($1, 10); $1=""; $2=""; print $0 >> "shard/"lang".txt"}'

#echo "Removing tmp files"
rm "tmp/${FILENAME}.lid"
rm "tmp/${FILENAME}.warc.wet"
