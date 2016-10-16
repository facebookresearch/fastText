#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"


if [ ! -f "${DATADIR}/dbpedia.train" ]
then
  wget -c "https://googledrive.com/host/0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k" -O "${DATADIR}/dbpedia_csv.tar.gz"
  tar -xzvf "${DATADIR}/dbpedia_csv.tar.gz" -C "${DATADIR}"
  cat "${DATADIR}/dbpedia_csv/train.csv" | normalize_text > "${DATADIR}/dbpedia.train"
  cat "${DATADIR}/dbpedia_csv/test.csv" | normalize_text > "${DATADIR}/dbpedia.test"
fi

make


if [ "$1" = "" ] ; then
    SPLIT_SIZE=10000
else
    SPLIT_SIZE=$1
fi

echo "======================== BEGIN: incremental learning demo ============================"
echo " split size = $SPLIT_SIZE  (by line)"
echo "======================================================================================"
mkdir -p "${DATADIR}/split"


rm "${DATADIR}/split"/* "${RESULTDIR}/dbpedia."[0-9]*.bin
cd "${DATADIR}/split"

split ../dbpedia.train -l $SPLIT_SIZE -a 3 -d dbpedia.train.
for i in dbpedia.train.* ; do 
    echo item: "$i"
done
cd -

COUNTER=0
for i in ${DATADIR}/split/dbpedia.train.* ; do
    echo "processing: $i"

    if [ "$i" = "${DATADIR}/split/dbpedia.train.000" ] ; then
        echo "./fasttext supervised -input $i -output ${RESULTDIR}/dbpedia.base -dim 10 -lr 0.08 -wordNgrams 2 -minCount 1 -bucket 1000000 -epoch 5 -thread 4"
          time ./fasttext supervised -input "$i" -output "${RESULTDIR}/dbpedia.base" -dim 10 -lr 0.08 -wordNgrams 2 -minCount 1 -bucket 1000000 -epoch 5 -thread 4
    else
        echo "./fasttext supervised-append ${RESULTDIR}/dbpedia."$[$COUNTER - 1]" $i" 0.07 
        time ./fasttext supervised-append "${RESULTDIR}/dbpedia.base" "$i"  0.07
    fi

    cp "${RESULTDIR}/dbpedia.base.bin" "${RESULTDIR}/dbpedia.${COUNTER}.bin"
    COUNTER=$[$COUNTER + 1]

    ./fasttext test "${RESULTDIR}/dbpedia.base.bin" "${DATADIR}/dbpedia.test"

done

./fasttext predict "${RESULTDIR}/dbpedia.base.bin" "${DATADIR}/dbpedia.test" > "${RESULTDIR}/dbpedia.test.predict"

