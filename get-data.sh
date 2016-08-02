#!/bin/sh
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

normalize_text() {
  # sed -e 's/^"//g;s/"$//g;s/","/ /g' | awk '{print "__label__" $0}' | tr '[:upper:]' '[:lower:]' | \
  #        sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
  #            -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
  #            -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
  #            -e 's/«/ /g' | tr -s " " | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);'
  sed -e 's/^"//g;s/"$//g;s/","/ /g;s/,"/ /g;s/",/ /g' | tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);'
}

DATADIR=data

mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/text9" ]
then
  wget -c http://mattmahoney.net/dc/enwik9.zip -P "${DATADIR}"
  unzip "${DATADIR}/enwik9.zip" -d "${DATADIR}"
  perl wikifil.pl "${DATADIR}/enwik9" > "${DATADIR}"/text9
fi

if [ ! -f "${DATADIR}/rw/rw.txt" ]
then
  wget -c http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip -P "${DATADIR}"
  unzip "${DATADIR}/rw.zip" -d "${DATADIR}"
fi

DATASET=( ag_news amazon_review_full amazon_review_polarity dbpedia
          sogou_news yahoo_answers yelp_review_full yelp_review_polarity )
ID=( 0Bz8a_Dbh9QhbUDNpeUdjb0wxRms 0Bz8a_Dbh9QhbZVhsUnRWRDhETzA
      0Bz8a_Dbh9QhbaW12WVVZS2drcnM 0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k
      0Bz8a_Dbh9QhbUkVqNEszd0pHaFE 0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU
      0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0 0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg )

for i in {0..7}
do
  echo "Downloading dataset ${DATASET[i]}"
  if [ ! -f "${DATADIR}/${DATASET[i]}_csv/train.csv" ]
  then
    wget -c "https://googledrive.com/host/${ID[i]}" -O "${DATADIR}/${DATASET[i]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[i]}_csv.tar.gz" -C "${DATADIR}"
  fi
  cat "${DATADIR}/${DATASET[i]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.train"
  cat "${DATADIR}/${DATASET[i]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.test"
done
