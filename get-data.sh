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
  sed -e 's/^"//g;s/"$//g;s/","/ /g' | awk '{print "__label__" $0}' | awk '{print tolower($0);}' | \
         sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
             -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
             -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
             -e 's/«/ /g' | tr -s " "
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

if [ ! -f "${DATADIR}/ag_news.train" ]
then
  wget 'https://googledrive.com/host/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms' -O "${DATADIR}/ag_news_csv.tar.gz"
  tar -xzvf "${DATADIR}/ag_news_csv.tar.gz" -C "${DATADIR}"
  cat "${DATADIR}/ag_news_csv/train.csv" | normalize_text > "${DATADIR}/ag_news.train"
  cat "${DATADIR}/ag_news_csv/test.csv" | normalize_text > "${DATADIR}/ag_news.test"
fi

if [ ! -f "${DATADIR}/yelp_review_full" ]
then
  wget 'https://googledrive.com/host/0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0' -O "${DATADIR}/yelp_review_full_csv.tar.gz"
  tar -xzvf "${DATADIR}/yelp_review_full_csv.tar.gz" -C "${DATADIR}"
  cat "${DATADIR}/yelp_review_full_csv/train.csv" | normalize_text > "${DATADIR}/yelp_review_full.train"
  cat "${DATADIR}/yelp_review_full_csv/test.csv" | normalize_text > "${DATADIR}/yelp_review_full.test"
fi


