#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

DATADIR=${DATADIR:-data}

report_error() {
   echo "Error on line $1 of $0"
}

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

set -e
trap 'report_error $LINENO' ERR

mkdir -p "${DATADIR}"


# Unsupervised datasets

data_result="${DATADIR}/rw_queries.txt"
if [ ! -f "$data_result" ]
then
  cut -f 1,2 "${DATADIR}"/rw/rw.txt | awk '{print tolower($0)}' | tr '\t' '\n' > "$data_result" || rm -f "$data_result"
fi

data_result="${DATADIR}/enwik9.zip"
if [ ! -f "$data_result" ] || \
   [ $(md5sum "$data_result" | cut -f 1 -d ' ') != "3e773f8a1577fda2e27f871ca17f31fd" ]
then
  wget -c http://mattmahoney.net/dc/enwik9.zip -P "${DATADIR}" || rm -f "$data_result"
  unzip "$data_result" -d "${DATADIR}" || rm -f "$data_result"
fi

data_result="${DATADIR}/fil9"
if [ ! -f "$data_result" ]
then
  perl wikifil.pl "${DATADIR}/enwik9" > "$data_result" || rm -f "$data_result"
fi

data_result="${DATADIR}/rw/rw.txt"
if [ ! -f "$data_result" ]
then
  wget -c https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip -P "${DATADIR}"
  unzip "${DATADIR}/rw.zip" -d "${DATADIR}" || rm -f "$data_result"
fi

# Supervised datasets
# Each datasets comes with a .train and a .test to measure performance

echo "Downloading dataset dbpedia"

data_result="${DATADIR}/dbpedia_csv.tar.gz"
if [ ! -f "$data_result" ] || \
   [ $(md5sum "$data_result" | cut -f 1 -d ' ') != "8139d58cf075c7f70d085358e73af9b3" ]
then
  wget -c "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz" -O "$data_result"
  tar -xzvf "$data_result" -C "${DATADIR}"
fi

data_result="${DATADIR}/dbpedia.train"
if [ ! -f "$data_result" ]
then
  cat "${DATADIR}/dbpedia_csv/train.csv" | normalize_text > "$data_result" || rm -f "$data_result"
fi

data_result="${DATADIR}/dbpedia.test"
if [ ! -f "$data_result" ]
then
  cat "${DATADIR}/dbpedia_csv/test.csv" | normalize_text > "$data_result" || rm -f "$data_result"
fi

echo "Downloading dataset tatoeba for langid"

data_result="${DATADIR}"/langid/all.txt
if [ ! -f "$data_result" ]
then
  mkdir -p "${DATADIR}"/langid
  wget http://downloads.tatoeba.org/exports/sentences.tar.bz2 -O "${DATADIR}"/langid/sentences.tar.bz2
  tar xvfj "${DATADIR}"/langid/sentences.tar.bz2 --directory "${DATADIR}"/langid || exit 1
  awk -F"\t" '{print"__label__"$2" "$3}' < "${DATADIR}"/langid/sentences.csv | shuf > "$data_result"
fi

data_result="${DATADIR}/langid.train"
if [ ! -f "$data_result" ]
then
  tail -n +10001 "${DATADIR}"/langid/all.txt > "$data_result"
fi

data_result="${DATADIR}/langid.valid"
if [ ! -f "$data_result" ]
then
  head -n 10000 "${DATADIR}"/langid/all.txt > "$data_result"
fi

echo "Downloading cooking dataset"

data_result="${DATADIR}"/cooking/cooking.stackexchange.txt
if [ ! -f "$data_result" ]
then
  mkdir -p "${DATADIR}"/cooking/
  wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz -O "${DATADIR}"/cooking/cooking.stackexchange.tar.gz
  tar xvzf "${DATADIR}"/cooking/cooking.stackexchange.tar.gz --directory "${DATADIR}"/cooking || exit 1
  cat "${DATADIR}"/cooking/cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > "${DATADIR}"/cooking/cooking.preprocessed.txt
fi

data_result="${DATADIR}"/cooking.train
if [ ! -f "$data_result" ]
then
  head -n 12404 "${DATADIR}"/cooking/cooking.preprocessed.txt > "${DATADIR}"/cooking.train
fi

data_result="${DATADIR}"/cooking.valid
if [ ! -f "$data_result" ]
then
  tail -n 3000 "${DATADIR}"/cooking/cooking.preprocessed.txt > "${DATADIR}"/cooking.valid
fi

echo "Checking for YFCC100M"

data_result="${DATADIR}"/YFCC100M/train
if [ ! -f "$data_result" ]
then
  echo 'Download YFCC100M, unpack it and place train into the following path: '"$data_result"
  echo 'You can download YFCC100M at :'"https://fasttext.cc/docs/en/dataset.html"
  echo 'After you download this, run the script again'
  exit 1
fi

data_result="${DATADIR}"/YFCC100M/test
if [ ! -f "$data_result" ]
then
  echo 'Download YFCC100M, unpack it and place test into the following path: '"$data_result"
  echo 'You can download YFCC100M at :'"https://fasttext.cc/docs/en/dataset.html"
  echo 'After you download this, run the script again'
  exit 1
fi

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

ID=(
  0Bz8a_Dbh9QhbUDNpeUdjb0wxRms # ag_news
  0Bz8a_Dbh9QhbUkVqNEszd0pHaFE # sogou_news
  0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k # dbpedia
  0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg # yelp_review_polarity
  0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0 # yelp_review_full
  0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU # yahoo_answers
  0Bz8a_Dbh9QhbZVhsUnRWRDhETzA # amazon_review_full
  0Bz8a_Dbh9QhbaW12WVVZS2drcnM # amazon_review_polarity
)

# Small datasets first

for i in {0..0}
do
  echo "Downloading dataset ${DATASET[i]}"
  if [ ! -f "${DATADIR}/${DATASET[i]}.train" ]
  then
    wget -c "https://drive.google.com/uc?export=download&id=${ID[i]}" -O "${DATADIR}/${DATASET[i]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[i]}_csv.tar.gz" -C "${DATADIR}"
    cat "${DATADIR}/${DATASET[i]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.train"
    cat "${DATADIR}/${DATASET[i]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.test"
  fi
done

# Large datasets require a bit more work due to the extra request page

for i in {1..7}
do
  echo "Downloading dataset ${DATASET[i]}"
  if [ ! -f "${DATADIR}/${DATASET[i]}.train" ]
  then
    curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${ID[i]}" > /tmp/intermezzo.html
    curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > "${DATADIR}/${DATASET[i]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[i]}_csv.tar.gz" -C "${DATADIR}"
    cat "${DATADIR}/${DATASET[i]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.train"
    cat "${DATADIR}/${DATASET[i]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.test"
  fi
done
