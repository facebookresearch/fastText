#!/usr/bin/env bash

RESULTDIR=result
DATADIR=/Users/jaanaltosaar/installations/models/food2vec/

mkdir -p "${RESULTDIR}"

make

./fasttext sentence_context -input "${DATADIR}"/dat/processed/kaggle_and_nature_train -output "${RESULTDIR}"/model -lr 0.025 -dim 100 \
  -ws 0 -epoch 3 -minCount 5 -neg 5 -loss ns -bucket 2000000 \
  -wordNgrams 0 \
  -minn 0 -maxn 0 -thread 8 -t 1 -lrUpdateRate 100

cat "${DATADIR}"/fit/nature_and_kaggle_vocab_words.txt | ./fasttext print-vectors "${RESULTDIR}"/model.bin > "${RESULTDIR}"/vectors.txt

python eval_interactive.py -m "${RESULTDIR}"/vectors.txt
