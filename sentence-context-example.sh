#!/usr/bin/env bash

RESULTDIR=result
DATADIR=/Users/jaanaltosaar/installations/models/food2vec/

mkdir -p "${RESULTDIR}"

make

./fasttext sentence_context -input "${DATADIR}"/dat/processed/train -output "${RESULTDIR}"/model -lr 0.025 -dim 100 \
  -ws 0 -epoch 1 -minCount 5 -neg 5 -loss ns -bucket 2000000 \
  -wordNgrams 0 \
  -minn 0 -maxn 0 -thread 8 -t 1 -lrUpdateRate 100

tail -n+3 $RESULTDIR/model.vec > $RESULTDIR/vectors.txt

python eval_interactive.py -m "${RESULTDIR}"/vectors.txt
