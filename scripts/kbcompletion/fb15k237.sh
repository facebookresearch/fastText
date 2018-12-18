#!/usr/bin/env bash
#
# copyright (c) 2017-present, facebook, inc.
# all rights reserved.
#
# this source code is licensed under the MIT license found in the
# license file in the root directory of this source tree.
#
# script for FB15k237
DIR=data/Release/
FASTTEXTDIR=../../

# compile

pushd $FASTTEXTDIR
make opt
popd
ft=${FASTTEXTDIR}/fasttext

g++ -std=c++0x eval.cpp -o eval

## Train model and test it on validation:

pred=data/fb237pred
model=data/fb15k237
dim=50
epoch=10
neg=500

echo "---- train ----"
$ft supervised -input $DIR/ft_train.txt \
  -dim $dim -epoch $epoch -output ${model} -lr .2 -thread 20 -loss ns -neg $neg -minCount 0

echo "computing filtered hit@10..."
$ft predict ${model}.bin $DIR/ft_test.txt 20000 > $pred
./eval $pred ${DIR}/ft_test.txt $DIR/ft_full.txt 10 | awk '{if(NR==2) print "filtered hit@10="$2}'

echo  "---- train+val ----"

$ft supervised -input $DIR/ft_valid+train.txt \
  -dim ${dim} -epoch ${dim} -output ${model} -lr .2 -thread 20 -loss ns -neg ${neg} -minCount 0

echo "computing filtered hit@10..."
$ft predict ${model}.bin $DIR/ft_test.txt 20000 > $pred
./eval $pred ${DIR}/ft_test.txt $DIR/ft_full.txt 10 | awk '{if(NR==2) print "filtered hit@10="$2}'
