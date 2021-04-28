#!/usr/bin/env bash
#
# copyright (c) 2017-present, facebook, inc.
# all rights reserved.
#
# this source code is licensed under the MIT license found in the
# license file in the root directory of this source tree.
#
# script for SVO
DIR=data/SVO-tensor-dataset
FASTTEXTDIR=../../

# compile
pushd $FASTTEXTDIR
make opt
popd
ft=${FASTTEXTDIR}/fasttext

## Train model and test it on validation:

dim=200
epoch=3
model=svo

echo  "---- train ----"
time $ft supervised -input ${DIR}/ft_svo_data_train_1000000.dat  \
  -dim $dim -epoch $epoch -output ${model} -lr .2 -thread 20

echo "computing raw hit@5%..."
$ft test ${model}.bin ${DIR}/ft_svo_data_test_250000.dat 227 2> /dev/null | awk '{if(NR==3) print "raw hit@5%="$2}'


echo  "---- train + valid ----"
time $ft supervised -input ${DIR}/ft_svo_data-valid+train.dat  \
  -dim $dim -epoch $epoch -output ${model} -lr .2 -thread 20

echo "computing raw hit@5%..."
$ft test ${model}.bin ${DIR}/ft_svo_data_test_250000.dat 227 2> /dev/null | awk '{if(NR==3) print "raw hit@5%="$2}'
