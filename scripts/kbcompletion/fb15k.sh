#!/usr/bin/env bash
#
# copyright (c) 2017-present, facebook, inc.
# all rights reserved.
#
# this source code is licensed under the MIT license found in the
# license file in the root directory of this source tree.
#
# script for FB15k
DIR=data/FB15k/
FASTTEXTDIR=../../

# compile
pushd $FASTTEXTDIR
make opt
popd
ft=${FASTTEXTDIR}/fasttext

g++ -std=c++0x eval.cpp -o eval

## Train model and test it on validation:
dim=100
epoch=100
neg=100
model=data/fb15
pred=data/fbpred

echo "---- train ----"
$ft supervised -input $DIR/ft_freebase_mtr100_mte100-train.txt \
  -dim $dim -epoch $epoch -output ${model} -lr .2 -thread 20 -loss ns -neg $neg -minCount 0

echo "computing raw hits@10..."
$ft test ${model}.bin $DIR/ft_freebase_mtr100_mte100-test.txt 10 2> /dev/null | awk '{if(NR==3) print "raw hit@10="$2}'

echo "computing filtered hit@10..."
$ft predict ${model}.bin $DIR/ft_freebase_mtr100_mte100-test.txt 20000 > $pred
./eval $pred ${DIR}/ft_freebase_mtr100_mte100-test.txt $DIR/ft_freebase_mtr100_mte100-full.txt 10 | awk '{if(NR==2) print "filtered hit@10="$2}'

echo  "---- train+val ----"

$ft supervised -input $DIR/ft_freebase_mtr100_mte100-valid+train.txt \
  -dim ${dim} -epoch ${dim} -output ${model} -lr .2 -thread 20 -loss ns -neg ${neg} -minCount 0

echo "computing raw hits@10..."
$ft test ${model}.bin $DIR/ft_freebase_mtr100_mte100-test.txt 10  2> /dev/null | awk '{if(NR==3) print "raw hit@10="$2}'

echo "computing filtered hit@10..."
$ft predict ${model}.bin $DIR/ft_freebase_mtr100_mte100-test.txt 20000 > $pred
./eval $pred ${DIR}/ft_freebase_mtr100_mte100-test.txt $DIR/ft_freebase_mtr100_mte100-full.txt 10 | awk '{if(NR==2) print "filtered hit@10="$2}'
