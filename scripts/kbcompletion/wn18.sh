#!/usr/bin/env bash
#
# copyright (c) 2017-present, facebook, inc.
# all rights reserved.
#
# this source code is licensed under the MIT license found in the
# license file in the root directory of this source tree.
#
# script for WN11
DIR=data/wordnet-mlj12/
FASTTEXTDIR=../../

# compile

pushd $FASTTEXTDIR
make opt
popd
ft=${FASTTEXTDIR}/fasttext

g++ -std=c++0x eval.cpp -o eval

# Train model and test it:
dim=100
epoch=100
neg=500
model=data/wn
pred=data/wnpred

echo  "---- train ----"
$ft supervised -input ${DIR}/ft_wordnet-mlj12-train.txt  \
  -dim $dim -epoch $epoch -output ${model} -lr .2 -thread 20 -loss ns -neg $neg

echo "computing raw hits@10..."
$ft test ${model}.bin ${DIR}/ft_wordnet-mlj12-test.txt 10 2> /dev/null | awk '{if(NR==3) print "raw hit@10 = "$2}'

echo "computing filtered hit@10..."
$ft predict ${model}.bin ${DIR}/ft_wordnet-mlj12-test.txt 20000 > $pred
./eval $pred ${DIR}/ft_wordnet-mlj12-test.txt $DIR/ft_wordnet-mlj12-full.txt 10 | awk '{if(NR==2) print "filtered hit@10 = "$2}'

echo  "---- train+val ----"
$ft supervised -input ${DIR}/ft_wordnet-mlj12-valid+train.txt \
  -dim $dim -epoch $epoch -output ${model} -lr .2 -thread 20 -loss ns -neg $neg

echo "computing raw hits@10..."
$ft test ${model}.bin ${DIR}/ft_wordnet-mlj12-test.txt 10  2> /dev/null | awk '{if(NR==3) print "raw hit@10 = "$2}'

echo "computing filtered hit@10..."
$ft predict ${model}.bin ${DIR}/ft_wordnet-mlj12-test.txt 20000 > $pred
./eval $pred ${DIR}/ft_wordnet-mlj12-test.txt $DIR/ft_wordnet-mlj12-full.txt 10 | awk '{if(NR==2) print "filtered hit@10 = "$2}'
