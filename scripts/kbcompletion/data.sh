#!/usr/bin/env bash
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
set -e
DATADIR=data/

if [ ! -d "$DATADIR" ]; then
  mkdir $DATADIR
fi

cd $DATADIR
echo "preparing WN18"
#wget -P . https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz
#mv fetch.php\?media\=en\:wordnet-mlj12.tar.gz wordnet-mlj12.tar.gz
wget -P . https://github.com/mana-ysh/knowledge-graph-embeddings/raw/master/dat/wordnet-mlj12.tar.gz
tar -xzvf wordnet-mlj12.tar.gz
DIR=wordnet-mlj12
for f in ${DIR}/wordnet-ml*.txt;
do
  fn=${DIR}/ft_$(basename $f)
  awk '{print "__label__"$1,"0_"$2, $3;print $1,"1_"$2," __label__"$3}' < ${f} > ${fn};
done
cat ${DIR}/ft_* > ${DIR}/ft_wordnet-mlj12-full.txt
cat ${DIR}/ft_*train.txt ${DIR}/ft_*valid.txt > ${DIR}/ft_wordnet-mlj12-valid+train.txt

echo "preparing FB15K"
#wget https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz
#mv fetch.php\?media\=en\:fb15k.tgz fb15k.tgz
wget https://github.com/mana-ysh/knowledge-graph-embeddings/raw/master/dat/fb15k.tgz
tar -xzvf fb15k.tgz
DIR=FB15k/
for f in ${DIR}/freebase*.txt;
do
  fn=${DIR}/ft_$(basename $f)
  echo $f " --> " $fn
  awk '{print "__label__"$1,"0_"$2, $3;print $1,"1_"$2," __label__"$3}' < ${f} > ${fn};
done
cat ${DIR}/ft_* > ${DIR}/ft_freebase_mtr100_mte100-full.txt
cat ${DIR}/ft_*train.txt ${DIR}/ft_*valid.txt > ${DIR}/ft_freebase_mtr100_mte100-valid+train.txt

echo "preparing FB15K-237"
wget https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip
unzip FB15K-237.2.zip
DIR=Release/
for f in train.txt test.txt valid.txt
do
  fn=${DIR}/ft_$(basename $f)
  echo $f " --> " $fn
  awk -F "\t" '{print "__label__"$1,"0_"$2, $3;print $1,"1_"$2," __label__"$3}' < ${DIR}/${f} > ${fn};
done
cat ${DIR}/ft_*.txt > ${DIR}/ft_full.txt
cat ${DIR}/ft_train.txt ${DIR}/ft_valid.txt > ${DIR}/ft_valid+train.txt

echo "preparing SVO"
wget . https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:svo-tensor-dataset.tar.gz
mv fetch.php?media=en:svo-tensor-dataset.tar.gz svo-tensor-dataset.tar.gz
tar -xzvf svo-tensor-dataset.tar.gz
DIR=SVO-tensor-dataset
for f in ${DIR}/svo_data*.dat;
do
  fn=${DIR}/ft_$(basename $f)
  awk '{print "0_"$1,"1_"$3,"__label__"$2;}' < ${f} > ${fn};
done
cat ${DIR}/ft_*train*.dat ${DIR}/ft_*valid*.dat > ${DIR}/ft_svo_data-valid+train.dat
