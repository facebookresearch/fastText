#!/bin/sh
# 
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# 

DATADIR=data

mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}"/text9 ]
then
  wget -c http://mattmahoney.net/dc/enwik9.zip -P "${DATADIR}"
  unzip "${DATADIR}"/enwik9.zip -d "${DATADIR}"
  perl wikifil.pl "${DATADIR}"/enwik9 > "${DATADIR}"/text9
fi

if [ ! -f "${DATADIR}"/rw/rw.txt ]
then
  wget -c http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip -P "${DATADIR}"
  unzip "${DATADIR}"/rw.zip -d "${DATADIR}"
fi
