#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from scipy import stats
import sys
import os
import math
import argparse

def compat_splitting(line):
    return line.decode('utf8').split()

def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', '-m', dest='modelPath', action='store', required=True, help='path to model')
parser.add_argument('--data', '-d', dest='dataPath', action='store', required=True, help='path to data')
args = parser.parse_args()

vectors = {}
fin = open(args.modelPath, 'rb')
for i, line in enumerate(fin):
    try:
        tab = compat_splitting(line)
        vec = np.array(tab[1:], dtype=float)
        word = tab[0]
        if np.linalg.norm(vec) == 0:
            continue
        if not word in vectors:
            vectors[word] = vec
    except ValueError:
        continue
    except UnicodeDecodeError:
        continue
fin.close()

mysim = []
gold = []
drop = 0.0
nwords = 0.0

fin = open(args.dataPath, 'rb')
for line in fin:
    tline = compat_splitting(line)
    word1 = tline[0].lower()
    word2 = tline[1].lower()
    nwords = nwords + 1.0

    if (word1 in vectors) and (word2 in vectors):
        v1 = vectors[word1]
        v2 = vectors[word2]
        d = similarity(v1, v2)
        mysim.append(d)
        gold.append(float(tline[2]))
    else:
        drop = drop + 1.0
fin.close()

corr = stats.spearmanr(mysim, gold)
dataset = os.path.basename(args.dataPath)
print("{0:20s}: {1:2.0f}  (OOV: {2:2.0f}%)"
      .format(dataset, corr[0] * 100, math.ceil(drop / nwords * 100.0)))
