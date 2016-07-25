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
import heapq
from scipy import stats
import sys
import os
import math

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', '-m', dest='modelPath', action='store', required=True, help='path to model')
parser.add_argument('--data', '-d', dest='dataPath', action='store', required=True, help='path to data')

args = parser.parse_args()

try:
    f = open(args.modelPath, 'r')
except IOError:
    sys.exit(0)


embeds = {}
for i, line in enumerate(f):
    try:
        tab = line.decode('utf8').split()
        vec = np.array(tab[1:], dtype=float)
        word = tab[0]
        #word = tab[0].replace('í', 'i').replace('á', 'a').replace('ó', 'o').replace('ñ', 'n').replace('é', 'e').replace('ú', 'u')
        if not word in embeds:
            embeds[word] = vec
    except ValueError:
        continue
    except UnicodeDecodeError:
        continue

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def findNearest(query, embeds):
    me = 100
    for w,vec in embeds.iteritems():
        e = levenshtein(query, w)
        if e < me:
            me = e
            nw = w
            # print("{0:s} {1:s} {2:f}".format(query, w, e))
    return nw

def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    dp = np.dot(v1, v2)
    d = dp / n1 / n2
    return d

f = open(args.dataPath, 'r')

doEdit = False

mysim = []
gold = []
mysimDrop = []
goldDrop = []
drop = 0.0
nwords = 0.0

for line in f:
    zz = line.decode('utf8').split()
    z1 = zz[0].lower()
    z2 = zz[1].lower()
    score = float(zz[2])
    nwords = nwords + 1.0

    if (z1 in embeds) and (z2 in embeds):
        v1 = embeds[z1]
        v2 = embeds[z2]
        d = similarity(v1, v2)
        mysim.append(d)
        gold.append(float(zz[2]))
    elif (doEdit):
        if (z1 in embeds):
            w1 = z1
        else:
            w1 = findNearest(z1, embeds)

        if (z2 in embeds):
            w2 = z2
        else:
            w2 = findNearest(z2, embeds)
        v1 = embeds[w1]
        v2 = embeds[w2]
        d = similarity(v1, v2)
        mysimDrop.append(d)
        goldDrop.append(score)
        drop = drop + 1.0
        sys.stdout.write(str(drop) + " ")
        sys.stdout.flush()
    else:
        drop = drop + 1.0

pr = stats.spearmanr(mysim, gold)
dataset = os.path.basename(args.dataPath)
print("{0:20s} & {2:2.0f}\% & {1:2.0f}".format(dataset, pr[0] * 100, math.ceil(drop / nwords * 100.0)))
