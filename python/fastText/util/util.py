# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# NOTE: The purpose of this file is not to accumulate all useful utility
# functions. This file should contain very commonly used and requested functions
# (such as test). If you think you have a function at that level, please create
# an issue and we will happily review your suggestion. This file is also not supposed
# to pull in dependencies outside of numpy/scipy without very good reasons. For
# example, this file should not use sklearn and matplotlib to produce a t-sne
# plot of word embeddings or such.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


# TODO: Add example on reproducing model.test with util.test and model.get_line
def test(predictions, labels, k=1):
    """
    Return precision and recall modeled after fasttext's test
    """
    precision = 0.0
    nexamples = 0
    nlabels = 0
    for prediction, labels in zip(predictions, labels):
        for p in prediction:
            if p in labels:
                precision += 1
        nexamples += 1
        nlabels += len(labels)
    return (precision / (k * nexamples), precision / nlabels)


def find_nearest_neighbor(query, vectors, ban_set, cossims=None):
    """
    query is a 1d numpy array corresponding to the vector to which you want to
    find the closest vector
    vectors is a 2d numpy array corresponding to the vectors you want to consider
    ban_set is a set of indicies within vectors you want to ignore for nearest match
    cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency

    returns the index of the closest match to query within vectors

    """
    if cossims is None:
        cossims = np.matmul(vectors, query, out=cossims)
    else:
        np.matmul(vectors, query, out=cossims)
    rank = len(cossims) - 1
    result_i = np.argpartition(cossims, rank)[rank]
    while result_i in ban_set:
        rank -= 1
        result_i = np.argpartition(cossims, rank)[rank]
    return result_i
