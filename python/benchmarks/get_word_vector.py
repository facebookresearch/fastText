# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from fastText import load_model
from fastText import tokenize
import sys
import time
import tempfile
import argparse


def get_word_vector(data, model):
    t1 = time.time()
    print("Reading")
    with open(data, 'r') as f:
        tokens = tokenize(f.read())
    t2 = time.time()
    print("Read TIME: " + str(t2 - t1))
    print("Read NUM : " + str(len(tokens)))
    f = load_model(model)
    # This is not equivalent to piping the data into
    # print-word-vector, because the data is tokenized
    # first.
    t3 = time.time()
    i = 0
    for t in tokens:
        vec = f.get_word_vector(t)
        i += 1
        if i % 10000 == 0:
            sys.stderr.write("\ri: " + str(float(i / len(tokens))))
            sys.stderr.flush()
    t4 = time.time()
    print("\nVectoring: " + str(t4 - t3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple benchmark for get_word_vector.')
    parser.add_argument('model', help='A model file to use for benchmarking.')
    parser.add_argument('data', help='A data file to use for benchmarking.')
    args = parser.parse_args()
    get_word_vector(args.data, args.model)
