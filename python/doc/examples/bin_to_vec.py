#!/usr/bin/env python

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
from __future__ import division, absolute_import, print_function

from fastText import load_model
import argparse
import errno

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print fasttext .vec file to stdout from .bin file"
        )
    )
    parser.add_argument(
        "model", help="Model to use",
    )
    args = parser.parse_args()

    f = load_model(args.model)
    words = f.get_words()
    print(str(len(words)) + " " + str(f.get_dimension()))
    for w in words:
        v = f.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            print(w + vstr)
        except IOError as e:
            if e.errno == errno.EPIPE:
                pass
