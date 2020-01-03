#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from fasttext import train_supervised


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


if __name__ == "__main__":
    train_data = os.path.join(os.getenv("DATADIR", ''), 'cooking.train')
    valid_data = os.path.join(os.getenv("DATADIR", ''), 'cooking.valid')

    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )
    print_results(*model.test(valid_data))

    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1,
        loss="hs"
    )
    print_results(*model.test(valid_data))
    model.save_model("cooking.bin")

    model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    print_results(*model.test(valid_data))
    model.save_model("cooking.ftz")
