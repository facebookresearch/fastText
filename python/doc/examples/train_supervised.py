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

import os
from fastText import train_supervised
from fastText.util import test

if __name__ == "__main__":
    train_data = os.path.join(os.getenv("DATADIR", ''), 'cooking.train')
    valid_data = os.path.join(os.getenv("DATADIR", ''), 'cooking.valid')
    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=1, minCount=1
    )
    true_labels = []
    all_words = []
    with open(valid_data, 'r') as fid:
        for line in fid:
            words, labels = model.get_line(line.strip())
            all_words.append(" ".join(words))
            true_labels += [labels]
    predictions, _ = model.predict(all_words)
    p, r = test(predictions, true_labels)
    print("N\t" + str(len(predictions)))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    model.save_model("cooking.bin")
