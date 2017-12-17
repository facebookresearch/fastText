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


# Return top-k predictions and probabilities for each line in the given file.
def get_predictions(filename, model, k=1):
    predictions = []
    probabilities = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            labels, probs = model.predict(line, k)
            predictions.append(labels)
            probabilities.append(probs)
    return predictions, probabilities


# Parse and return list of labels
def get_labels_from_file(filename, prefix="__label__"):
    labels = []
    with open(filename) as f:
        for line in f:
            line_labels = []
            tokens = line.split()
            for token in tokens:
                if token.startswith(prefix):
                    line_labels.append(token)
            labels.append(line_labels)
    return labels


if __name__ == "__main__":
    train_data = os.path.join(os.getenv("DATADIR", ''), 'cooking.train')
    valid_data = os.path.join(os.getenv("DATADIR", ''), 'cooking.valid')
    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )
    k = 1
    predictions, _ = get_predictions(valid_data, model, k=k)
    valid_labels = get_labels_from_file(valid_data)
    p, r = test(predictions, valid_labels, k=k)
    print("N\t" + str(len(valid_labels)))
    print("P@{}\t{:.3f}".format(k, p))
    print("R@{}\t{:.3f}".format(k, r))
    model.save_model(train_data + '.bin')
