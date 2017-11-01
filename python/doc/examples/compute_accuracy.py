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
import numpy as np


def closest_ind(query, vectors, cossims):
    np.matmul(vectors, query, out=cossims)


def print_score(
    question, correct, num_qs, total_accuracy, semantic_accuracy, syntactic_accuracy
):
    print(
        (
            "{0:>30}: ACCURACY TOP1: {3:.2f} %  ({1} / {2})\t  Total accuracy: {4:.2f} %   Semantic accuracy: {5:.2f} %   Syntactic accuracy: {6:.2f} %"
        ).format(
            question, correct, num_qs, correct / float(num_qs) * 100, total_accuracy * 100,
            semantic_accuracy * 100, syntactic_accuracy * 100,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "compute_accuracy equivalent in Python. "
            "See https://github.com/tmikolov/word2vec/blob/master/demo-word-accuracy.sh"
        )
    )
    parser.add_argument(
        "model",
        help="Model to use",
    )
    parser.add_argument(
        "question_words",
        help="word questions similar to tmikolov's file (see help for link)",
    )
    parser.add_argument(
        "threshold",
        help="threshold used to limit number of words used",
    )
    args = parser.parse_args()
    args.threshold = int(args.threshold)

    f = load_model(args.model)
    words, freq = f.get_words(include_freq=True)
    words = words[:args.threshold]
    vectors = np.zeros((len(words), f.get_dimension()), dtype=float)
    for i in range(len(words)):
        wv = f.get_word_vector(words[i])
        wv = wv / np.linalg.norm(wv)
        vectors[i] = wv

    total_correct = 0
    total_qs = 0
    num_lines = 0

    total_se_correct = 0
    total_se_qs = 0

    total_sy_correct = 0
    total_sy_qs = 0

    qid = 0
    with open(args.question_words, 'r') as fqw:
        correct = 0
        num_qs = 0
        question = ""
        # For efficiency
        cossims = np.zeros(len(words), dtype=float)
        for line in fqw:
            if line[0] == ":":
                if question != "":
                    total_qs += num_qs
                    total_correct += correct
                    score = correct / num_qs
                    if (qid <= 5):
                        total_se_correct += correct
                        total_se_qs += num_qs
                    else:
                        total_sy_correct += correct
                        total_sy_qs += num_qs
                    print_score(
                        question,
                        correct,
                        num_qs,
                        total_correct / float(total_qs),
                        total_se_correct / float(total_se_qs) if total_se_qs > 0 else 0,
                        total_sy_correct / float(total_sy_qs) if total_sy_qs > 0 else 0,
                    )
                correct = 0
                num_qs = 0
                question = line.strip().replace(":", "")
                qid += 1
            else:
                num_lines += 1
                qwords = line.split()
                qwords = [x.lower().strip() for x in qwords]
                found = True
                for w in qwords:
                    if w not in words:
                        found = False
                        break
                if not found:
                    continue
                query = qwords[:3]
                query = [f.get_word_vector(x) for x in query]
                query = [x / np.linalg.norm(x) for x in query]
                query = query[1] - query[0] + query[2]
                ban_set = qwords[:3]
                closest_ind(query, vectors, cossims)
                rank = len(cossims) - 1
                result_i = np.argpartition(cossims, rank)[rank]
                result = words[result_i]
                while result in ban_set:
                    rank -= 1
                    result_i = np.argpartition(cossims, rank)[rank]
                    result = words[result_i]
                if result == qwords[3]:
                    correct += 1
                num_qs += 1

    total_qs += num_qs
    total_correct += correct
    total_sy_correct += correct
    total_sy_qs += num_qs
    print_score(
        question,
        correct,
        num_qs,
        total_correct / float(total_qs),
        total_se_correct / float(total_se_qs) if total_se_qs > 0 else 0,
        total_sy_correct / float(total_sy_qs) if total_sy_qs > 0 else 0,
    )
    print(
            "Questions seen / total: {0} {1}   {2:.2f} %".format(total_qs, num_lines,
        total_qs / num_lines * 100)
    )
