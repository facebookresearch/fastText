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
from __future__ import division, absolute_import, print_function

from fasttext import load_model
from fasttext import util
import argparse
import numpy as np


def process_question(question, cossims, model, words, vectors):
    correct = 0
    num_qs = 0
    num_lines = 0
    for line in question:
        num_lines += 1
        qwords = line.split()
        # We lowercase all words to correspond to the preprocessing
        # we applied to our data.
        qwords = [x.lower().strip() for x in qwords]
        # If one of the words is not in the vocabulary we skip this question
        found = True
        for w in qwords:
            if w not in words:
                found = False
                break
        if not found:
            continue
        # The first three words form the query
        # We retrieve their word vectors and normalize them
        query = qwords[:3]
        query = [model.get_word_vector(x) for x in query]
        query = [x / np.linalg.norm(x) for x in query]
        # Get the query vector. Example:
        # Germany  - Berlin + France
        query = query[1] - query[0] + query[2]
        # We don't need to rank all the words, only until we found
        # the first word not equal to our set of query words.
        ban_set = list(map(lambda x: words.index(x), qwords[:3]))
        if words[util.find_nearest_neighbor(
            query, vectors, ban_set, cossims=cossims
        )] == qwords[3]:
            correct += 1
        num_qs += 1
    return correct, num_qs, num_lines


# We use the same conventions as within compute-accuracy
def print_compute_accuracy_score(
    question, correct, num_qs, total_accuracy, semantic_accuracy,
    syntactic_accuracy
):
    print(
        (
            "{0:>30}: ACCURACY TOP1: {3:.2f} %  ({1} / {2})\t  Total accuracy: {4:.2f} %   Semantic accuracy: {5:.2f} %   Syntactic accuracy: {6:.2f} %"
        ).format(
            question,
            correct,
            num_qs,
            correct / float(num_qs) * 100 if num_qs > 0 else 0,
            total_accuracy * 100,
            semantic_accuracy * 100,
            syntactic_accuracy * 100,
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

    # Retrieve list of normalized word vectors for the first words up
    # until the threshold count.
    f = load_model(args.model)
    # Gets words with associated frequeny sorted by default by descending order
    words, freq = f.get_words(include_freq=True)
    words = words[:args.threshold]
    vectors = np.zeros((len(words), f.get_dimension()), dtype=float)
    for i in range(len(words)):
        wv = f.get_word_vector(words[i])
        wv = wv / np.linalg.norm(wv)
        vectors[i] = wv

    total_correct = 0
    total_qs = 0
    total_num_lines = 0

    total_se_correct = 0
    total_se_qs = 0

    total_sy_correct = 0
    total_sy_qs = 0

    qid = 0
    questions = []
    with open(args.question_words, 'r') as fqw:
        questions = fqw.read().split(':')[1:]
    # For efficiency preallocate the memory to calculate cosine similarities
    cossims = np.zeros(len(words), dtype=float)
    for question in questions:
        quads = question.split('\n')
        question = quads[0].strip()
        quads = quads[1:-1]
        correct, num_qs, num_lines = process_question(
            quads, cossims, f, words, vectors
        )
        total_qs += num_qs
        total_correct += correct
        total_num_lines += num_lines

        if (qid < 5):
            total_se_correct += correct
            total_se_qs += num_qs
        else:
            total_sy_correct += correct
            total_sy_qs += num_qs

        print_compute_accuracy_score(
            question,
            correct,
            num_qs,
            total_correct / float(total_qs) if total_qs > 0 else 0,
            total_se_correct / float(total_se_qs) if total_se_qs > 0 else 0,
            total_sy_correct / float(total_sy_qs) if total_sy_qs > 0 else 0,
        )
        qid += 1

    print(
        "Questions seen / total: {0} {1}   {2:.2f} %".
        format(
            total_qs,
            total_num_lines,
            total_qs / total_num_lines * 100 if total_num_lines > 0 else 0,
        )
    )
