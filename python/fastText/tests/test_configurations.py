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

import multiprocessing
import os

# This script represents a collection of integration tests
# Each integration test comes with a full set of parameters,
# a dataset, and expected metrics.
# These configurations can be used by various fastText apis
# to confirm some level of correctness.

# Supervised models
# See https://fasttext.cc/docs/en/supervised-models.html


def max_thread():
    return multiprocessing.cpu_count() - 1


def get_supervised_models(data_dir=""):
    sup_job_dataset = [
        "ag_news", "sogou_news", "dbpedia", "yelp_review_polarity",
        "yelp_review_full", "yahoo_answers", "amazon_review_full",
        "amazon_review_polarity"
    ]

    sup_params = {
        "dim": 10,
        "wordNgrams": 2,
        "minCount": 1,
        "bucket": 10000000,
        "epoch": 5,
        "thread": max_thread(),
        "verbose": 1,
    }
    quant_params = {
        "retrain": True,
        "cutoff": 100000,
        "qnorm": True,
        "verbose": 1,
    }
    sup_job_lr = [0.25, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05]

    sup_job_n = [7600, 60000, 70000, 38000, 50000, 60000, 650000, 400000]

    sup_job_p1 = [0.921, 0.968, 0.984, 0.956, 0.638, 0.723, 0.603, 0.946]
    sup_job_r1 = [0.921, 0.968, 0.984, 0.956, 0.638, 0.723, 0.603, 0.946]
    sup_job_size = [
        405607193, 421445471, 447481878, 427867393, 431292576, 517549567,
        483742593, 493604598
    ]

    sup_job_quant_p1 = [0.918, 0.965, 0.984, 0.950, 0.625, 0.707, 0.58, 0.940]
    sup_job_quant_r1 = [0.918, 0.965, 0.984, 0.950, 0.625, 0.707, 0.58, 0.940]
    sup_job_quant_size = [
        1600000, 1457000, 1690000, 1550000, 1567896, 1655000, 1600000, 1575000
    ]

    configurations = []
    for i in range(len(sup_job_dataset)):
        configuration = {}
        configuration["dataset"] = sup_job_dataset[i]
        args = sup_params.copy()
        quant_args = quant_params.copy()
        args["lr"] = sup_job_lr[i]
        args["input"] = sup_job_dataset[i] + ".train"
        quant_args["lr"] = sup_job_lr[i]
        quant_args["input"] = sup_job_dataset[i] + ".train"
        if data_dir:
            args["input"] = os.path.join(data_dir, args["input"])
            quant_args["input"] = os.path.join(data_dir, quant_args["input"])
        configuration["train_args"] = args
        configuration["quant_args"] = quant_args
        test = {
            "n": sup_job_n[i],
            "p1": sup_job_p1[i],
            "r1": sup_job_r1[i],
            "size": sup_job_size[i],
            "data": sup_job_dataset[i] + ".test",
        }
        quant_test = {
            "n": sup_job_n[i],
            "p1": sup_job_quant_p1[i],
            "r1": sup_job_quant_r1[i],
            "size": sup_job_quant_size[i],
            "data": sup_job_dataset[i] + ".test",
        }
        if data_dir:
            test["data"] = os.path.join(data_dir, test["data"])
            quant_test["data"] = os.path.join(data_dir, quant_test["data"])
        configuration["test"] = test
        configuration["quant_test"] = quant_test
        configurations.append(configuration)
    return configurations
