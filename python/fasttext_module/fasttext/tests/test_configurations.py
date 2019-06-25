# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import multiprocessing

# This script represents a collection of integration tests
# Each integration test comes with a full set of parameters,
# a dataset, and expected metrics.
# These configurations can be used by various fastText APIs
# to confirm some level of correctness.


def max_thread():
    return multiprocessing.cpu_count() - 1


def check_supervised_configuration(configuration, verbose=1):
    configuration["args"]["verbose"] = verbose
    configuration["quant_args"]["verbose"] = verbose
    return configuration


def check_supervised_configurations(configurations, verbose=1):
    for i in range(len(configurations)):
        configurations[i] = check_supervised_configuration(
            configurations[i], verbose=verbose
        )
    return configurations


def flickr_job(thread=None):
    if thread is None:
        thread = max_thread()
    config = {}
    config["dataset"] = "YFCC100M"
    config["args"] = {
        "dim": 256,
        "wordNgrams": 2,
        "minCount": 10,
        "bucket": 10000000,
        "epoch": 20,
        "loss": "hs",
        "minCountLabel": 100,
        "thread": thread
    }
    config["args"]["input"] = "YFCC100M/train"
    config["quant_args"] = {
        "dsub": 2,
        "lr": 0.1,
        "epoch": 5,
        "cutoff": 100000,
        "qnorm": True,
        "retrain": True,
        "qout": True
    }
    config["quant_args"]["input"] = config["args"]["input"]
    config["test"] = {
        "n": 647224,
        "p1": 0.470,
        "r1": 0.071,
        "size": 12060039727,
        "data": "YFCC100M/test",
    }
    # One quant example (to illustrate slack): 0.344, 0.0528, 64506972
    config["quant_test"] = {
        "n": 647224,
        "p1": 0.300,
        "r1": 0.0450,
        "size": 70000000,
        "data": "YFCC100M/test",
    }
    return config


def langid_job1(thread=None):
    if thread is None:
        thread = max_thread()
    config = {}
    config["dataset"] = "langid"
    config["args"] = {"dim": 16, "minn": 2, "maxn": 4, "thread": thread}
    config["args"]["input"] = "langid.train"
    config["quant_args"] = {"qnorm": True, "cutoff": 50000, "retrain": True}
    config["quant_args"]["input"] = config["args"]["input"]
    config["test"] = {
        "n": 10000,
        "p1": 0.985,
        "r1": 0.985,
        "size": 368132610,
        "data": "langid.valid",
    }
    # One quant example (to illustrate slack): 0.984 0.984 932793
    config["quant_test"] = {
        "p1": 0.97,
        "r1": 0.97,
        "size": 1000000,
    }
    config["quant_test"]["n"] = config["test"]["n"]
    config["quant_test"]["data"] = config["test"]["data"]
    return config


def langid_job2(thread=None):
    if thread is None:
        thread = max_thread()
    config = langid_job1(thread).copy()
    config["args"]["loss"] = "hs"
    return config


def cooking_job1(thread=None):
    if thread is None:
        thread = max_thread()
    config = {}
    config["dataset"] = "cooking"
    config["args"] = {
        "epoch": 25,
        "lr": 1.0,
        "wordNgrams": 2,
        "minCount": 1,
        "thread": thread,
    }
    config["args"]["input"] = "cooking.train"
    config["quant_args"] = {"qnorm": True, "cutoff": 50000, "retrain": True}
    config["quant_args"]["input"] = config["args"]["input"]
    config["test"] = {
        "n": 3000,
        "p1": 0.59,
        "r1": 0.25,
        "size": 804047585,
        "data": "cooking.valid",
    }
    # One quant example (to illustrate slack): 0.602 0.26 3439172
    config["quant_test"] = {
        "p1": 0.55,
        "r1": 0.20,
        "size": 4000000,
    }
    config["quant_test"]["n"] = config["test"]["n"]
    config["quant_test"]["data"] = config["test"]["data"]
    return config


def cooking_job2(thread=None):
    if thread is None:
        thread = max_thread()
    config = cooking_job1(thread).copy()
    config["args"]["loss"] = "hs"
    return config


# Supervised models
# See https://fasttext.cc/docs/en/supervised-models.html
def get_supervised_models(thread=None, verbose=1):
    if thread is None:
        thread = max_thread()
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
        "thread": thread,
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

    sup_job_p1 = [0.915, 0.968, 0.983, 0.956, 0.638, 0.723, 0.600, 0.940]
    sup_job_r1 = [0.915, 0.968, 0.983, 0.956, 0.638, 0.723, 0.600, 0.940]
    sup_job_size = [
        405607193, 421445471, 447481878, 427867393, 431292576, 517549567,
        483742593, 493604598
    ]

    sup_job_quant_p1 = [0.918, 0.965, 0.983, 0.950, 0.625, 0.707, 0.58, 0.920]
    sup_job_quant_r1 = [0.918, 0.965, 0.983, 0.950, 0.625, 0.707, 0.58, 0.920]
    sup_job_quant_size = [
        1600000, 1500000, 1700000, 1600000, 1600000, 1700000, 1600000, 1600000
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
        configuration["args"] = args
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
        configuration["test"] = test
        configuration["quant_test"] = quant_test
        configurations.append(configuration)
    configurations.append(flickr_job())
    configurations.append(langid_job1())
    configurations.append(langid_job2())
    configurations.append(cooking_job1())
    configurations.append(cooking_job2())
    configurations = check_supervised_configurations(
        configurations, verbose=verbose
    )
    return configurations
