# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from fastText import train_supervised
from fastText import util
import os
import subprocess
import unittest
import tempfile
try:
    import unicode
except ImportError:
    pass
from fastText.tests.test_configurations import get_supervised_models


def read_labels(data_file):
    labels = []
    lines = []
    with open(data_file, 'r') as f:
        for line in f:
            labels_line = []
            words_line = []
            try:
                line = unicode(line, "UTF-8").split()
            except NameError:
                line = line.split()
            for word in line:
                if word.startswith("__label__"):
                    labels_line.append(word)
                else:
                    words_line.append(word)
            labels.append(labels_line)
            lines.append(" ".join(words_line))
    return lines, labels


# Generate a supervised test case
# The returned function will be set as an attribute to a test class
def gen_sup_test(configuration):
    def sup_test(self):
        def get_path_size(path):
            path_size = subprocess.check_output(["stat", "-c", "%s",
                                                 path]).decode('utf-8')
            path_size = int(path_size)
            return path_size

        def check(model, model_filename, test, lessthan, msg_prefix=""):
            lines, labels = read_labels(test["data"])
            predictions = []
            for line in lines:
                pred_label, _ = model.predict(line)
                predictions.append(pred_label)
            p1_local_out, r1_local_out = util.test(predictions, labels)
            self.assertEqual(
                len(predictions), test["n"], msg_prefix + "N: Want: " +
                str(test["n"]) + " Is: " + str(len(predictions))
            )
            self.assertTrue(
                p1_local_out >= test["p1"], msg_prefix + "p1: Want: " +
                str(test["p1"]) + " Is: " + str(p1_local_out)
            )
            self.assertTrue(
                r1_local_out >= test["r1"], msg_prefix + "r1: Want: " +
                str(test["r1"]) + " Is: " + str(r1_local_out)
            )
            path_size = get_path_size(model_filename)
            size_msg = str(test["size"]) + " Is: " + str(path_size)
            if lessthan:
                self.assertTrue(
                    path_size <= test["size"],
                    msg_prefix + "Size: Want at most: " + size_msg
                )
            else:
                self.assertTrue(
                    path_size == test["size"],
                    msg_prefix + "Size: Want: " + size_msg
                )

        output = os.path.join(tempfile.mkdtemp(), configuration["dataset"])
        model = train_supervised(**configuration["train_args"])
        model.save_model(output + ".bin")
        check(model, output + ".bin", configuration["test"], False)
        model.quantize(**configuration["quant_args"])
        model.save_model(output + ".ftz")
        check(
            model, output + ".ftz", configuration["quant_test"], True, "Quant: "
        )

    return sup_test


def gen_small_tests(data_dir):
    class TestFastTextSmallPy(unittest.TestCase):
        pass

    for configuration in get_supervised_models(data_dir=data_dir):
        if configuration["dataset"] == "dbpedia":
            setattr(
                TestFastTextSmallPy, "test_small_" + configuration["dataset"],
                gen_sup_test(configuration)
            )
    return TestFastTextSmallPy


def gen_tests(data_dir):
    class TestFastTextPy(unittest.TestCase):
        pass

    for configuration in get_supervised_models(data_dir=data_dir):
        setattr(
            TestFastTextPy, "test_" + configuration["dataset"],
            gen_sup_test(configuration)
        )
    return TestFastTextPy
