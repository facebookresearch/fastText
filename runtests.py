#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

# To run the integration tests you must first fetch all the required test data.
# Have a look at tests/fetch_test_data.sh
# You will then need to point this script to the corresponding folder

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import argparse
from fastText.tests import gen_tests
from fastText.tests import gen_unit_tests


def run_tests(tests):
    suite = unittest.TestLoader().loadTestsFromTestCase(tests)
    unittest.TextTestRunner(verbosity=3).run(suite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--unit-tests", help="run unit tests", action="store_true"
    )
    parser.add_argument(
        "-i",
        "--integration-tests",
        help="run integration tests",
        action="store_true"
    )
    parser.add_argument("--data_dir", help="Full path to data directory")
    args = parser.parse_args()
    if args.unit_tests:
        run_tests(gen_unit_tests())
    if args.integration_tests:
        if args.data_dir is None:
            raise ValueError(
                "Need data directory! Consult tests/fetch_test_data.sh"
            )
        run_tests(gen_tests(args.data_dir))
    if not args.unit_tests and not args.integration_tests:
        print("Ran no tests")
