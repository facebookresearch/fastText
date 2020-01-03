#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import fasttext.util


args = None


def command_download(lang_id, if_exists):
    """
        Download pre-trained common-crawl vectors from fastText's website
        https://fasttext.cc/docs/en/crawl-vectors.html
    """
    fasttext.util.download_model(lang_id, if_exists)


def main():
    global args

    parser = argparse.ArgumentParser(
        description='fastText helper tool to reduce model dimensions.')
    parser.add_argument("language", type=str, default="en",
                        help="language identifier of the pre-trained vectors. For example `en` or `fr`.")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite if file exists.")

    args = parser.parse_args()

    command_download(args.language, if_exists=(
        'overwrite' if args.overwrite else 'strict'))


if __name__ == '__main__':
    main()
