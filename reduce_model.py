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
import os
import re
import sys

import fasttext
import fasttext.util

args = None


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def guess_target_name(model_file, initial_dim, target_dim):
    """
    Given a model name with the convention a.<dim>.b, this function
    returns the model's name with `target_dim` value.
    For example model_file name `cc.en.300.bin` with initial dim 300 becomes
    `cc.en.100.bin` when the `target_dim` is 100.
    """
    prg = re.compile("(.*).%s.(.*)" % initial_dim)
    m = prg.match(model_file)
    if m:
        return "%s.%d.%s" % (m.group(1), target_dim, m.group(2))

    sp_ext = os.path.splitext(model_file)
    return "%s.%d%s" % (sp_ext[0], target_dim, sp_ext[1])


def command_reduce(model_file, target_dim, if_exists):
    """
    Given a `model_file`, this function reduces its dimension to `target_dim`
    by applying a PCA.
    """
    eprint("Loading model")

    ft = fasttext.load_model(model_file)
    initial_dim = ft.get_dimension()
    if target_dim >= initial_dim:
        raise Exception("Target dimension (%d) should be less than initial dimension (%d)." % (
            target_dim, initial_dim))

    result_filename = guess_target_name(model_file, initial_dim, target_dim)
    if os.path.isfile(result_filename):
        if if_exists == 'overwrite':
            pass
        elif if_exists == 'strict':
            raise Exception(
                "File already exists. Use --overwrite to overwrite.")
        elif if_exists == 'ignore':
            return result_filename

    eprint("Reducing matrix dimensions")
    fasttext.util.reduce_model(ft, target_dim)

    eprint("Saving model")
    ft.save_model(result_filename)
    eprint("%s saved" % result_filename)

    return result_filename


def main():
    global args

    parser = argparse.ArgumentParser(
        description='fastText helper tool to reduce model dimensions.')
    parser.add_argument("model", type=str,
                        help="model file to reduce. model.bin")
    parser.add_argument("dim", type=int,
                        help="targeted dimension of word vectors.")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite if file exists.")

    args = parser.parse_args()

    command_reduce(args.model, args.dim, if_exists=(
        'overwrite' if args.overwrite else 'strict'))


if __name__ == '__main__':
    main()
