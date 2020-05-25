#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: The purpose of this file is not to accumulate all useful utility
# functions. This file should contain very commonly used and requested functions
# (such as test). If you think you have a function at that level, please create
# an issue and we will happily review your suggestion. This file is also not supposed
# to pull in dependencies outside of numpy/scipy without very good reasons. For
# example, this file should not use sklearn and matplotlib to produce a t-sne
# plot of word embeddings or such.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import shutil
import os
import gzip

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


valid_lang_ids = {"af", "sq", "als", "am", "ar", "an", "hy", "as", "ast",
                  "az", "ba", "eu", "bar", "be", "bn", "bh", "bpy", "bs",
                  "br", "bg", "my", "ca", "ceb", "bcl", "ce", "zh", "cv",
                  "co", "hr", "cs", "da", "dv", "nl", "pa", "arz", "eml",
                  "en", "myv", "eo", "et", "hif", "fi", "fr", "gl", "ka",
                  "de", "gom", "el", "gu", "ht", "he", "mrj", "hi", "hu",
                  "is", "io", "ilo", "id", "ia", "ga", "it", "ja", "jv",
                  "kn", "pam", "kk", "km", "ky", "ko", "ku", "ckb", "la",
                  "lv", "li", "lt", "lmo", "nds", "lb", "mk", "mai", "mg",
                  "ms", "ml", "mt", "gv", "mr", "mzn", "mhr", "min", "xmf",
                  "mwl", "mn", "nah", "nap", "ne", "new", "frr", "nso",
                  "no", "nn", "oc", "or", "os", "pfl", "ps", "fa", "pms",
                  "pl", "pt", "qu", "ro", "rm", "ru", "sah", "sa", "sc",
                  "sco", "gd", "sr", "sh", "scn", "sd", "si", "sk", "sl",
                  "so", "azb", "es", "su", "sw", "sv", "tl", "tg", "ta",
                  "tt", "te", "th", "bo", "tr", "tk", "uk", "hsb", "ur",
                  "ug", "uz", "vec", "vi", "vo", "wa", "war", "cy", "vls",
                  "fy", "pnb", "yi", "yo", "diq", "zea"}


# TODO: Add example on reproducing model.test with util.test and model.get_line
def test(predictions, labels, k=1):
    """
    Return precision and recall modeled after fasttext's test
    """
    precision = 0.0
    nexamples = 0
    nlabels = 0
    for prediction, labels in zip(predictions, labels):
        for p in prediction:
            if p in labels:
                precision += 1
        nexamples += 1
        nlabels += len(labels)
    return (precision / (k * nexamples), precision / nlabels)


def find_nearest_neighbor(query, vectors, ban_set, cossims=None):
    """
    query is a 1d numpy array corresponding to the vector to which you want to
    find the closest vector
    vectors is a 2d numpy array corresponding to the vectors you want to consider
    ban_set is a set of indicies within vectors you want to ignore for nearest match
    cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency

    returns the index of the closest match to query within vectors

    """
    if cossims is None:
        cossims = np.matmul(vectors, query, out=cossims)
    else:
        np.matmul(vectors, query, out=cossims)
    rank = len(cossims) - 1
    result_i = np.argpartition(cossims, rank)[rank]
    while result_i in ban_set:
        rank -= 1
        result_i = np.argpartition(cossims, rank)[rank]
    return result_i


def _reduce_matrix(X_orig, dim, eigv):
    """
    Reduces the dimension of a (m × n)   matrix `X_orig` to
                          to a (m × dim) matrix `X_reduced`
    It uses only the first 100000 rows of `X_orig` to do the mapping.
    Matrix types are all `np.float32` in order to avoid unncessary copies.
    """
    if eigv is None:
        mapping_size = 100000
        X = X_orig[:mapping_size]
        X = X - X.mean(axis=0, dtype=np.float32)
        C = np.divide(np.matmul(X.T, X), X.shape[0] - 1, dtype=np.float32)
        _, U = np.linalg.eig(C)
        eigv = U[:, :dim]

    X_reduced = np.matmul(X_orig, eigv)

    return (X_reduced, eigv)


def reduce_model(ft_model, target_dim):
    """
    ft_model is an instance of `_FastText` class
    This function computes the PCA of the input and the output matrices
    and sets the reduced ones.
    """
    inp_reduced, proj = _reduce_matrix(
        ft_model.get_input_matrix(), target_dim, None)
    out_reduced, _ = _reduce_matrix(
        ft_model.get_output_matrix(), target_dim, proj)

    ft_model.set_matrices(inp_reduced, out_reduced)

    return ft_model


def _print_progress(downloaded_bytes, total_size):
    percent = float(downloaded_bytes) / total_size
    bar_size = 50
    bar = int(percent * bar_size)
    percent = round(percent * 100, 2)
    sys.stdout.write(" (%0.2f%%) [" % percent)
    sys.stdout.write("=" * bar)
    sys.stdout.write(">")
    sys.stdout.write(" " * (bar_size - bar))
    sys.stdout.write("]\r")
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write('\n')


def _download_file(url, write_file_name, chunk_size=2**13):
    print("Downloading %s" % url)
    response = urlopen(url)
    if hasattr(response, 'getheader'):
        file_size = int(response.getheader('Content-Length').strip())
    else:
        file_size = int(response.info().getheader('Content-Length').strip())
    downloaded = 0
    download_file_name = write_file_name + ".part"
    with open(download_file_name, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            _print_progress(downloaded, file_size)

    os.rename(download_file_name, write_file_name)


def _download_gz_model(gz_file_name, if_exists):
    if os.path.isfile(gz_file_name):
        if if_exists == 'ignore':
            return True
        elif if_exists == 'strict':
            print("gzip File exists. Use --overwrite to download anyway.")
            return False
        elif if_exists == 'overwrite':
            pass

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/%s" % gz_file_name
    _download_file(url, gz_file_name)

    return True


def download_model(lang_id, if_exists='strict', dimension=None):
    """
        Download pre-trained common-crawl vectors from fastText's website
        https://fasttext.cc/docs/en/crawl-vectors.html
    """
    if lang_id not in valid_lang_ids:
        raise Exception("Invalid lang id. Please select among %s" %
                        repr(valid_lang_ids))

    file_name = "cc.%s.300.bin" % lang_id
    gz_file_name = "%s.gz" % file_name

    if os.path.isfile(file_name):
        if if_exists == 'ignore':
            return file_name
        elif if_exists == 'strict':
            print("File exists. Use --overwrite to download anyway.")
            return
        elif if_exists == 'overwrite':
            pass

    if _download_gz_model(gz_file_name, if_exists):
        with gzip.open(gz_file_name, 'rb') as f:
            with open(file_name, 'wb') as f_out:
                shutil.copyfileobj(f, f_out)

    return file_name
