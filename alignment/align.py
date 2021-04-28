#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
from utils import *
import sys

parser = argparse.ArgumentParser(description='RCSLS for supervised word alignment')

parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')

parser.add_argument("--dico_train", type=str, default='', help="train dictionary")
parser.add_argument("--dico_test", type=str, default='', help="validation dictionary")

parser.add_argument("--output", type=str, default='', help="where to save aligned embeddings")

parser.add_argument("--knn", type=int, default=10, help="number of nearest neighbors in RCSL/CSLS")
parser.add_argument("--maxneg", type=int, default=200000, help="Maximum number of negatives for the Extended RCSLS")
parser.add_argument("--maxsup", type=int, default=-1, help="Maximum number of training examples")
parser.add_argument("--maxload", type=int, default=200000, help="Maximum number of loaded vectors")

parser.add_argument("--model", type=str, default="none", help="Set of constraints: spectral or none")
parser.add_argument("--reg", type=float, default=0.0 , help='regularization parameters')

parser.add_argument("--lr", type=float, default=1.0, help='learning rate')
parser.add_argument("--niter", type=int, default=10, help='number of iterations')
parser.add_argument('--sgd', action='store_true', help='use sgd')
parser.add_argument("--batchsize", type=int, default=10000, help="batch size for sgd")

params = parser.parse_args()

###### SPECIFIC FUNCTIONS ######
# functions specific to RCSLS
# the rest of the functions are in utils.py

def getknn(sc, x, y, k=10):
    sidx = np.argpartition(sc, -k, axis=1)[:, -k:]
    ytopk = y[sidx.flatten(), :]
    ytopk = ytopk.reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
    f = np.sum(sc[np.arange(sc.shape[0])[:, None], sidx])
    df = np.dot(ytopk.sum(1).T, x)
    return f / k, df / k


def rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, knn=10):
    X_trans = np.dot(X_src, R.T)
    f = 2 * np.sum(X_trans * Y_tgt)
    df = 2 * np.dot(Y_tgt.T, X_src)
    fk0, dfk0 = getknn(np.dot(X_trans, Z_tgt.T), X_src, Z_tgt, knn)
    fk1, dfk1 = getknn(np.dot(np.dot(Z_src, R.T), Y_tgt.T).T, Y_tgt, Z_src, knn)
    f = f - fk0 -fk1
    df = df - dfk0 - dfk1.T
    return -f / X_src.shape[0], -df / X_src.shape[0]


def proj_spectral(R):
    U, s, V = np.linalg.svd(R)
    s[s > 1] = 1
    s[s < 0] = 0
    return np.dot(U, np.dot(np.diag(s), V))


###### MAIN ######

# load word embeddings
words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)

# load validation bilingual lexicon
src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)

# word --> vector indices
idx_src = idx(words_src)
idx_tgt = idx(words_tgt)

# load train bilingual lexicon
pairs = load_pairs(params.dico_train, idx_src, idx_tgt)
if params.maxsup > 0 and params.maxsup < len(pairs):
    pairs = pairs[:params.maxsup]

# selecting training vector  pairs
X_src, Y_tgt = select_vectors_from_pairs(x_src, x_tgt, pairs)

# adding negatives for RCSLS
Z_src = x_src[:params.maxneg, :]
Z_tgt = x_tgt[:params.maxneg, :]

# initialization:
R = procrustes(X_src, Y_tgt)
nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
print("[init -- Procrustes] NN: %.4f"%(nnacc))
sys.stdout.flush()

# optimization
fold, Rold = 0, []
niter, lr = params.niter, params.lr

for it in range(0, niter + 1):
    if lr < 1e-4:
        break

    if params.sgd:
        indices = np.random.choice(X_src.shape[0], size=params.batchsize, replace=False)
        f, df = rcsls(X_src[indices, :], Y_tgt[indices, :], Z_src, Z_tgt, R, params.knn)
    else:
        f, df = rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, params.knn)

    if params.reg > 0:
        R *= (1 - lr * params.reg)
    R -= lr * df
    if params.model == "spectral":
        R = proj_spectral(R)

    print("[it=%d] f = %.4f" % (it, f))
    sys.stdout.flush()

    if f > fold and it > 0 and not params.sgd:
        lr /= 2
        f, R = fold, Rold

    fold, Rold = f, R

    if (it > 0 and it % 10 == 0) or it == niter:
        nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
        print("[it=%d] NN = %.4f - Coverage = %.4f" % (it, nnacc, len(src2tgt) / lexicon_size))

nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
print("[final] NN = %.4f - Coverage = %.4f" % (nnacc, len(src2tgt) / lexicon_size))

if params.output != "":
    print("Saving all aligned vectors at %s" % params.output)
    words_full, x_full = load_vectors(params.src_emb, maxload=-1, center=params.center, verbose=False)
    x = np.dot(x_full, R.T)
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    save_vectors(params.output, x, words_full)
    save_matrix(params.output + "-mat",  R)
