#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io, os, ot, argparse, random
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description=' ')

parser.add_argument('--embdir', default='data/', type=str)
parser.add_argument('--outdir', default='output/', type=str)
parser.add_argument('--lglist', default='en-fr-es-it-pt-de-pl-ru-da-nl-cs',  type=str,
        help='list of languages. The first element is the pivot. Example: en-fr-es to align English, French and Spanish with English as the pivot.')

parser.add_argument('--maxload', default=20000, type=int, help='Max number of loaded vectors')
parser.add_argument('--uniform', action='store_true', help='switch to uniform probability of picking language pairs')

# optimization parameters for the square loss
parser.add_argument('--epoch', default=2, type=int, help='nb of epochs for square loss')
parser.add_argument('--niter', default=500, type=int, help='max number of iteration per epoch for square loss')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for square loss')
parser.add_argument('--bsz', default=500, type=int, help='batch size for square loss')

# optimization parameters for the RCSLS loss
parser.add_argument('--altepoch', default=100, type=int, help='nb of epochs for RCSLS loss')
parser.add_argument('--altlr', default=25, type=float, help='learning rate for RCSLS loss')
parser.add_argument("--altbsz", type=int, default=1000, help="batch size for RCSLS")

args = parser.parse_args()

###### SPECIFIC FUNCTIONS ######

def getknn(sc, x, y, k=10):
    sidx = np.argpartition(sc, -k, axis=1)[:, -k:]
    ytopk = y[sidx.flatten(), :]
    ytopk = ytopk.reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
    f = np.sum(sc[np.arange(sc.shape[0])[:, None], sidx])
    df = np.dot(ytopk.sum(1).T, x)
    return f / k, df / k


def rcsls(Xi, Xj, Zi, Zj, R, knn=10):
    X_trans = np.dot(Xi, R.T)
    f = 2 * np.sum(X_trans * Xj)
    df = 2 * np.dot(Xj.T, Xi)
    fk0, dfk0 = getknn(np.dot(X_trans, Zj.T), Xi, Zj, knn)
    fk1, dfk1 = getknn(np.dot(np.dot(Zi, R.T), Xj.T).T, Xj, Zi, knn)
    f = f - fk0 -fk1
    df = df - dfk0 - dfk1.T
    return -f / Xi.shape[0], -df.T / Xi.shape[0]


def GWmatrix(emb0):
    N = np.shape(emb0)[0]
    N2 = .5* np.linalg.norm(emb0, axis=1).reshape(1, N)
    C2 = np.tile(N2.transpose(), (1, N)) + np.tile(N2, (N, 1))
    C2 -= np.dot(emb0,emb0.T)
    return C2


def gromov_wasserstein(x_src, x_tgt, C2):
    N = x_src.shape[0]
    C1 = GWmatrix(x_src)
    M = ot.gromov_wasserstein(C1,C2,np.ones(N),np.ones(N),'square_loss',epsilon=0.55,max_iter=100,tol=1e-4)
    return procrustes(np.dot(M,x_tgt), x_src)


def align(EMB, TRANS, lglist, args):
    nmax, l = args.maxload, len(lglist)
    # create a list of language pairs to sample from
    # (default == higher probability to pick a language pair contianing the pivot)
    # if --uniform: uniform probability of picking a language pair
    samples = []
    for i in range(l):
        for j in range(l):
            if j == i :
                continue
            if j > 0 and args.uniform == False:
                samples.append((0,j))
            if i > 0 and args.uniform == False:
                samples.append((i,0))
            samples.append((i,j))

    # optimization of the l2 loss
    print('start optimizing L2 loss')
    lr0, bsz, nepoch, niter = args.lr, args.bsz, args.epoch, args.niter
    for epoch in range(nepoch):
        print("start epoch %d / %d"%(epoch+1, nepoch))
        ones = np.ones(bsz)
        f, fold, nb, lr = 0.0, 0.0, 0.0, lr0
        for it in range(niter):
            if it > 1 and f > fold + 1e-3:
                lr /= 2
            if lr < .05:
                break
            fold = f
            f, nb = 0.0, 0.0
            for k in range(100 *  (l-1)):
                (i,j) = random.choice(samples)
                embi = EMB[i][np.random.permutation(nmax)[:bsz], :]
                embj = EMB[j][np.random.permutation(nmax)[:bsz], :]
                perm = ot.sinkhorn(ones, ones, np.linalg.multi_dot([embi, -TRANS[i], TRANS[j].T,embj.T]), reg = 0.025, stopThr = 1e-3)
                grad = np.linalg.multi_dot([embi.T, perm, embj])
                f -= np.trace(np.linalg.multi_dot([TRANS[i].T, grad, TRANS[j]])) / embi.shape[0]
                nb += 1
                if i > 0:
                    TRANS[i] = proj_ortho(TRANS[i] + lr * np.dot(grad, TRANS[j]))
                if j > 0:
                    TRANS[j] = proj_ortho(TRANS[j] + lr * np.dot(grad.transpose(), TRANS[i]))
            print("iter %d / %d - epoch %d - loss: %.5f  lr: %.4f" % (it, niter, epoch+1, f / nb , lr))
        print("end of epoch %d - loss: %.5f - lr: %.4f" % (epoch+1, f / max(nb,1), lr))
        niter, bsz = max(int(niter/2),2), min(1000, bsz * 2)
    #end for epoch in range(nepoch):

    # optimization of the RCSLS loss
    print('start optimizing RCSLS loss')
    f, fold, nb, lr = 0.0, 0.0, 0.0, args.altlr
    for epoch in range(args.altepoch):
        if epoch > 1  and f-fold > -1e-4 * abs(fold):
            lr/= 2
        if lr < 1e-1:
            break
        fold = f
        f, nb = 0.0, 0.0
        for k in range(round(nmax / args.altbsz) * 10 * (l-1)):
            (i,j) = random.choice(samples)
            sgdidx = np.random.choice(nmax, size=args.altbsz, replace=False)
            embi = EMB[i][sgdidx, :]
            embj = EMB[j][:nmax, :]
            # crude alignment approximation:
            T = np.dot(TRANS[i], TRANS[j].T)
            scores = np.linalg.multi_dot([embi, T, embj.T])
            perm = np.zeros_like(scores)
            perm[np.arange(len(scores)), scores.argmax(1)] = 1
            embj = np.dot(perm, embj)
            # normalization over a subset of embeddings for speed up
            fi, grad = rcsls(embi, embj, embi, embj, T.T)
            f += fi
            nb += 1
            if i > 0:
                TRANS[i] = proj_ortho(TRANS[i] - lr * np.dot(grad, TRANS[j]))
            if j > 0:
                TRANS[j] = proj_ortho(TRANS[j] - lr * np.dot(grad.transpose(), TRANS[i]))
        print("epoch %d - loss: %.5f - lr: %.4f" % (epoch+1, f / max(nb,1), lr))
    #end for epoch in range(args.altepoch):
    return TRANS

def convex_init(X, Y, niter=100, reg=0.05, apply_sqrt=False):
    n, d = X.shape
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([n, n]) / float(n)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    return procrustes(np.dot(P, X), Y).T


###### MAIN ######

lglist = args.lglist.split('-')
l = len(lglist)

# embs:
EMB = {}
for i in range(l):
    fn = args.embdir + '/wiki.' + lglist[i] + '.vec'
    _, vecs = load_vectors(fn, maxload=args.maxload)
    EMB[i] = vecs

#init
print("Computing initial bilingual apping with Gromov-Wasserstein...")
TRANS={}
maxinit = 2000
emb0 = EMB[0][:maxinit,:]
C0 = GWmatrix(emb0)
TRANS[0] = np.eye(300)
for i in range(1, l):
    print("init "+lglist[i])
    embi = EMB[i][:maxinit,:]
    TRANS[i] = gromov_wasserstein(embi, emb0, C0)

# align
align(EMB, TRANS, lglist, args)

print('saving matrices in ' + args.outdir)
languages=''.join(lglist)
for i in range(l):
    save_matrix(args.outdir + '/W-' + languages + '-' + lglist[i], TRANS[i])
