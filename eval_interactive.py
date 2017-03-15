from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from scipy import stats
import sys
import os
import math
import argparse

def compat_splitting(line):
    return line.decode('utf8').split()

def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2

def top_k(query_vector, vectors):
  similarities = []
  for word in vectors:
    similarities.append((word, similarity(query_vector, vectors[word])))
  return sorted(similarities, key=lambda x: x[1])[::-1][0:10]


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', '-m', dest='modelPath', action='store', required=True, help='path to model')
args = parser.parse_args()

vectors = {}
fin = open(args.modelPath, 'rb')
for i, line in enumerate(fin):
    try:
        tab = compat_splitting(line)
        vec = np.array(tab[1:], dtype=float)
        word = tab[0]
        if not word in vectors:
            vectors[word] = vec
    except ValueError:
        continue
    except UnicodeDecodeError:
        continue
fin.close()

def nearest(word):
  return top_k(vectors[word], vectors)[1:]

def normalize(vector):
  return vector / np.sum(vector)

def analogy(a, b, c):
  query_vector = normalize(vectors[a]) - normalize(vectors[b]) + normalize(vectors[c])
  res = top_k(query_vector, vectors)
  return list(filter(lambda x: x[0] not in [a, b, c], res))

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)

_start_shell()


