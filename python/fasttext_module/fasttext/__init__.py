# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .FastText import train_supervised
from .FastText import train_unsupervised
from .FastText import load_model
from .FastText import tokenize
from .FastText import EOS
from .FastText import BOW
from .FastText import EOW

from .FastText import cbow
from .FastText import skipgram
from .FastText import supervised
