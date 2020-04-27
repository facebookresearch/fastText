# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import fasttext_pybind as fasttext
import numpy as np
import multiprocessing
import sys
from itertools import chain

loss_name = fasttext.loss_name
model_name = fasttext.model_name
EOS = "</s>"
BOW = "<"
EOW = ">"

displayed_errors = {}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class _Meter(object):
    def __init__(self, fasttext_model, meter):
        self.f = fasttext_model
        self.m = meter

    def score_vs_true(self, label):
        """Return scores and the gold of each sample for a specific label"""
        label_id = self.f.get_label_id(label)
        pair_list = self.m.scoreVsTrue(label_id)

        if pair_list:
            y_scores, y_true = zip(*pair_list)
        else:
            y_scores, y_true = ([], ())

        return np.array(y_scores, copy=False), np.array(y_true, copy=False)

    def precision_recall_curve(self, label=None):
        """Return precision/recall curve"""
        if label:
            label_id = self.f.get_label_id(label)
            pair_list = self.m.precisionRecallCurveLabel(label_id)
        else:
            pair_list = self.m.precisionRecallCurve()

        if pair_list:
            precision, recall = zip(*pair_list)
        else:
            precision, recall = ([], ())

        return np.array(precision, copy=False), np.array(recall, copy=False)

    def precision_at_recall(self, recall, label=None):
        """Return precision for a given recall"""
        if label:
            label_id = self.f.get_label_id(label)
            precision = self.m.precisionAtRecallLabel(label_id, recall)
        else:
            precision = self.m.precisionAtRecall(recall)

        return precision

    def recall_at_precision(self, precision, label=None):
        """Return recall for a given precision"""
        if label:
            label_id = self.f.get_label_id(label)
            recall = self.m.recallAtPrecisionLabel(label_id, precision)
        else:
            recall = self.m.recallAtPrecision(precision)

        return recall


class _FastText(object):
    """
    This class defines the API to inspect models and should not be used to
    create objects. It will be returned by functions such as load_model or
    train.

    In general this API assumes to be given only unicode for Python2 and the
    Python3 equvalent called str for any string-like arguments. All unicode
    strings are then encoded as UTF-8 and fed to the fastText C++ API.
    """

    def __init__(self, model_path=None, args=None):
        self.f = fasttext.fasttext()
        if model_path is not None:
            self.f.loadModel(model_path)
        self._words = None
        self._labels = None
        self.set_args(args)

    def set_args(self, args=None):
        if args:
            arg_names = ['lr', 'dim', 'ws', 'epoch', 'minCount',
                         'minCountLabel', 'minn', 'maxn', 'neg', 'wordNgrams',
                         'loss', 'bucket', 'thread', 'lrUpdateRate', 't',
                         'label', 'verbose', 'pretrainedVectors']
            for arg_name in arg_names:
                setattr(self, arg_name, getattr(args, arg_name))

    def is_quantized(self):
        return self.f.isQuant()

    def get_dimension(self):
        """Get the dimension (size) of a lookup vector (hidden layer)."""
        a = self.f.getArgs()
        return a.dim

    def get_word_vector(self, word):
        """Get the vector representation of word."""
        dim = self.get_dimension()
        b = fasttext.Vector(dim)
        self.f.getWordVector(b, word)
        return np.array(b)

    def get_sentence_vector(self, text):
        """
        Given a string, get a single vector represenation. This function
        assumes to be given a single line of text. We split words on
        whitespace (space, newline, tab, vertical tab) and the control
        characters carriage return, formfeed and the null character.
        """
        if text.find('\n') != -1:
            raise ValueError(
                "predict processes one line at a time (remove \'\\n\')"
            )
        text += "\n"
        dim = self.get_dimension()
        b = fasttext.Vector(dim)
        self.f.getSentenceVector(b, text)
        return np.array(b)

    def get_nearest_neighbors(self, word, k=10, on_unicode_error='strict'):
        return self.f.getNN(word, k, on_unicode_error)

    def get_analogies(self, wordA, wordB, wordC, k=10,
                      on_unicode_error='strict'):
        return self.f.getAnalogies(wordA, wordB, wordC, k, on_unicode_error)

    def get_word_id(self, word):
        """
        Given a word, get the word id within the dictionary.
        Returns -1 if word is not in the dictionary.
        """
        return self.f.getWordId(word)

    def get_label_id(self, label):
        """
        Given a label, get the label id within the dictionary.
        Returns -1 if label is not in the dictionary.
        """
        return self.f.getLabelId(label)

    def get_subword_id(self, subword):
        """
        Given a subword, return the index (within input matrix) it hashes to.
        """
        return self.f.getSubwordId(subword)

    def get_subwords(self, word, on_unicode_error='strict'):
        """
        Given a word, get the subwords and their indicies.
        """
        pair = self.f.getSubwords(word, on_unicode_error)
        return pair[0], np.array(pair[1])

    def get_input_vector(self, ind):
        """
        Given an index, get the corresponding vector of the Input Matrix.
        """
        dim = self.get_dimension()
        b = fasttext.Vector(dim)
        self.f.getInputVector(b, ind)
        return np.array(b)

    def predict(self, text, k=1, threshold=0.0, on_unicode_error='strict'):
        """
        Given a string, get a list of labels and a list of
        corresponding probabilities. k controls the number
        of returned labels. A choice of 5, will return the 5
        most probable labels. By default this returns only
        the most likely label and probability. threshold filters
        the returned labels by a threshold on probability. A
        choice of 0.5 will return labels with at least 0.5
        probability. k and threshold will be applied together to
        determine the returned labels.

        This function assumes to be given
        a single line of text. We split words on whitespace (space,
        newline, tab, vertical tab) and the control characters carriage
        return, formfeed and the null character.

        If the model is not supervised, this function will throw a ValueError.

        If given a list of strings, it will return a list of results as usually
        received for a single line of text.
        """

        def check(entry):
            if entry.find('\n') != -1:
                raise ValueError(
                    "predict processes one line at a time (remove \'\\n\')"
                )
            entry += "\n"
            return entry

        if type(text) == list:
            text = [check(entry) for entry in text]
            all_labels, all_probs = self.f.multilinePredict(
                text, k, threshold, on_unicode_error)

            return all_labels, all_probs
        else:
            text = check(text)
            predictions = self.f.predict(text, k, threshold, on_unicode_error)
            if predictions:
                probs, labels = zip(*predictions)
            else:
                probs, labels = ([], ())

            return labels, np.array(probs, copy=False)

    def get_input_matrix(self):
        """
        Get a reference to the full input matrix of a Model. This only
        works if the model is not quantized.
        """
        if self.f.isQuant():
            raise ValueError("Can't get quantized Matrix")
        return np.array(self.f.getInputMatrix())

    def get_output_matrix(self):
        """
        Get a reference to the full output matrix of a Model. This only
        works if the model is not quantized.
        """
        if self.f.isQuant():
            raise ValueError("Can't get quantized Matrix")
        return np.array(self.f.getOutputMatrix())

    def get_words(self, include_freq=False, on_unicode_error='strict'):
        """
        Get the entire list of words of the dictionary optionally
        including the frequency of the individual words. This
        does not include any subwords. For that please consult
        the function get_subwords.
        """
        pair = self.f.getVocab(on_unicode_error)
        if include_freq:
            return (pair[0], np.array(pair[1]))
        else:
            return pair[0]

    def get_labels(self, include_freq=False, on_unicode_error='strict'):
        """
        Get the entire list of labels of the dictionary optionally
        including the frequency of the individual labels. Unsupervised
        models use words as labels, which is why get_labels
        will call and return get_words for this type of
        model.
        """
        a = self.f.getArgs()
        if a.model == model_name.supervised:
            pair = self.f.getLabels(on_unicode_error)
            if include_freq:
                return (pair[0], np.array(pair[1]))
            else:
                return pair[0]
        else:
            return self.get_words(include_freq)

    def get_line(self, text, on_unicode_error='strict'):
        """
        Split a line of text into words and labels. Labels must start with
        the prefix used to create the model (__label__ by default).
        """

        def check(entry):
            if entry.find('\n') != -1:
                raise ValueError(
                    "get_line processes one line at a time (remove \'\\n\')"
                )
            entry += "\n"
            return entry

        if type(text) == list:
            text = [check(entry) for entry in text]
            return self.f.multilineGetLine(text, on_unicode_error)
        else:
            text = check(text)
            return self.f.getLine(text, on_unicode_error)

    def save_model(self, path):
        """Save the model to the given path"""
        self.f.saveModel(path)

    def test(self, path, k=1, threshold=0.0):
        """Evaluate supervised model using file given by path"""
        return self.f.test(path, k, threshold)

    def test_label(self, path, k=1, threshold=0.0):
        """
        Return the precision and recall score for each label.

        The returned value is a dictionary, where the key is the label.
        For example:
        f.test_label(...)
        {'__label__italian-cuisine' : {'precision' : 0.7, 'recall' : 0.74}}
        """
        return self.f.testLabel(path, k, threshold)

    def get_meter(self, path, k=-1):
        meter = _Meter(self, self.f.getMeter(path, k))

        return meter

    def quantize(
        self,
        input=None,
        qout=False,
        cutoff=0,
        retrain=False,
        epoch=None,
        lr=None,
        thread=None,
        verbose=None,
        dsub=2,
        qnorm=False
    ):
        """
        Quantize the model reducing the size of the model and
        it's memory footprint.
        """
        a = self.f.getArgs()
        if not epoch:
            epoch = a.epoch
        if not lr:
            lr = a.lr
        if not thread:
            thread = a.thread
        if not verbose:
            verbose = a.verbose
        if retrain and not input:
            raise ValueError("Need input file path if retraining")
        if input is None:
            input = ""
        self.f.quantize(
            input, qout, cutoff, retrain, epoch, lr, thread, verbose, dsub,
            qnorm
        )

    def set_matrices(self, input_matrix, output_matrix):
        """
        Set input and output matrices. This function assumes you know what you
        are doing.
        """
        self.f.setMatrices(input_matrix.astype(np.float32),
                           output_matrix.astype(np.float32))

    @property
    def words(self):
        if self._words is None:
            self._words = self.get_words()
        return self._words

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.get_labels()
        return self._labels

    def __getitem__(self, word):
        return self.get_word_vector(word)

    def __contains__(self, word):
        return word in self.words


def _parse_model_string(string):
    if string == "cbow":
        return model_name.cbow
    if string == "skipgram":
        return model_name.skipgram
    if string == "supervised":
        return model_name.supervised
    else:
        raise ValueError("Unrecognized model name")


def _parse_loss_string(string):
    if string == "ns":
        return loss_name.ns
    if string == "hs":
        return loss_name.hs
    if string == "softmax":
        return loss_name.softmax
    if string == "ova":
        return loss_name.ova
    else:
        raise ValueError("Unrecognized loss name")


def _build_args(args, manually_set_args):
    args["model"] = _parse_model_string(args["model"])
    args["loss"] = _parse_loss_string(args["loss"])
    if type(args["autotuneModelSize"]) == int:
        args["autotuneModelSize"] = str(args["autotuneModelSize"])

    a = fasttext.args()
    for (k, v) in args.items():
        setattr(a, k, v)
        if k in manually_set_args:
            a.setManual(k)
    a.output = ""  # User should use save_model
    a.saveOutput = 0  # Never use this
    if a.wordNgrams <= 1 and a.maxn == 0:
        a.bucket = 0
    return a


def tokenize(text):
    """Given a string of text, tokenize it and return a list of tokens"""
    f = fasttext.fasttext()
    return f.tokenize(text)


def load_model(path):
    """Load a model given a filepath and return a model object."""
    eprint("Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.")
    return _FastText(model_path=path)


unsupervised_default = {
    'model': "skipgram",
    'lr': 0.05,
    'dim': 100,
    'ws': 5,
    'epoch': 5,
    'minCount': 5,
    'minCountLabel': 0,
    'minn': 3,
    'maxn': 6,
    'neg': 5,
    'wordNgrams': 1,
    'loss': "ns",
    'bucket': 2000000,
    'thread': multiprocessing.cpu_count() - 1,
    'lrUpdateRate': 100,
    't': 1e-4,
    'label': "__label__",
    'verbose': 2,
    'pretrainedVectors': "",
    'seed': 0,
    'autotuneValidationFile': "",
    'autotuneMetric': "f1",
    'autotunePredictions': 1,
    'autotuneDuration': 60 * 5,  # 5 minutes
    'autotuneModelSize': ""
}


def read_args(arg_list, arg_dict, arg_names, default_values):
    param_map = {
        'min_count': 'minCount',
        'word_ngrams': 'wordNgrams',
        'lr_update_rate': 'lrUpdateRate',
        'label_prefix': 'label',
        'pretrained_vectors': 'pretrainedVectors'
    }

    ret = {}
    manually_set_args = set()
    for (arg_name, arg_value) in chain(zip(arg_names, arg_list), arg_dict.items()):
        if arg_name in param_map:
            arg_name = param_map[arg_name]
        if arg_name not in arg_names:
            raise TypeError("unexpected keyword argument '%s'" % arg_name)
        if arg_name in ret:
            raise TypeError("multiple values for argument '%s'" % arg_name)
        ret[arg_name] = arg_value
        manually_set_args.add(arg_name)

    for (arg_name, arg_value) in default_values.items():
        if arg_name not in ret:
            ret[arg_name] = arg_value

    return (ret, manually_set_args)


def train_supervised(*kargs, **kwargs):
    """
    Train a supervised model and return a model object.

    input must be a filepath. The input text does not need to be tokenized
    as per the tokenize function, but it must be preprocessed and encoded
    as UTF-8. You might want to consult standard preprocessing scripts such
    as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html

    The input file must must contain at least one label per line. For an
    example consult the example datasets which are part of the fastText
    repository such as the dataset pulled by classification-example.sh.
    """
    supervised_default = unsupervised_default.copy()
    supervised_default.update({
        'lr': 0.1,
        'minCount': 1,
        'minn': 0,
        'maxn': 0,
        'loss': "softmax",
        'model': "supervised"
    })

    arg_names = ['input', 'lr', 'dim', 'ws', 'epoch', 'minCount',
                 'minCountLabel', 'minn', 'maxn', 'neg', 'wordNgrams', 'loss', 'bucket',
                 'thread', 'lrUpdateRate', 't', 'label', 'verbose', 'pretrainedVectors',
                 'seed', 'autotuneValidationFile', 'autotuneMetric',
                 'autotunePredictions', 'autotuneDuration', 'autotuneModelSize']
    args, manually_set_args = read_args(kargs, kwargs, arg_names,
                                        supervised_default)
    a = _build_args(args, manually_set_args)
    ft = _FastText(args=a)
    fasttext.train(ft.f, a)
    ft.set_args(ft.f.getArgs())
    return ft


def train_unsupervised(*kargs, **kwargs):
    """
    Train an unsupervised model and return a model object.

    input must be a filepath. The input text does not need to be tokenized
    as per the tokenize function, but it must be preprocessed and encoded
    as UTF-8. You might want to consult standard preprocessing scripts such
    as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html

    The input field must not contain any labels or use the specified label prefix
    unless it is ok for those words to be ignored. For an example consult the
    dataset pulled by the example script word-vector-example.sh, which is
    part of the fastText repository.
    """
    arg_names = ['input', 'model', 'lr', 'dim', 'ws', 'epoch', 'minCount',
                 'minCountLabel', 'minn', 'maxn', 'neg', 'wordNgrams', 'loss', 'bucket',
                 'thread', 'lrUpdateRate', 't', 'label', 'verbose', 'pretrainedVectors']
    args, manually_set_args = read_args(kargs, kwargs, arg_names,
                                        unsupervised_default)
    a = _build_args(args, manually_set_args)
    ft = _FastText(args=a)
    fasttext.train(ft.f, a)
    ft.set_args(ft.f.getArgs())
    return ft


def cbow(*kargs, **kwargs):
    raise Exception("`cbow` is not supported any more. Please use `train_unsupervised` with model=`cbow`. For more information please refer to https://fasttext.cc/blog/2019/06/25/blog-post.html#2-you-were-using-the-unofficial-fasttext-module")


def skipgram(*kargs, **kwargs):
    raise Exception("`skipgram` is not supported any more. Please use `train_unsupervised` with model=`skipgram`. For more information please refer to https://fasttext.cc/blog/2019/06/25/blog-post.html#2-you-were-using-the-unofficial-fasttext-module")


def supervised(*kargs, **kwargs):
    raise Exception("`supervised` is not supported any more. Please use `train_supervised`. For more information please refer to https://fasttext.cc/blog/2019/06/25/blog-post.html#2-you-were-using-the-unofficial-fasttext-module")
