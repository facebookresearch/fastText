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

loss_name = fasttext.loss_name
model_name = fasttext.model_name
EOS = "</s>"
BOW = "<"
EOW = ">"


class _FastText():
    """
    This class defines the API to inspect models and should not be used to
    create objects. It will be returned by functions such as load_model or
    train.

    In general this API assumes to be given only unicode for Python2 and the
    Python3 equvalent called str for any string-like arguments. All unicode
    strings are then encoded as UTF-8 and fed to the fastText C++ API.
    """

    def __init__(self, model=None):
        self.f = fasttext.fasttext()
        if model is not None:
            self.f.loadModel(model)

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

    def get_word_id(self, word):
        """
        Given a word, get the word id within the dictionary.
        Returns -1 if word is not in the dictionary.
        """
        return self.f.getWordId(word)

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
            predictions = self.f.multilinePredict(text, k, threshold, on_unicode_error)
            dt = np.dtype([('probability', 'float64'), ('label', 'object')])
            result_as_pair = np.array(predictions, dtype=dt)

            return result_as_pair['label'].tolist(), result_as_pair['probability']
        else:
            text = check(text)
            predictions = self.f.predict(text, k, threshold, on_unicode_error)
            probs, labels = zip(*predictions)

            return labels, np.array(probs, copy=False)

    def get_input_matrix(self):
        """
        Get a copy of the full input matrix of a Model. This only
        works if the model is not quantized.
        """
        if self.f.isQuant():
            raise ValueError("Can't get quantized Matrix")
        return np.array(self.f.getInputMatrix())

    def get_output_matrix(self):
        """
        Get a copy of the full output matrix of a Model. This only
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

    def test(self, path, k=1):
        """Evaluate supervised model using file given by path"""
        return self.f.test(path, k)

    def test_label(self, path, k=1, threshold=0.0):
        """
        Return the precision and recall score for each label.

        The returned value is a dictionary, where the key is the label.
        For example:
        f.test_label(...)
        {'__label__italian-cuisine' : {'precision' : 0.7, 'recall' : 0.74}}
        """
        return self.f.testLabel(path, k, threshold)

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


# TODO:
# Not supported:
# - pretrained vectors


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


def _build_args(args):
    args["model"] = _parse_model_string(args["model"])
    args["loss"] = _parse_loss_string(args["loss"])
    a = fasttext.args()
    for (k, v) in args.items():
        setattr(a, k, v)
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
    return _FastText(path)


def train_supervised(
    input,
    lr=0.1,
    dim=100,
    ws=5,
    epoch=5,
    minCount=1,
    minCountLabel=0,
    minn=0,
    maxn=0,
    neg=5,
    wordNgrams=1,
    loss="softmax",
    bucket=2000000,
    thread=multiprocessing.cpu_count() - 1,
    lrUpdateRate=100,
    t=1e-4,
    label="__label__",
    verbose=2,
    pretrainedVectors="",
):
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
    model = "supervised"
    a = _build_args(locals())
    ft = _FastText()
    fasttext.train(ft.f, a)
    return ft


def train_unsupervised(
    input,
    model="skipgram",
    lr=0.05,
    dim=100,
    ws=5,
    epoch=5,
    minCount=5,
    minCountLabel=0,
    minn=3,
    maxn=6,
    neg=5,
    wordNgrams=1,
    loss="ns",
    bucket=2000000,
    thread=multiprocessing.cpu_count() -1,
    lrUpdateRate=100,
    t=1e-4,
    label="__label__",
    verbose=2,
    pretrainedVectors="",
):
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
    a = _build_args(locals())
    ft = _FastText()
    fasttext.train(ft.f, a)
    return ft
