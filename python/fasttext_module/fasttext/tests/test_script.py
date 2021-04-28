# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from fasttext import train_supervised
from fasttext import train_unsupervised
from fasttext import util
import fasttext
import os
import subprocess
import unittest
import tempfile
import random
import sys
import copy
import numpy as np
try:
    import unicode
except ImportError:
    pass
from fasttext.tests.test_configurations import get_supervised_models


def eprint(cls, *args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_random_unicode(length):
    # See: https://stackoverflow.com/questions/1477294/generate-random-utf-8-string-in-python

    try:
        get_char = unichr
    except NameError:
        get_char = chr

    # Update this to include code point ranges to be sampled
    include_ranges = [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]

    alphabet = [
        get_char(code_point)
        for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]
    return ''.join(random.choice(alphabet) for i in range(length))


def get_random_words(N, a=1, b=20, unique=True):
    words = []
    while (len(words) < N):
        length = random.randint(a, b)
        word = get_random_unicode(length)
        if unique and word not in words:
            words.append(word)
        else:
            words.append(word)
    return words


def get_random_data(
    num_lines=100,
    max_vocab_size=100,
    min_words_line=0,
    max_words_line=20,
    min_len_word=1,
    max_len_word=10,
    unique_words=True,
):
    random_words = get_random_words(
        max_vocab_size, min_len_word, max_len_word, unique=unique_words
    )
    lines = []
    for _ in range(num_lines):
        line = []
        line_length = random.randint(min_words_line, max_words_line)
        for _ in range(line_length):
            i = random.randint(0, max_vocab_size - 1)
            line.append(random_words[i])
        line = " ".join(line)
        lines.append(line)
    return lines


def default_kwargs(kwargs):
    default = {"thread": 1, "epoch": 1, "minCount": 1, "bucket": 1000}
    for k, v in default.items():
        if k not in kwargs:
            kwargs[k] = v
    return kwargs


def build_unsupervised_model(data, kwargs):
    kwargs = default_kwargs(kwargs)
    with tempfile.NamedTemporaryFile(delete=False) as tmpf:
        for line in data:
            tmpf.write((line + "\n").encode("UTF-8"))
        tmpf.flush()
        model = train_unsupervised(input=tmpf.name, **kwargs)
    return model


def build_supervised_model(data, kwargs):
    kwargs = default_kwargs(kwargs)
    with tempfile.NamedTemporaryFile(delete=False) as tmpf:
        for line in data:
            line = "__label__" + line.strip() + "\n"
            tmpf.write(line.encode("UTF-8"))
        tmpf.flush()
        model = train_supervised(input=tmpf.name, **kwargs)
    return model


def read_labels(data_file):
    labels = []
    lines = []
    with open(data_file, 'r') as f:
        for line in f:
            labels_line = []
            words_line = []
            try:
                line = unicode(line, "UTF-8").split()
            except NameError:
                line = line.split()
            for word in line:
                if word.startswith("__label__"):
                    labels_line.append(word)
                else:
                    words_line.append(word)
            labels.append(labels_line)
            lines.append(" ".join(words_line))
    return lines, labels


class TestFastTextUnitPy(unittest.TestCase):
    # TODO: Unit test copy behavior of fasttext

    def gen_test_get_vector(self, kwargs):
        # Confirm if no subwords, OOV is zero, confirm min=10 means words < 10 get zeros

        f = build_unsupervised_model(get_random_data(100), kwargs)
        words, _ = f.get_words(include_freq=True)
        words += get_random_words(100)
        for word in words:
            f.get_word_vector(word)

    def gen_test_multi_get_line(self, kwargs):
        data = get_random_data(100)
        model1 = build_supervised_model(data, kwargs)
        model2 = build_unsupervised_model(data, kwargs)
        lines1 = []
        lines2 = []
        for line in data:
            words, labels = model1.get_line(line)
            lines1.append(words)
            self.assertEqual(len(labels), 0)
            words, labels = model2.get_line(line)
            lines2.append(words)
            self.assertEqual(len(labels), 0)
        all_lines1, all_labels1 = model1.get_line(data)
        all_lines2, all_labels2 = model2.get_line(data)
        self.assertEqual(lines1, all_lines1)
        self.assertEqual(lines2, all_lines2)
        for labels in all_labels1:
            self.assertEqual(len(labels), 0)
        for labels in all_labels2:
            self.assertEqual(len(labels), 0)

    def gen_test_supervised_util_test(self, kwargs):
        def check(data):
            third = int(len(data) / 3)
            train_data = data[:2 * third]
            valid_data = data[third:]
            with tempfile.NamedTemporaryFile(
                delete=False
            ) as tmpf, tempfile.NamedTemporaryFile(delete=False) as tmpf2:
                for line in train_data:
                    tmpf.write(
                        ("__label__" + line.strip() + "\n").encode("UTF-8")
                    )
                tmpf.flush()
                for line in valid_data:
                    tmpf2.write(
                        ("__label__" + line.strip() + "\n").encode("UTF-8")
                    )
                tmpf2.flush()
                model = train_supervised(input=tmpf.name, **kwargs)
                true_labels = []
                all_words = []
                with open(tmpf2.name, 'r') as fid:
                    for line in fid:
                        if sys.version_info < (3, 0):
                            line = line.decode("UTF-8")
                        if len(line.strip()) == 0:
                            continue
                        words, labels = model.get_line(line.strip())
                        if len(labels) == 0:
                            continue
                        all_words.append(" ".join(words))
                        true_labels += [labels]
                predictions, _ = model.predict(all_words)
                p, r = util.test(predictions, true_labels)
                N = len(predictions)
                Nt, pt, rt = model.test(tmpf2.name)
                self.assertEqual(N, Nt)
                self.assertEqual(p, pt)
                self.assertEqual(r, rt)

        # Need at least one word to have a label and a word to prevent error
        check(get_random_data(100, min_words_line=2))

    def gen_test_supervised_predict(self, kwargs):
        # Confirm number of labels, confirm labels for easy dataset
        # Confirm 1 label and 0 label dataset

        f = build_supervised_model(get_random_data(100), kwargs)
        words = get_random_words(100)
        for k in [1, 2, 5]:
            for w in words:
                labels, probs = f.predict(w, k)
            data = get_random_data(100)
            for line in data:
                labels, probs = f.predict(line, k)

    def gen_test_supervised_multiline_predict(self, kwargs):
        # Confirm number of labels, confirm labels for easy dataset
        # Confirm 1 label and 0 label dataset

        def check_predict(f):
            for k in [1, 2, 5]:
                words = get_random_words(10)
                agg_labels = []
                agg_probs = []
                for w in words:
                    labels, probs = f.predict(w, k)
                    agg_labels += [labels]
                    agg_probs += [probs]
                all_labels1, all_probs1 = f.predict(words, k)
                data = get_random_data(10)
                for line in data:
                    labels, probs = f.predict(line, k)
                    agg_labels += [labels]
                    agg_probs += [probs]
                all_labels2, all_probs2 = f.predict(data, k)
                all_labels = list(all_labels1) + list(all_labels2)
                all_probs = list(all_probs1) + list(all_probs2)
                for label1, label2 in zip(all_labels, agg_labels):
                    self.assertEqual(list(label1), list(label2))
                for prob1, prob2 in zip(all_probs, agg_probs):
                    self.assertEqual(list(prob1), list(prob2))

        check_predict(build_supervised_model(get_random_data(100), kwargs))
        check_predict(
            build_supervised_model(
                get_random_data(100, min_words_line=1), kwargs
            )
        )

    def gen_test_vocab(self, kwargs):
        # Confirm empty dataset, confirm all label dataset

        data = get_random_data(100)
        words_python = {}
        for line in data:
            line_words = line.split()
            for w in line_words:
                if w not in words_python:
                    words_python[w] = 0
                words_python[w] += 1
        f = build_unsupervised_model(data, kwargs)
        words, freqs = f.get_words(include_freq=True)
        foundEOS = False
        for word, freq in zip(words, freqs):
            if word == fasttext.EOS:
                foundEOS = True
            else:
                self.assertEqual(words_python[word], freq)
        # EOS is special to fasttext, but still part of the vocab
        self.assertEqual(len(words_python), len(words) - 1)
        self.assertTrue(foundEOS)

        # Should cause "Empty vocabulary" error.
        data = get_random_data(0)
        gotError = False
        try:
            build_unsupervised_model(data, kwargs)
        except ValueError:
            gotError = True
        self.assertTrue(gotError)

    def gen_test_subwords(self, kwargs):
        # Define expected behavior
        f = build_unsupervised_model(get_random_data(100), kwargs)
        words, _ = f.get_words(include_freq=True)
        words += get_random_words(10, 1, 10)
        for w in words:
            f.get_subwords(w)

    def gen_test_tokenize(self, kwargs):
        self.assertEqual(["asdf", "asdb"], fasttext.tokenize("asdf asdb"))
        self.assertEqual(["asdf"], fasttext.tokenize("asdf"))
        self.assertEqual([fasttext.EOS], fasttext.tokenize("\n"))
        self.assertEqual(["asdf", fasttext.EOS], fasttext.tokenize("asdf\n"))
        self.assertEqual([], fasttext.tokenize(""))
        self.assertEqual([], fasttext.tokenize(" "))
        # An empty string is not a token (it's just whitespace)
        # So the minimum length must be 1
        words = get_random_words(100, 1, 20)
        self.assertEqual(words, fasttext.tokenize(" ".join(words)))

    def gen_test_unsupervised_dimension(self, kwargs):
        if "dim" in kwargs:
            f = build_unsupervised_model(get_random_data(100), kwargs)
            self.assertEqual(f.get_dimension(), kwargs["dim"])

    def gen_test_supervised_dimension(self, kwargs):
        if "dim" in kwargs:
            f = build_supervised_model(get_random_data(100), kwargs)
            self.assertEqual(f.get_dimension(), kwargs["dim"])

    def gen_test_subword_vector(self, kwargs):
        f = build_unsupervised_model(get_random_data(100), kwargs)
        words, _ = f.get_words(include_freq=True)
        words += get_random_words(100, 1, 20)
        input_matrix = f.get_input_matrix()
        for word in words:
            # Universal API to get word vector
            vec1 = f.get_word_vector(word)

            # Build word vector from subwords
            subwords, subinds = f.get_subwords(word)
            subvectors = list(map(lambda x: f.get_input_vector(x), subinds))
            if len(subvectors) == 0:
                vec2 = np.zeros((f.get_dimension(), ))
            else:
                subvectors = np.vstack(subvectors)
                vec2 = np.sum((subvectors / len(subwords)), 0)

            # Build word vector from subinds
            if len(subinds) == 0:
                vec3 = np.zeros((f.get_dimension(), ))
            else:
                vec3 = np.sum(input_matrix[subinds] / len(subinds), 0)

            # Build word vectors from word and subword ids
            wid = f.get_word_id(word)
            if wid >= 0:
                swids = list(map(lambda x: f.get_subword_id(x), subwords[1:]))
                swids.append(wid)
            else:
                swids = list(map(lambda x: f.get_subword_id(x), subwords))
            if len(swids) == 0:
                vec4 = np.zeros((f.get_dimension(), ))
            else:
                swids = np.array(swids)
                vec4 = np.sum(input_matrix[swids] / len(swids), 0)

            self.assertTrue(np.isclose(vec1, vec2, atol=1e-5, rtol=0).all())
            self.assertTrue(np.isclose(vec2, vec3, atol=1e-5, rtol=0).all())
            self.assertTrue(np.isclose(vec3, vec4, atol=1e-5, rtol=0).all())
            self.assertTrue(np.isclose(vec4, vec1, atol=1e-5, rtol=0).all())

    def gen_test_unsupervised_get_words(self, kwargs):
        # Check more corner cases of 0 vocab, empty file etc.
        f = build_unsupervised_model(get_random_data(100), kwargs)
        words1, freq1 = f.get_words(include_freq=True)
        words2 = f.get_words(include_freq=False)
        self.assertEqual(len(words1), len(words2))
        self.assertEqual(len(words1), len(freq1))

    def gen_test_supervised_get_words(self, kwargs):
        f = build_supervised_model(get_random_data(100), kwargs)
        words1, freq1 = f.get_words(include_freq=True)
        words2 = f.get_words(include_freq=False)
        self.assertEqual(len(words1), len(words2))
        self.assertEqual(len(words1), len(freq1))

    def gen_test_unsupervised_get_labels(self, kwargs):
        f = build_unsupervised_model(get_random_data(100), kwargs)
        labels1, freq1 = f.get_labels(include_freq=True)
        labels2 = f.get_labels(include_freq=False)
        words2 = f.get_words(include_freq=False)
        self.assertEqual(len(labels1), len(labels2))
        self.assertEqual(len(labels1), len(freq1))
        self.assertEqual(len(labels1), len(words2))
        for w1, w2 in zip(labels2, words2):
            self.assertEqual(w1, w2)

    def gen_test_supervised_get_labels(self, kwargs):
        f = build_supervised_model(get_random_data(100), kwargs)
        labels1, freq1 = f.get_labels(include_freq=True)
        labels2 = f.get_labels(include_freq=False)
        self.assertEqual(len(labels1), len(labels2))
        self.assertEqual(len(labels1), len(freq1))

    def gen_test_unsupervised_exercise_is_quant(self, kwargs):
        f = build_unsupervised_model(get_random_data(100), kwargs)
        gotError = False
        try:
            f.quantize()
        except ValueError:
            gotError = True
        self.assertTrue(gotError)

    def gen_test_supervised_exercise_is_quant(self, kwargs):
        f = build_supervised_model(
            get_random_data(1000, max_vocab_size=1000), kwargs
        )
        self.assertTrue(not f.is_quantized())
        f.quantize()
        self.assertTrue(f.is_quantized())

    def gen_test_newline_predict_sentence(self, kwargs):
        f = build_supervised_model(get_random_data(100), kwargs)
        sentence = " ".join(get_random_words(20))
        f.predict(sentence, k=5)
        sentence += "\n"
        gotError = False
        try:
            f.predict(sentence, k=5)
        except ValueError:
            gotError = True
        self.assertTrue(gotError)

        f = build_supervised_model(get_random_data(100), kwargs)
        sentence = " ".join(get_random_words(20))
        f.get_sentence_vector(sentence)
        sentence += "\n"
        gotError = False
        try:
            f.get_sentence_vector(sentence)
        except ValueError:
            gotError = True
        self.assertTrue(gotError)


# Generate a supervised test case
# The returned function will be set as an attribute to a test class
def gen_sup_test(configuration, data_dir):
    def sup_test(self):
        def get_path_size(path):
            path_size = subprocess.check_output(["stat", "-c", "%s",
                                                 path]).decode('utf-8')
            path_size = int(path_size)
            return path_size

        def check(model, model_filename, test, lessthan, msg_prefix=""):
            N_local_out, p1_local_out, r1_local_out = model.test(test["data"])
            self.assertEqual(
                N_local_out, test["n"], msg_prefix + "N: Want: " +
                str(test["n"]) + " Is: " + str(N_local_out)
            )
            self.assertTrue(
                p1_local_out >= test["p1"], msg_prefix + "p1: Want: " +
                str(test["p1"]) + " Is: " + str(p1_local_out)
            )
            self.assertTrue(
                r1_local_out >= test["r1"], msg_prefix + "r1: Want: " +
                str(test["r1"]) + " Is: " + str(r1_local_out)
            )
            path_size = get_path_size(model_filename)
            size_msg = str(test["size"]) + " Is: " + str(path_size)
            if lessthan:
                self.assertTrue(
                    path_size <= test["size"],
                    msg_prefix + "Size: Want at most: " + size_msg
                )
            else:
                self.assertTrue(
                    path_size == test["size"],
                    msg_prefix + "Size: Want: " + size_msg
                )

        configuration["args"]["input"] = os.path.join(
            data_dir, configuration["args"]["input"]
        )
        configuration["quant_args"]["input"] = configuration["args"]["input"]
        configuration["test"]["data"] = os.path.join(
            data_dir, configuration["test"]["data"]
        )
        configuration["quant_test"]["data"] = configuration["test"]["data"]
        output = os.path.join(tempfile.mkdtemp(), configuration["dataset"])
        print()
        model = train_supervised(**configuration["args"])
        model.save_model(output + ".bin")
        check(
            model,
            output + ".bin",
            configuration["test"],
            False,
            msg_prefix="Supervised: "
        )
        print()
        model.quantize(**configuration["quant_args"])
        model.save_model(output + ".ftz")
        check(
            model,
            output + ".ftz",
            configuration["quant_test"],
            True,
            msg_prefix="Quantized: "
        )

    return sup_test


def gen_unit_tests(verbose=0):
    gen_funcs = [
        func for func in dir(TestFastTextUnitPy)
        if callable(getattr(TestFastTextUnitPy, func))
        if func.startswith("gen_test_")
    ]
    general_settings = [
        {
            "minn": 2,
            "maxn": 4,
        }, {
            "minn": 0,
            "maxn": 0,
            "bucket": 0
        }, {
            "dim": 1
        }, {
            "dim": 5
        }
    ]
    supervised_settings = [
        {
            "minn": 2,
            "maxn": 4,
        }, {
            "minn": 0,
            "maxn": 0,
            "bucket": 0
        }, {
            "dim": 1
        }, {
            "dim": 5
        }, {
            "dim": 5,
            "loss": "hs"
        }
    ]
    unsupervised_settings = [
        {
            "minn": 2,
            "maxn": 4,
        }, {
            "minn": 0,
            "maxn": 0,
            "bucket": 0
        }, {
            "dim": 1
        }, {
            "dim": 5,
            "model": "cbow"
        }, {
            "dim": 5,
            "model": "skipgram"
        }
    ]
    for gen_func in gen_funcs:

        def build_test(test_name, kwargs=None):
            if kwargs is None:
                kwargs = {}
            kwargs["verbose"] = verbose

            def test(self):
                return getattr(TestFastTextUnitPy,
                               "gen_" + test_name)(self, copy.deepcopy(kwargs))

            return test

        test_name = gen_func[4:]
        if "_unsupervised_" in test_name:
            for i, setting in enumerate(unsupervised_settings):
                setattr(
                    TestFastTextUnitPy, test_name + "_" + str(i),
                    build_test(test_name, setting)
                )
        elif "_supervised_" in test_name:
            for i, setting in enumerate(supervised_settings):
                setattr(
                    TestFastTextUnitPy, test_name + "_" + str(i),
                    build_test(test_name, setting)
                )
        else:
            for i, setting in enumerate(general_settings):
                setattr(
                    TestFastTextUnitPy, test_name + "_" + str(i),
                    build_test(test_name, setting)
                )

    return TestFastTextUnitPy


def gen_tests(data_dir, verbose=1):
    class TestFastTextPy(unittest.TestCase):
        pass

    i = 0
    for configuration in get_supervised_models(verbose=verbose):
        setattr(
            TestFastTextPy,
            "test_sup_" + str(i) + "_" + configuration["dataset"],
            gen_sup_test(configuration, data_dir)
        )
        i += 1
    return TestFastTextPy
