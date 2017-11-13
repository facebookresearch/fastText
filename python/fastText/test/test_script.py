# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from fastText import train_supervised
from fastText import train_unsupervised
from fastText import load_model
from fastText import tokenize
import random
import sys
import os
import subprocess
import multiprocessing
import numpy as np
import unittest
import tempfile
import math
from scipy import stats


def compat_splitting(line):
    return line.decode('utf8').split()


def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2


def read_vectors(model_path):
    vectors = {}
    with open(model_path, 'rb') as fin:
        for _, line in enumerate(fin):
            try:
                tab = compat_splitting(line)
                vec = np.array(tab[1:], dtype=float)
                word = tab[0]
                if np.linalg.norm(vec) == 0:
                    continue
                if word not in vectors:
                    vectors[word] = vec
            except ValueError:
                continue
            except UnicodeDecodeError:
                continue
    return vectors


def compute_similarity(model_path, data_path, vectors=None):
    if not vectors:
        vectors = read_vectors(model_path)

    mysim = []
    gold = []
    drop = 0.0
    nwords = 0.0

    with open(data_path, 'rb') as fin:
        for line in fin:
            tline = compat_splitting(line)
            word1 = tline[0].lower()
            word2 = tline[1].lower()
            nwords = nwords + 1.0

            if (word1 in vectors) and (word2 in vectors):
                v1 = vectors[word1]
                v2 = vectors[word2]
                d = similarity(v1, v2)
                mysim.append(d)
                gold.append(float(tline[2]))
            else:
                drop = drop + 1.0

    corr = stats.spearmanr(mysim, gold)
    dataset = os.path.basename(data_path)
    correlation = corr[0] * 100
    oov = math.ceil(drop / nwords * 100.0)
    return dataset, correlation, oov


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


def get_random_words(N, a, b):
    words = []
    for _ in range(N):
        length = random.randint(a, b)
        words.append(get_random_unicode(length))
    return words


class TestFastTextPy(unittest.TestCase):
    @classmethod
    def eprint(cls, *args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    @classmethod
    def num_thread(cls):
        return multiprocessing.cpu_count() - 1

    @classmethod
    def build_paths(cls, train, test, output):
        train = os.path.join(cls.data_dir, train)
        test = os.path.join(cls.data_dir, test)
        output = os.path.join(cls.result_dir, output)
        return train, test, output

    @classmethod
    def build_train_args(cls, params, mode, train, output):
        args = [cls.bin, mode, "-input", train, "-output", output]
        return args + params.split(' ')

    @classmethod
    def get_train_output(cls, train_args):
        cls.eprint("Executing: " + ' '.join(train_args))
        return subprocess.check_output(train_args).decode('utf-8')

    @classmethod
    def get_path_size(cls, path):
        path_size = subprocess.check_output(["stat", "-c", "%s",
                                             path]).decode('utf-8')
        path_size = int(path_size)
        return path_size

    @classmethod
    def default_test_args(cls, model, test, quantize=False):
        return [cls.bin, "test", model, test]

    @classmethod
    def get_test_output(cls, test_args):
        cls.eprint("Executing: " + ' '.join(test_args))
        test_output = subprocess.check_output(test_args)
        test_output = test_output.decode('utf-8')
        cls.eprint("Test output:\n" + test_output)
        return list(
            map(lambda x: x.split('\t')[1], test_output.split('\n')[:-1])
        )

    @classmethod
    def train_generic_classifier(cls, train, output):
        thread = cls.num_thread()
        cls.eprint("Using {} threads".format(thread))
        sup_params = (
            "-dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 "
            "-epoch 5 -thread {}".format(thread)
        )
        mode = 'supervised'
        cls.get_train_output(
            cls.build_train_args(sup_params, mode, train, output)
        )

    @classmethod
    def train_generic_embeddings(cls, train, output):
        thread = cls.num_thread()
        cls.eprint("Using {} threads".format(thread))
        unsup_params = (
            "-thread {} -lr 0.025 -dim 100 -ws 5 -epoch 1 -minCount 5 "
            "-neg 5 -loss ns -bucket 2000000 -minn 3 -maxn 6 -t 1e-4 "
            "-lrUpdateRate 100".format(thread)
        )
        mode = 'cbow'
        cls.get_train_output(
            cls.build_train_args(unsup_params, mode, train, output)
        )

    def get_predictions_from_list(self, output, words, k):
        args = [self.bin, "predict-prob", output + '.bin', '-', str(k)]
        self.eprint("Executing: " + ' '.join(args))
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        test_text = ""
        if words:
            test_text = '\n'.join(words) + '\n'
        test_text = test_text.encode('utf-8')
        stdout, stderr = p.communicate(test_text)
        stdout = stdout.decode('utf-8')
        return stdout, stderr, p.returncode

    def get_word_vectors_from_list(self, output, words):
        args = [self.bin, "print-word-vectors", output + '.bin']
        self.eprint("Executing: " + ' '.join(args))
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        test_text = '\n'.join(words).encode('utf-8')
        stdout, stderr = p.communicate(test_text)
        return stdout


class TestFastTextPyUnit(TestFastTextPy):
    @classmethod
    def setUpClass(cls):
        cls.bin = os.environ['FASTTEXT_BIN']
        cls.data_dir = os.environ['FASTTEXT_DATA']
        cls.result_dir = tempfile.mkdtemp()
        train, _, output = cls.build_paths("fil9", "rw/rw.txt", "fil9")
        cls.train_generic_embeddings(train, output)
        cls.output = output
        train, _, output_sup = cls.build_paths(
            "dbpedia.train", "dbpedia.test", "dbpedia"
        )
        cls.train_generic_classifier(train, output_sup)
        cls.output_sup = output_sup

    @classmethod
    def tearDownClass(cls):
        pass
        # shutil.rmtree(cls.result_dir)

    # Check if get_word_vector aligns with vectors from stdin
    def test_getvector(self):
        f = load_model(self.output + '.bin')
        words, _ = f.get_words(include_freq=True)
        words += get_random_words(100, 1, 100)
        ftbin_vectors = self.get_word_vectors_from_list(self.output, words)
        ftbin_vectors = ftbin_vectors.decode('utf-8').split('\n')[:-1]
        for v in ftbin_vectors:
            word = v.split(' ')[0]
            vector = v.split(' ')[1:-1]
            vector = np.array(list(map(float, vector)))
            pvec = f.get_word_vector(word)
            # The fasttext cli returns floats with 5 digits,
            # but we use the full 6 digits.
            self.assertTrue(np.allclose(vector, pvec, rtol=1e-04))

    def test_predict(self):
        # TODO: I went a little crazy here as an exercise for
        # a rigorous test case. This could be turned into
        # a few utility functions.
        f = load_model(self.output_sup + '.bin')

        def _test(N, min_length, max_length, k, add_vocab=0):
            words = get_random_words(N, min_length, max_length)
            if add_vocab > 0:
                vocab, _ = f.get_words(include_freq=True)
                for _ in range(add_vocab):
                    ind = random.randint(0, len(vocab))
                    words += [vocab[ind]]
            all_labels = []
            all_probs = []
            ii = 0
            gotError = False
            for w in words:
                try:
                    labels, probs = f.predict(w, k)
                except ValueError:
                    gotError = True
                    continue
                all_labels.append(labels)
                all_probs.append(probs)
                ii += 1
            preds, _, retcode = self.get_predictions_from_list(
                self.output_sup, words, k
            )
            if gotError and retcode == 0:
                self.eprint(
                    "Didn't get error. Make sure your compiled "
                    "binary kept the assert statements"
                )
                self.assertTrue(False)
            else:
                return
            preds = preds.split('\n')[:-1]
            self.assertEqual(len(preds), len(all_labels))
            for i in range(len(preds)):
                labels = preds[i].split()
                probs = np.array(list(map(float, labels[1::2])))
                labels = np.array(labels[::2])
                self.assertTrue(np.allclose(probs, all_probs[i], rtol=1e-04))
                self.assertTrue(np.array_equal(labels, all_labels[i]))

        _test(0, 0, 0, 0)
        _test(1, 0, 0, 0)
        _test(10, 0, 0, 0)
        _test(1, 1, 1, 0)
        _test(1, 1, 1, 1)
        _test(1, 2, 3, 0)
        _test(1, 2, 3, 1)
        _test(10, 1, 1, 1)
        _test(1, 1, 1, 0, add_vocab=10)
        _test(1, 1, 1, 1, add_vocab=10)
        _test(1, 2, 3, 0, add_vocab=10)
        _test(1, 2, 3, 1, add_vocab=10)
        reach = 10
        for _ in range(10):
            N = random.randint(0, reach)
            init = random.randint(0, reach)
            offset = random.randint(0, reach)
            k = random.randint(0, reach)
            _test(N, init, init + offset, k)

    def test_vocab(self):
        f = load_model(self.output + '.bin')
        words, freq = f.get_words(include_freq=True)
        self.eprint(
            "There is no way to access words from the cli yet. "
            "Therefore there can be no rigorous test."
        )

    def test_subwords(self):
        f = load_model(self.output + '.bin')
        words, _ = f.get_words(include_freq=True)
        words += get_random_words(10, 1, 10)
        for w in words:
            f.get_subwords(w)
        self.eprint(
            "There is no way to access words from the cli yet. "
            "Therefore there can be no test."
        )

    def test_tokenize(self):
        train, _, _ = self.build_paths("fil9", "rw/rw.txt", "fil9")
        with open(train, 'r') as f:
            _ = tokenize(f.read())

    def test_dimension(self):
        f = load_model(self.output + '.bin')
        f.get_dimension()

    def test_subword_vector(self):
        f = load_model(self.output + '.bin')
        words, _ = f.get_words(include_freq=True)
        words += get_random_words(10000, 1, 200)
        input_matrix = f.get_input_matrix()
        for word in words:

            # Universal api to get word vector
            vec1 = f.get_word_vector(word)

            # Build word vector from subwords
            subwords, subinds = f.get_subwords(word)
            subvectors = list(map(lambda x: f.get_input_vector(x), subinds))
            subvectors = np.stack(subvectors)
            vec2 = np.sum((subvectors / len(subwords)), 0)

            # Build word vector from subinds
            vec3 = np.sum(input_matrix[subinds] / len(subinds), 0)

            # Build word vectors from word and subword ids
            wid = f.get_word_id(word)
            if wid >= 0:
                swids = list(map(lambda x: f.get_subword_id(x), subwords[1:]))
                swids.append(wid)
            else:
                swids = list(map(lambda x: f.get_subword_id(x), subwords))
            swids = np.array(swids)
            vec4 = np.sum(input_matrix[swids] / len(swids), 0)

            self.assertTrue(np.isclose(vec1, vec2, atol=1e-5, rtol=0).all())
            self.assertTrue(np.isclose(vec2, vec3, atol=1e-5, rtol=0).all())
            self.assertTrue(np.isclose(vec3, vec4, atol=1e-5, rtol=0).all())
            self.assertTrue(np.isclose(vec4, vec1, atol=1e-5, rtol=0).all())

    # TODO: Compare with .vec file
    def test_get_words(self):
        f = load_model(self.output + '.bin')
        words1, freq1 = f.get_words(include_freq=True)
        words2 = f.get_words(include_freq=False)
        self.assertEqual(len(words1), len(words2))
        self.assertEqual(len(words1), len(freq1))
        f = load_model(self.output_sup + '.bin')
        words1, freq1 = f.get_words(include_freq=True)
        words2 = f.get_words(include_freq=False)
        self.assertEqual(len(words1), len(words2))
        self.assertEqual(len(words1), len(freq1))

    # TODO: Compare with .vec file for unsup
    def test_get_labels(self):
        f = load_model(self.output + '.bin')
        labels1, freq1 = f.get_labels(include_freq=True)
        labels2 = f.get_labels(include_freq=False)
        words2 = f.get_words(include_freq=False)
        self.assertEqual(len(labels1), len(labels2))
        self.assertEqual(len(labels1), len(freq1))
        self.assertEqual(len(labels1), len(words2))
        for w1, w2 in zip(labels2, words2):
            self.assertEqual(w1, w2)
        f = load_model(self.output_sup + '.bin')
        labels1, freq1 = f.get_labels(include_freq=True)
        labels2 = f.get_labels(include_freq=False)
        self.assertEqual(len(labels1), len(labels2))
        self.assertEqual(len(labels1), len(freq1))

    def test_exercise_is_quant(self):
        f = load_model(self.output + '.bin')
        gotError = False
        try:
            f.quantize()
        except ValueError:
            gotError = True
        self.assertTrue(gotError)
        f = load_model(self.output_sup + '.bin')
        self.assertTrue(not f.is_quantized())
        f.quantize()
        self.assertTrue(f.is_quantized())

    def test_newline_predict_sentence(self):
        f = load_model(self.output_sup + '.bin')
        sentence = get_random_words(1, 1000, 2000)[0]
        f.predict(sentence, k=5)
        sentence += "\n"
        gotError = False
        try:
            f.predict(sentence, k=5)
        except ValueError:
            gotError = True
        self.assertTrue(gotError)

        f = load_model(self.output + '.bin')
        sentence = get_random_words(1, 1000, 2000)[0]
        f.get_sentence_vector(sentence)
        sentence += "\n"
        gotError = False
        try:
            f.get_sentence_vector(sentence)
        except ValueError:
            gotError = True
        self.assertTrue(gotError)


class TestFastTextPyIntegration(TestFastTextPy):
    @classmethod
    def setUpClass(cls):
        cls.bin = os.environ['FASTTEXT_BIN']
        cls.data_dir = os.environ['FASTTEXT_DATA']
        cls.result_dir = tempfile.mkdtemp()

    def test_unsup1(self):
        train, test, output = self.build_paths("fil9", "rw/rw.txt", "fil9")

        model = train_unsupervised(
            input=train,
            model="skipgram",
            lr=0.025,
            dim=100,
            ws=5,
            epoch=1,
            minCount=5,
            neg=5,
            loss="ns",
            bucket=2000000,
            minn=3,
            maxn=6,
            t=1e-4,
            lrUpdateRate=100,
            thread=self.num_thread(),
        )
        model.save_model(output)

        path_size = self.get_path_size(output)
        vectors = {}
        with open(test, 'r') as test_f:
            for line in test_f:
                query0 = line.split()[0].strip()
                query1 = line.split()[1].strip()
                vector0 = model.get_word_vector(query0)
                vector1 = model.get_word_vector(query1)
                vectors[query0] = vector0
                vectors[query1] = vector1
        dataset, correlation, oov = compute_similarity(None, test, vectors)
        correlation = np.around(correlation)

        self.assertTrue(
            correlation >= 41, "Correlation: Want: 41 Is: " + str(correlation)
        )
        self.assertEqual(oov, 0.0, "Oov: Want: 0 Is: " + str(oov))
        self.assertEqual(
            path_size, 978480868, "Size: Want: 978480868 Is: " + str(path_size)
        )


def gen_sup_test(lr, dataset, n, p1, r1, p1_q, r1_q, size, quant_size):
    def sup_test(self):
        def check(
            output_local, test_local, n_local, p1_local, r1_local, size_local,
            lessthan
        ):
            test_args = self.default_test_args(output_local, test_local)
            test_output = self.get_test_output(test_args)
            self.assertEqual(
                str(test_output[0]),
                str(n_local),
                "N: Want: " + str(n_local) + " Is: " + str(test_output[0])
            )
            self.assertTrue(
                float(test_output[1]) >= float(p1_local),
                "p1: Want: " + str(p1_local) + " Is: " + str(test_output[1])
            )
            self.assertTrue(
                float(test_output[2]) >= float(r1_local),
                "r1: Want: " + str(r1_local) + " Is: " + str(test_output[2])
            )
            path_size = self.get_path_size(output_local)
            if lessthan:
                self.assertTrue(
                    path_size <= size_local, "Size: Want at most: " +
                    str(size_local) + " Is: " + str(path_size)
                )
            else:
                self.assertTrue(
                    path_size == size_local,
                    "Size: Want: " + str(size_local) + " Is: " + str(path_size)
                )

        train, test, output = self.build_paths(
            dataset + ".train", dataset + ".test", dataset
        )
        model = train_supervised(
            input=train,
            dim=10,
            lr=lr,
            wordNgrams=2,
            minCount=1,
            bucket=10000000,
            epoch=5,
            thread=self.num_thread()
        )
        model.save_model(output)
        check(output, test, n, p1, r1, size, False)
        # Exercising
        model.predict("hello world")
        model.quantize(input=train, retrain=True, cutoff=100000, qnorm=True)
        model.save_model(output + ".ftz")
        # Exercising
        model.predict("hello world")
        check(output + ".ftz", test, n, p1_q, r1_q, quant_size, True)

    return sup_test


if __name__ == "__main__":
    sup_job_lr = [0.25, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05]
    sup_job_n = [7600, 60000, 70000, 38000, 50000, 60000, 650000, 400000]
    sup_job_p1 = [0.921, 0.968, 0.984, 0.956, 0.638, 0.723, 0.603, 0.946]
    sup_job_r1 = [0.921, 0.968, 0.984, 0.956, 0.638, 0.723, 0.603, 0.946]
    sup_job_quant_p1 = [0.918, 0.965, 0.984, 0.953, 0.629, 0.707, 0.58, 0.940]
    sup_job_quant_r1 = [0.918, 0.965, 0.984, 0.953, 0.629, 0.707, 0.58, 0.940]
    sup_job_size = [
        405607193, 421445471, 447481878, 427867393, 431292576, 517549567,
        483742593, 493604598
    ]
    sup_job_quant_size = [
        405607193, 421445471, 447481878, 427867393, 431292576, 517549567,
        483742593, 493604598
    ]
    sup_job_quant_size = [
        1600000, 1457000, 1690000, 1550000, 1567896, 1655000, 1600000, 1575010
    ]
    # Yelp_review_full can be a bit flaky
    sup_job_dataset = [
        "ag_news", "sogou_news", "dbpedia", "yelp_review_polarity",
        "yelp_review_full", "yahoo_answers", "amazon_review_full",
        "amazon_review_polarity"
    ]
    sup_job_args = [
        sup_job_lr, sup_job_dataset, sup_job_n, sup_job_p1, sup_job_r1,
        sup_job_quant_p1, sup_job_quant_r1, sup_job_size, sup_job_quant_size
    ]
    for lr, dataset, n, p1, r1, p1_q, r1_q, size, quant_size in zip(
        *sup_job_args
    ):
        setattr(
            TestFastTextPyIntegration, "test_" + dataset,
            gen_sup_test(lr, dataset, n, p1, r1, p1_q, r1_q, size, quant_size)
        )
    unittest.main()
