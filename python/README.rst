fastText |CircleCI|
===================

`fastText <https://fasttext.cc/>`__ is a library for efficient learning
of word representations and sentence classification.

In this document we present how to use fastText in python.

Table of contents
-----------------

-  `Requirements <#requirements>`__
-  `Installation <#installation>`__
-  `Usage overview <#usage-overview>`__
-  `Word representation model <#word-representation-model>`__
-  `Text classification model <#text-classification-model>`__
-  `IMPORTANT: Preprocessing data / encoding
   conventions <#important-preprocessing-data-encoding-conventions>`__
-  `More examples <#more-examples>`__
-  `API <#api>`__
-  `train_unsupervised parameters <#train_unsupervised-parameters>`__
-  `train_supervised parameters <#train_supervised-parameters>`__
-  `model object <#model-object>`__

Requirements
============

`fastText <https://fasttext.cc/>`__ builds on modern Mac OS and Linux
distributions. Since it uses C++11 features, it requires a compiler with
good C++11 support. You will need `Python <https://www.python.org/>`__
(version 2.7 or ≥ 3.4), `NumPy <http://www.numpy.org/>`__ &
`SciPy <https://www.scipy.org/>`__ and
`pybind11 <https://github.com/pybind/pybind11>`__.

Installation
============

To install the latest release, you can do :

.. code:: bash

    $ pip install fasttext

or, to get the latest development version of fasttext, you can install
from our github repository :

.. code:: bash

    $ git clone https://github.com/facebookresearch/fastText.git
    $ cd fastText
    $ sudo pip install .
    $ # or :
    $ sudo python setup.py install

Usage overview
==============

Word representation model
-------------------------

In order to learn word vectors, as `described
here <https://fasttext.cc/docs/en/references.html#enriching-word-vectors-with-subword-information>`__,
we can use ``fasttext.train_unsupervised`` function like this:

.. code:: py

    import fasttext

    # Skipgram model :
    model = fasttext.train_unsupervised('data.txt', model='skipgram')

    # or, cbow model :
    model = fasttext.train_unsupervised('data.txt', model='cbow')

where ``data.txt`` is a training file containing utf-8 encoded text.

The returned ``model`` object represents your learned model, and you can
use it to retrieve information.

.. code:: py

    print(model.words)   # list of words in dictionary
    print(model['king']) # get the vector of the word 'king'

Saving and loading a model object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can save your trained model object by calling the function
``save_model``.

.. code:: py

    model.save_model("model_filename.bin")

and retrieve it later thanks to the function ``load_model`` :

.. code:: py

    model = fasttext.load_model("model_filename.bin")

For more information about word representation usage of fasttext, you
can refer to our `word representations
tutorial <https://fasttext.cc/docs/en/unsupervised-tutorial.html>`__.

Text classification model
-------------------------

In order to train a text classifier using the method `described
here <https://fasttext.cc/docs/en/references.html#bag-of-tricks-for-efficient-text-classification>`__,
we can use ``fasttext.train_supervised`` function like this:

.. code:: py

    import fasttext

    model = fasttext.train_supervised('data.train.txt')

where ``data.train.txt`` is a text file containing a training sentence
per line along with the labels. By default, we assume that labels are
words that are prefixed by the string ``__label__``

Once the model is trained, we can retrieve the list of words and labels:

.. code:: py

    print(model.words)
    print(model.labels)

To evaluate our model by computing the precision at 1 (P@1) and the
recall on a test set, we use the ``test`` function:

.. code:: py

    def print_results(N, p, r):
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))

    print_results(*model.test('test.txt'))

We can also predict labels for a specific text :

.. code:: py

    model.predict("Which baking dish is best to bake a banana bread ?")

By default, ``predict`` returns only one label : the one with the
highest probability. You can also predict more than one label by
specifying the parameter ``k``:

.. code:: py

    model.predict("Which baking dish is best to bake a banana bread ?", k=3)

If you want to predict more than one sentence you can pass an array of
strings :

.. code:: py

    model.predict(["Which baking dish is best to bake a banana bread ?", "Why not put knives in the dishwasher?"], k=3)

Of course, you can also save and load a model to/from a file as `in the
word representation usage <#saving-and-loading-a-model-object>`__.

For more information about text classification usage of fasttext, you
can refer to our `text classification
tutorial <https://fasttext.cc/docs/en/supervised-tutorial.html>`__.

Compress model files with quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you want to save a supervised model file, fastText can compress it
in order to have a much smaller model file by sacrificing only a little
bit performance.

.. code:: py

    # with the previously trained `model` object, call :
    model.quantize(input='data.train.txt', retrain=True)

    # then display results and save the new model :
    print_results(*model.test(valid_data))
    model.save_model("model_filename.ftz")

``model_filename.ftz`` will have a much smaller size than
``model_filename.bin``.

For further reading on quantization, you can refer to `this paragraph
from our blog
post <https://fasttext.cc/blog/2017/10/02/blog-post.html#model-compression>`__.

IMPORTANT: Preprocessing data / encoding conventions
----------------------------------------------------

In general it is important to properly preprocess your data. In
particular our example scripts in the `root
folder <https://github.com/facebookresearch/fastText>`__ do this.

fastText assumes UTF-8 encoded text. All text must be `unicode for
Python2 <https://docs.python.org/2/library/functions.html#unicode>`__
and `str for
Python3 <https://docs.python.org/3.5/library/stdtypes.html#textseq>`__.
The passed text will be `encoded as UTF-8 by
pybind11 <https://pybind11.readthedocs.io/en/master/advanced/cast/strings.html?highlight=utf-8#strings-bytes-and-unicode-conversions>`__
before passed to the fastText C++ library. This means it is important to
use UTF-8 encoded text when building a model. On Unix-like systems you
can convert text using `iconv <https://en.wikipedia.org/wiki/Iconv>`__.

fastText will tokenize (split text into pieces) based on the following
ASCII characters (bytes). In particular, it is not aware of UTF-8
whitespace. We advice the user to convert UTF-8 whitespace / word
boundaries into one of the following symbols as appropiate.

-  space
-  tab
-  vertical tab
-  carriage return
-  formfeed
-  the null character

The newline character is used to delimit lines of text. In particular,
the EOS token is appended to a line of text if a newline character is
encountered. The only exception is if the number of tokens exceeds the
MAX\_LINE\_SIZE constant as defined in the `Dictionary
header <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h>`__.
This means if you have text that is not separate by newlines, such as
the `fil9 dataset <http://mattmahoney.net/dc/textdata>`__, it will be
broken into chunks with MAX\_LINE\_SIZE of tokens and the EOS token is
not appended.

The length of a token is the number of UTF-8 characters by considering
the `leading two bits of a
byte <https://en.wikipedia.org/wiki/UTF-8#Description>`__ to identify
`subsequent bytes of a multi-byte
sequence <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc>`__.
Knowing this is especially important when choosing the minimum and
maximum length of subwords. Further, the EOS token (as specified in the
`Dictionary
header <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h>`__)
is considered a character and will not be broken into subwords.

More examples
-------------

In order to have a better knowledge of fastText models, please consider
the main
`README <https://github.com/facebookresearch/fastText/blob/master/README.md>`__
and in particular `the tutorials on our
website <https://fasttext.cc/docs/en/supervised-tutorial.html>`__.

You can find further python examples in `the doc
folder <https://github.com/facebookresearch/fastText/tree/master/python/doc/examples>`__.

As with any package you can get help on any Python function using the
help function.

For example

::

    +>>> import fasttext
    +>>> help(fasttext.FastText)

    Help on module fasttext.FastText in fasttext:

    NAME
        fasttext.FastText

    DESCRIPTION
        # Copyright (c) 2017-present, Facebook, Inc.
        # All rights reserved.
        #
        # This source code is licensed under the MIT license found in the
        # LICENSE file in the root directory of this source tree.

    FUNCTIONS
        load_model(path)
            Load a model given a filepath and return a model object.

        tokenize(text)
            Given a string of text, tokenize it and return a list of tokens
    [...]

API
===

``train_unsupervised`` parameters
---------------------------------

.. code:: python

        input             # training file path (required)
        model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
        lr                # learning rate [0.05]
        dim               # size of word vectors [100]
        ws                # size of the context window [5]
        epoch             # number of epochs [5]
        minCount          # minimal number of word occurences [5]
        minn              # min length of char ngram [3]
        maxn              # max length of char ngram [6]
        neg               # number of negatives sampled [5]
        wordNgrams        # max length of word ngram [1]
        loss              # loss function {ns, hs, softmax, ova} [ns]
        bucket            # number of buckets [2000000]
        thread            # number of threads [number of cpus]
        lrUpdateRate      # change the rate of updates for the learning rate [100]
        t                 # sampling threshold [0.0001]
        verbose           # verbose [2]

``train_supervised`` parameters
-------------------------------

.. code:: python

        input             # training file path (required)
        lr                # learning rate [0.1]
        dim               # size of word vectors [100]
        ws                # size of the context window [5]
        epoch             # number of epochs [5]
        minCount          # minimal number of word occurences [1]
        minCountLabel     # minimal number of label occurences [1]
        minn              # min length of char ngram [0]
        maxn              # max length of char ngram [0]
        neg               # number of negatives sampled [5]
        wordNgrams        # max length of word ngram [1]
        loss              # loss function {ns, hs, softmax, ova} [softmax]
        bucket            # number of buckets [2000000]
        thread            # number of threads [number of cpus]
        lrUpdateRate      # change the rate of updates for the learning rate [100]
        t                 # sampling threshold [0.0001]
        label             # label prefix ['__label__']
        verbose           # verbose [2]
        pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []

``model`` object
----------------

``train_supervised``, ``train_unsupervised`` and ``load_model``
functions return an instance of ``_FastText`` class, that we generaly
name ``model`` object.

This object exposes those training arguments as properties : ``lr``,
``dim``, ``ws``, ``epoch``, ``minCount``, ``minCountLabel``, ``minn``,
``maxn``, ``neg``, ``wordNgrams``, ``loss``, ``bucket``, ``thread``,
``lrUpdateRate``, ``t``, ``label``, ``verbose``, ``pretrainedVectors``.
So ``model.wordNgrams`` will give you the max length of word ngram used
for training this model.

In addition, the object exposes several functions :

.. code:: python

        get_dimension           # Get the dimension (size) of a lookup vector (hidden layer).
                                # This is equivalent to `dim` property.
        get_input_vector        # Given an index, get the corresponding vector of the Input Matrix.
        get_input_matrix        # Get a copy of the full input matrix of a Model.
        get_labels              # Get the entire list of labels of the dictionary
                                # This is equivalent to `labels` property.
        get_line                # Split a line of text into words and labels.
        get_output_matrix       # Get a copy of the full output matrix of a Model.
        get_sentence_vector     # Given a string, get a single vector represenation. This function
                                # assumes to be given a single line of text. We split words on
                                # whitespace (space, newline, tab, vertical tab) and the control
                                # characters carriage return, formfeed and the null character.
        get_subword_id          # Given a subword, return the index (within input matrix) it hashes to.
        get_subwords            # Given a word, get the subwords and their indicies.
        get_word_id             # Given a word, get the word id within the dictionary.
        get_word_vector         # Get the vector representation of word.
        get_words               # Get the entire list of words of the dictionary
                                # This is equivalent to `words` property.
        is_quantized            # whether the model has been quantized
        predict                 # Given a string, get a list of labels and a list of corresponding probabilities.
        quantize                # Quantize the model reducing the size of the model and it's memory footprint.
        save_model              # Save the model to the given path
        test                    # Evaluate supervised model using file given by path
        test_label              # Return the precision and recall score for each label.

The properties ``words``, ``labels`` return the words and labels from
the dictionary :

.. code:: py

    model.words         # equivalent to model.get_words()
    model.labels        # equivalent to model.get_labels()

The object overrides ``__getitem__`` and ``__contains__`` functions in
order to return the representation of a word and to check if a word is
in the vocabulary.

.. code:: py

    model['king']       # equivalent to model.get_word_vector('king')
    'king' in model     # equivalent to `'king' in model.get_words()`

Join the fastText community
---------------------------

-  `Facebook page <https://www.facebook.com/groups/1174547215919768>`__
-  `Stack
   overflow <https://stackoverflow.com/questions/tagged/fasttext>`__
-  `Google
   group <https://groups.google.com/forum/#!forum/fasttext-library>`__
-  `GitHub <https://github.com/facebookresearch/fastText>`__

.. |CircleCI| image:: https://circleci.com/gh/facebookresearch/fastText/tree/master.svg?style=svg
   :target: https://circleci.com/gh/facebookresearch/fastText/tree/master
