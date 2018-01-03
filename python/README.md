# fastText

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

## Requirements

[fastText](https://fasttext.cc/) builds on modern Mac OS and Linux distributions.
Since it uses C\++11 features, it requires a compiler with good C++11 support.
These include :

* (gcc-4.8 or newer) or (clang-3.3 or newer)

You will need

* [Python](https://www.python.org/) version 2.7 or >=3.4
* [NumPy](http://www.numpy.org/) & [SciPy](https://www.scipy.org/)
* [pybind11](https://github.com/pybind/pybind11)

## Building fastText

The easiest way to install fastText is to use [pip](https://pip.pypa.io/en/stable/).

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```

Alternatively you can also install fastText using setuptools.

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ python setup.py install
```

Now you can import this library with

```
import fastText
```

## Examples

In general it is assumed that the reader already has good knowledge of fastText. For this consider the main [README](https://github.com/facebookresearch/fastText/blob/master/README.md) and in particular [the tutorials on our website](https://fasttext.cc/docs/en/supervised-tutorial.html).

We recommend you look at the [examples within the doc folder](https://github.com/facebookresearch/fastText/tree/master/python/doc/examples).

As with any package you can get help on any Python function using the help function.

For example

```
+>>> import fastText
+>>> help(fastText.FastText)

Help on module fastText.FastText in fastText:

NAME
    fastText.FastText

DESCRIPTION
    # Copyright (c) 2017-present, Facebook, Inc.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in the root directory of this source tree. An additional grant
    # of patent rights can be found in the PATENTS file in the same directory.

FUNCTIONS
    load_model(path)
        Load a model given a filepath and return a model object.

    tokenize(text)
        Given a string of text, tokenize it and return a list of tokens
[...]
```

## IMPORTANT: Preprocessing data / enconding conventions

In general it is important to properly preprocess your data. In particular our example scripts in the [root folder](https://github.com/facebookresearch/fastText) do this. 

fastText assumes UTF-8 encoded text. All text must be [unicode for Python2](https://docs.python.org/2/library/functions.html#unicode) and [str for Python3](https://docs.python.org/3.5/library/stdtypes.html#textseq). The passed text will be [encoded as UTF-8 by pybind11](https://pybind11.readthedocs.io/en/master/advanced/cast/strings.html?highlight=utf-8#strings-bytes-and-unicode-conversions) before passed to the fastText C++ library. This means it is important to use UTF-8 encoded text when building a model. On Unix-like systems you can convert text using [iconv](https://en.wikipedia.org/wiki/Iconv).

fastText will tokenize (split text into pieces) based on the following ASCII characters (bytes). In particular, it is not aware of UTF-8 whitespace. We advice the user to convert UTF-8 whitespace / word boundaries into one of the following symbols as appropiate.

* space
* tab
* vertical tab
* carriage return
* formfeed
* the null character

The newline character is used to delimit lines of text. In particular, the EOS token is appended to a line of text if a newline character is encountered. The only exception is if the number of tokens exceeds the MAX\_LINE\_SIZE constant as defined in the [Dictionary header](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h). This means if you have text that is not separate by newlines, such as the [fil9 dataset](http://mattmahoney.net/dc/textdata), it will be broken into chunks with MAX\_LINE\_SIZE of tokens and the EOS token is not appended.

The length of a token is the number of UTF-8 characters by considering the [leading two bits of a byte](https://en.wikipedia.org/wiki/UTF-8#Description) to identify [subsequent bytes of a multi-byte sequence](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc). Knowing this is especially important when choosing the minimum and maximum length of subwords. Further, the EOS token (as specified in the [Dictionary header](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h)) is considered a character and will not be broken into subwords.
