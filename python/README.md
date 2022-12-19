# fastText

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

In this document we present how to use fastText in python.

## Table of contents

* [Requirements](#requirements)
* [Installation](#installation)
* [Usage overview](#usage-overview)
* [API](#api)
   * [`model` object](#model-object)


# Requirements

[fastText-predict](https://fasttext.cc/) builds on modern Mac OS and Linux distributions.
Since it uses C\++11 features, it requires a compiler with good C++11 support. You will need [Python](https://www.python.org/) (version 2.7 or ≥ 3.4) and [pybind11](https://github.com/pybind/pybind11).


# Installation

To install the latest release, you can do :
```bash
$ pip install fasttext-predict
```

or, to get the latest development version of fasttext, you can install from our github repository :
```bash
$ git clone https://github.com/dalf/fasttext-predic.git
$ cd fastText
$ sudo pip install .
$ # or :
$ sudo python setup.py install
```

# Usage overview

You can load a model:
```py
import fasttext
model = fasttext.load_model("model_filename.bin")
```


Then, We can also predict labels for a specific text :

```py
model.predict("Which baking dish is best to bake a banana bread ?")
```

By default, `predict` returns only one label : the one with the highest probability. You can also predict more than one label by specifying the parameter `k`:
```py
model.predict("Which baking dish is best to bake a banana bread ?", k=3)
```

If you want to predict more than one sentence you can pass an array of strings :

```py
model.predict(["Which baking dish is best to bake a banana bread ?", "Why not put knives in the dishwasher?"], k=3)
```

For more information about text classification usage of fasttext, you can refer to our [text classification tutorial](https://fasttext.cc/docs/en/supervised-tutorial.html).


## IMPORTANT: Preprocessing data / encoding conventions

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

# API

## `model` object

`train_supervised`, `train_unsupervised` and `load_model` functions return an instance of `_FastText` class, that we generaly name `model` object.

This object exposes those training arguments as properties : `lr`, `dim`, `ws`, `epoch`, `minCount`, `minCountLabel`, `minn`, `maxn`, `neg`, `wordNgrams`, `loss`, `bucket`, `thread`, `lrUpdateRate`, `t`, `label`, `verbose`, `pretrainedVectors`. So `model.wordNgrams` will give you the max length of word ngram used for training this model.

In addition, the object exposes several functions :

```python
    predict                 # Given a string, get a list of labels and a list of corresponding probabilities.  
```
