---
id: support
title: Get started
---

## What is fastText?

fastText is a library for efficient learning of word representations and sentence classification.

## Requirements

fastText builds on modern Mac OS and Linux distributions.
Since it uses C++11 features, it requires a compiler with good C++11 support.
These include :

* (gcc-4.6.3 or newer) or (clang-3.3 or newer)

Compilation is carried out using a Makefile, so you will need to have a working **make**.
For the word-similarity evaluation script you will need:

* python 2.6 or newer
* numpy & scipy

## Building fastText

In order to build `fastText`, use the following:

```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```

This will produce object files for all the classes as well as the main binary `fasttext`.
If you do not plan on using the default system-wide compiler, update the two macros defined at the beginning of the Makefile (CC and INCLUDES).

