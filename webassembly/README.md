 fastText [![CircleCI](https://circleci.com/gh/facebookresearch/fastText/tree/master.svg?style=svg)](https://circleci.com/gh/facebookresearch/fastText/tree/master)

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

In this document we present how to use fastText in a browser with WebAssembly.


# Requirements

[fastText](https://fasttext.cc/) builds on modern Mac OS and Linux distributions.
Since it uses C\++11 features, it requires a compiler with good C++11 support.
You will need [emscripten](https://emscripten.org/) and a [browser that supports WebAssembly](https://caniuse.com/#feat=wasm).


# Building WebAssembly binaries

First, download and install emscripten sdk as [described here](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions)


Make sure you activated the PATH for emscripten:
```bash
$ source /path/to/emsdk/emsdk_env.sh
```

Clone our [repository](https://github.com/facebookresearch/fastText/).

```bash
$ git clone git@github.com:facebookresearch/fastText.git
```

Build WebAssembly binaries:
```bash
$ cd fastText
$ make wasm
```


