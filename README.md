# fastText

fastText is a library for efficient computation of word representations and sentence classification.

## Requirements

fastText should compile on all modern platforms including Mac OS and Linux. Because of the use of C++ 11 features, it requires the use of a C++ 11 compatible compiler. These include :

* (gcc-4.6.3 or newer) or (clang-3.3 or newer)
* make

For the word-similarity evaluation script you will need:

* python 2.6 or newer

## Building fastText

Use the provided Makefile. At the command prompt, type:

```
$ make
```

This will produce object files for all the classes as well as the main binary `fasttext`.
If you do not plan on using the default system-wide compiler, please update the two macros defined at the beginning of the Makefile (CC and INCLUDES).

## Example use cases

We provide in this library two main use cases that we will describe here and that correspond to [1] and [2].

### text classification

In order to train a text classifier following [2], please follow the compilation steps and then issue:

```
$ ./fasttext supervised -input train.txt -output model
```

### predicting labels

where `train.txt` is a text file containing a training sentence per line along with the labels. By default, we assume that labels are words in a sentence that are prefixed by `__label__`. This will output two files: `model.bin` and `model.vec`. Once the model was trained, you can compute the test precision at 1 (P@1) using:

```
$ ./fasttext test model.bin test.txt
```

If you want to obtain the most likely label for a piece of text, please use:

```
$ ./fasttext predict model.bin test.txt
```

where test.txt contains a piece of text to classify per line. Doing so will output to standard output the most likely label per line. Please check `classification.sh` for an example use case. In order to reproduce results from the paper [2] please run `classification-results.sh`, this will download all the datasets and reproduce the results from Table 1.

### Word Representation

In order to compute word vectors as described in [1], please compile all the executables as described before. Then, given a training file `data.txt`, do:

```
$ ./fasttext skipgram -input data.txt -output model
```

This will launch the optimization and save two files: `model.bin` and `model.vec`.
`model.vec` is a text file containing the word vectors, one per line. `model.bin` is the binary containing all the parameters of the model. It can be used later to compute word vectors or to restart the optimization.

### obtaining word vectors For out-of-vocabulary words

Provided you have a text file `queries.txt` containing words for which you want to compute vectors, please issue the following command

```
$ ./fasttext print-vectors model.bin < queries.txt
```

This will output to standard output, the word and its vector, one per line.
Please note that this can be successfully used with pipes:

```
$ cat queries.txt | ./fasttext print-vectors model.bin
```

See the provided scripts for an example. For instance, running:

```
$ ./get-vectors.sh
```

will compile the code, download data, compute the word vectors and evaluate on the rare words similarity dataset RW [Thang et al. 2013].

## Full documentation

* input: text file used for training the model
* test: text file used for testing  the model. Only works in the classification setup
* output: prefix of the file saved at the end of optimization
* lr: learning rate (default of 0.05)
* dim: required dimension of the vectors
* ws: size of the context window considered around the word
* epoch: number of iterations over the training file
* minCount: minimal word occurence in the training file
* neg: number of negatives for the negative sampling approximation
* wordNgrams: number of word n-grams considered in the sentence classification setup
* sampling: word distribution used to sample negatives
    * log: use logarithm of unigram frequency
    * sqrt: use square root of unigram frequency
    * uni: uniform distribution over words
* loss: approximation to the softmax loss
    * hs: hierarchical softmax
    * ns: negative sampling
    * softmax: full softmax computation (slow)
* model: model used
    * cbow: continuous bag-of-words
    * sg: skip-gram
    * supervised: sentence classification
* bucket: number of n-gram hashes in use
* minn: shortest character n-gram used
* maxn: longest character n-gram used
* onlyWord: the `onlyWord` most frequent words don't use subword information
* thread: number of threads used for optimization
* verbose: print info to stdout every other `verbose` samples
* t: threshold for random word discarding based on unigram frequency
* label: what string to use as prefix for labels

## References

[1] Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov, Enriching Word Vectors with Subword Information, arXiv 1607.04606, 2016
[2] Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov, Bag of Tricks for Efficient Text Classification, arXiv 1607.01759, 2016

## Join the fastText community

* Facebook page: https://www.facebook.com/groups/fasttextusers
* Contact: [egrave@fb.com](mailto:egrave@fb.com) [bojanowski@fb.com](mailto:bojanowski@fb.com) [ajoulin@fb.com](mailto:ajoulin@fb.com) [tmikolov@fb.com](mailto:tmikolov@fb.com)

See the CONTRIBUTING file for information about how to help out.

## License

fastText is BSD-licensed. We also provide an additional patent grant.
