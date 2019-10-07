---
id: supervised-tutorial
title: Text classification
---

Text classification is a core problem to many applications, like spam detection, sentiment analysis or smart replies. In this tutorial, we describe how to build a text classifier with the fastText tool.

## What is text classification?

The goal of text classification is to assign documents (such as emails,  posts, text messages, product reviews, etc...) to one or multiple categories. Such categories can be review scores, spam v.s. non-spam, or the language in which the document was typed. Nowadays, the dominant approach to build such classifiers is  machine learning, that is  learning classification rules from examples. In order to build such classifiers, we need labeled data, which consists of documents and their corresponding categories (or tags, or labels).

As an example, we build a classifier which automatically classifies stackexchange questions about cooking into one of  several possible tags, such as `pot`, `bowl` or `baking`.

##  Installing fastText

The first step of this tutorial is to install and build fastText. It only requires a c++ compiler with good support of c++11.

Let us start by downloading the [most recent release](https://github.com/facebookresearch/fastText/releases):

```bash
$ wget https://github.com/facebookresearch/fastText/archive/v0.9.1.zip
$ unzip v0.9.1.zip
```

Move to the fastText directory and build it:

```bash
$ cd fastText-0.9.1
# for command line tool :
$ make
# for python bindings :
$ pip install .
```

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
<br />
Running the binary without any argument will print the high level documentation, showing the different use cases supported by fastText:

```bash
>> ./fasttext
usage: fasttext <command> <args>

The commands supported by fasttext are:

  supervised              train a supervised classifier
  quantize                quantize a model to reduce the memory usage
  test                    evaluate a supervised classifier
  predict                 predict most likely labels
  predict-prob            predict most likely labels with probabilities
  skipgram                train a skipgram model
  cbow                    train a cbow model
  print-word-vectors      print word vectors given a trained model
  print-sentence-vectors  print sentence vectors given a trained model
  nn                      query for nearest neighbors
  analogies               query for analogies

```

In this tutorial, we mainly use the `supervised`, `test` and `predict` subcommands, which corresponds to learning (and using) text classifier. For an introduction to the other functionalities of fastText, please see the [tutorial about learning word vectors](https://fasttext.cc/docs/en/unsupervised-tutorial.html).

<!--Python-->
<br />
Calling the help function will show high level documentation of the library:
```py
>>> import fasttext
>>> help(fasttext.FastText)
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
    
    read_args(arg_list, arg_dict, arg_names, default_values)
    
    tokenize(text)
        Given a string of text, tokenize it and return a list of tokens
    
    train_supervised(*kargs, **kwargs)
        Train a supervised model and return a model object.
        
        input must be a filepath. The input text does not need to be tokenized
        as per the tokenize function, but it must be preprocessed and encoded
        as UTF-8. You might want to consult standard preprocessing scripts such
        as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html
        
        The input file must must contain at least one label per line. For an
        example consult the example datasets which are part of the fastText
        repository such as the dataset pulled by classification-example.sh.
    
    train_unsupervised(*kargs, **kwargs)
        Train an unsupervised model and return a model object.
        
        input must be a filepath. The input text does not need to be tokenized
        as per the tokenize function, but it must be preprocessed and encoded
        as UTF-8. You might want to consult standard preprocessing scripts such
        as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html
        
        The input field must not contain any labels or use the specified label prefix
        unless it is ok for those words to be ignored. For an example consult the
        dataset pulled by the example script word-vector-example.sh, which is
        part of the fastText repository.
```

In this tutorial, we mainly use the `train_supervised`, which returns a model object, and call `test` and `predict` on this object. That corresponds to learning (and using) text classifier. For an introduction to the other functionalities of fastText, please see the [tutorial about learning word vectors](https://fasttext.cc/docs/en/unsupervised-tutorial.html).
<!--END_DOCUSAURUS_CODE_TABS-->

## Getting and preparing the data

As mentioned in the introduction, we need labeled data to train our supervised classifier. In this tutorial, we are interested in building a classifier to automatically recognize the topic of a stackexchange question about cooking. Let's download examples of questions from [the cooking section of Stackexchange](http://cooking.stackexchange.com/), and their associated tags:

```bash
>> wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz
>> head cooking.stackexchange.txt
```

Each line of the text file contains a list of labels, followed by the corresponding document. All the labels start by the `__label__` prefix, which is how fastText recognize what is a label or what is a word. The model is then trained to predict the labels given the word in the document.

Before training our first classifier, we need to split the data into train and validation. We will use the validation set to evaluate how good the learned classifier is on new data.

```bash
>> wc cooking.stackexchange.txt
   15404  169582 1401900 cooking.stackexchange.txt
```

Our full dataset contains 15404 examples. Let's split it into a training set of 12404 examples and a validation set of 3000 examples:

```bash
>> head -n 12404 cooking.stackexchange.txt > cooking.train
>> tail -n 3000 cooking.stackexchange.txt > cooking.valid
```

## Our first classifier

We are now ready to train our first classifier:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking
Read 0M words
Number of words:  14598
Number of labels: 734
Progress: 100.0%  words/sec/thread: 75109  lr: 0.000000  loss: 5.708354  eta: 0h0m
```

The `-input` command line option indicates the file containing the training examples, while the `-output` option indicates where to save the model. At the end of training, a file `model_cooking.bin`, containing the trained classifier, is created in the current directory.

<!--Python-->
```py
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train")
Read 0M words
Number of words:  14598
Number of labels: 734
Progress: 100.0%  words/sec/thread: 75109  lr: 0.000000  loss: 5.708354  eta: 0h0m
```
The `input` argument indicates the file containing the training examples. We can now use the `model` variable to access information on the trained model.

We can also call `save_model` to save it as a file and load it later with `load_model` function.
```py
>>> model.save_model("model_cooking.bin")
```
<!--END_DOCUSAURUS_CODE_TABS-->


Now, we can test our classifier, by :
<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext predict model_cooking.bin -
```

and then typing a sentence.  Let's first try the sentence:

*Which baking dish is best to bake a banana bread ?*

The predicted tag is `baking`  which fits well to this question. Let us now try a second example:

*Why not put knives in the dishwasher?*

<!--Python-->
```py
>>> model.predict("Which baking dish is best to bake a banana bread ?")
((u'__label__baking',), array([0.15613931]))
```
The predicted tag is `baking`  which fits well to this question. Let us now try a second example:

```py
>>> model.predict("Why not put knives in the dishwasher?")
((u'__label__food-safety',), array([0.08686075]))
```

<!--END_DOCUSAURUS_CODE_TABS-->


The label predicted by the model is `food-safety`, which is not relevant. Somehow, the model seems to fail on simple examples.

To get a better sense of its quality, let's test it on the validation data by running:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext test model_cooking.bin cooking.valid
N  3000
P@1  0.124
R@1  0.0541
Number of examples: 3000
```
The output of fastText are the precision at one (`P@1`) and the recall at one (`R@1`).

<!--Python-->
```py
>>> model.test("cooking.valid")
(3000L, 0.124, 0.0541)
```
The output are the number of samples (here `3000`), the precision at one (`0.124`) and the recall at one (`0.0541`).
<!--END_DOCUSAURUS_CODE_TABS-->

We can also compute the precision at five and recall at five with:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext test model_cooking.bin cooking.valid 5
N  3000
P@5  0.0668
R@5  0.146
Number of examples: 3000
```
<!--Python-->
```py
>>> model.test("cooking.valid", k=5)
(3000L, 0.0668, 0.146)
```
<!--END_DOCUSAURUS_CODE_TABS-->


## Advanced readers: precision and recall

The precision is the number of correct labels among the labels predicted by fastText. The recall is the number of labels that successfully were predicted, among all the real labels. Let's take an example to make this more clear:

*Why not put knives in the dishwasher?*

On Stack Exchange, this sentence is labeled with three tags: `equipment`, `cleaning` and `knives`. The top five labels predicted by the model can be obtained with:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext predict model_cooking.bin - 5
```
<!--Python-->
```py
>>> model.predict("Why not put knives in the dishwasher?", k=5)
((u'__label__food-safety', u'__label__baking', u'__label__equipment', u'__label__substitutions', u'__label__bread'), array([0.0857 , 0.0657, 0.0454, 0.0333, 0.0333]))
```
<!--END_DOCUSAURUS_CODE_TABS-->

are `food-safety`, `baking`, `equipment`, `substitutions` and `bread`.

Thus, one out of five labels predicted by the model is correct, giving a precision of 0.20. Out of the three real labels, only one is predicted by the model, giving a recall of 0.33.

For more details, see [the related Wikipedia page](https://en.wikipedia.org/wiki/Precision_and_recall).

## Making the model better

The model obtained by running fastText with the default arguments is pretty bad at classifying new questions. Let's try to improve the performance, by changing the default parameters.

### preprocessing the data

Looking at the data, we observe that some words contain uppercase letter or punctuation. One of the first step to improve the performance of our model is to apply some simple pre-processing. A crude normalization can be obtained using command line tools such as `sed` and `tr`:

```bash
>> cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
>> head -n 12404 cooking.preprocessed.txt > cooking.train
>> tail -n 3000 cooking.preprocessed.txt > cooking.valid
```

Let's train a new model on the pre-processed data:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 82041  lr: 0.000000  loss: 5.671649  eta: 0h0m

>> ./fasttext test model_cooking.bin cooking.valid
N  3000
P@1  0.164
R@1  0.0717
Number of examples: 3000
```
<!--Python-->
```py
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train")
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 82041  lr: 0.000000  loss: 5.671649  eta: 0h0m

>>> model.test("cooking.valid")
(3000L, 0.164, 0.0717)
```
<!--END_DOCUSAURUS_CODE_TABS-->

We observe that thanks to the pre-processing, the vocabulary is smaller (from 14k words to 9k). The precision is also starting to go up by 4%!

### more epochs and larger learning rate

By default, fastText sees each training example only five times during training, which is pretty small, given that our training set only have 12k training examples. The number of times each examples is seen (also known as the number of epochs), can be increased using the `-epoch` option:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking -epoch 25
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 77633  lr: 0.000000  loss: 7.147976  eta: 0h0m
```
<!--Python-->
```py
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train", epoch=25)
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 77633  lr: 0.000000  loss: 7.147976  eta: 0h0m
```
<!--END_DOCUSAURUS_CODE_TABS-->

Let's test the new model:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext test model_cooking.bin cooking.valid
N  3000
P@1  0.501
R@1  0.218
Number of examples: 3000
```
<!--Python-->
```py
>>> model.test("cooking.valid")
(3000L, 0.501, 0.218)
```
<!--END_DOCUSAURUS_CODE_TABS-->

This is much better! Another way to change the learning speed of our model is to increase (or decrease) the learning rate of the algorithm. This corresponds to how much the model changes after processing each example. A learning rate of 0 would mean that the model does not change at all, and thus, does not learn anything. Good values of the learning rate are in the range `0.1 - 1.0`.

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0  
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 81469  lr: 0.000000  loss: 6.405640  eta: 0h0m

>> ./fasttext test model_cooking.bin cooking.valid
N  3000
P@1  0.563
R@1  0.245
Number of examples: 3000
```
<!--Python-->
```py
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0)
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 81469  lr: 0.000000  loss: 6.405640  eta: 0h0m

>>> model.test("cooking.valid")
(3000L, 0.563, 0.245)
```
<!--END_DOCUSAURUS_CODE_TABS-->

Even better! Let's try both together:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0 -epoch 25
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 76394  lr: 0.000000  loss: 4.350277  eta: 0h0m

>> ./fasttext test model_cooking.bin cooking.valid
N  3000
P@1  0.585
R@1  0.255
Number of examples: 3000
```
<!--Python-->
```py
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25)
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 76394  lr: 0.000000  loss: 4.350277  eta: 0h0m

>>> model.test("cooking.valid")
(3000L, 0.585, 0.255)
```
<!--END_DOCUSAURUS_CODE_TABS-->

Let us now add a few more features to improve even further our performance!

### word n-grams

Finally, we can improve the performance of a model by using word bigrams, instead of just unigrams. This is especially important for classification problems where word order is important, such as sentiment analysis.

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0 -epoch 25 -wordNgrams 2
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 75366  lr: 0.000000  loss: 3.226064  eta: 0h0m

>> ./fasttext test model_cooking.bin cooking.valid
N  3000
P@1  0.599
R@1  0.261
Number of examples: 3000
```
<!--Python-->
```py
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2)
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 75366  lr: 0.000000  loss: 3.226064  eta: 0h0m

>>> model.test("cooking.valid")
(3000L, 0.599, 0.261)
```
<!--END_DOCUSAURUS_CODE_TABS-->

With a few steps, we were able to go from a precision at one of 12.4% to 59.9%. Important steps included:

* preprocessing the data ;
* changing the number of epochs (using the option `-epoch`, standard range `[5 - 50]`) ;
* changing the learning rate (using the option `-lr`, standard range `[0.1 - 1.0]`) ;
* using word n-grams (using the option `-wordNgrams`, standard range `[1 - 5]`).

## Advanced readers: What is a Bigram?

A 'unigram' refers to a single undividing unit, or token,  usually used as an input to a model. For example a unigram can be a word or a letter depending on the model. In fastText, we work at the word level and thus unigrams are words.

Similarly we denote by 'bigram' the concatenation of  2 consecutive tokens or words. Similarly we often talk about n-gram to refer to the concatenation any n consecutive tokens.

For example, in the sentence, 'Last donut of the night', the unigrams are  'last', 'donut', 'of', 'the' and 'night'. The bigrams are: 'Last donut', 'donut of', 'of the' and 'the night'.

Bigrams are particularly interesting because, for most sentences, you can reconstruct the order of the words just by looking at a bag of n-grams.

Let us illustrate this by a simple exercise, given the following bigrams, try to reconstruct the original sentence: 'all out',  'I am', 'of bubblegum', 'out of' and 'am all'.
It is common to refer to a word as a unigram.

## Scaling things up

Since we are training our model on a few thousands of examples, the training only takes a few seconds. But training models on larger datasets, with more labels can start to be too slow. A potential solution to make the training faster is to use the [hierarchical softmax](#advanced-readers-hierarchical-softmax), instead of the regular softmax. This can be done with the option `-loss hs`:

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0 -epoch 25 -wordNgrams 2 -bucket 200000 -dim 50 -loss hs
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 2199406  lr: 0.000000  loss: 1.718807  eta: 0h0m
```
<!--Python-->
```py
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 2199406  lr: 0.000000  loss: 1.718807  eta: 0h0m
```
<!--END_DOCUSAURUS_CODE_TABS-->

Training should now take less than a second.


## Advanced readers: hierarchical softmax

The hierarchical softmax is a loss function that approximates the softmax with a much faster computation.

The idea is to build a binary tree whose leaves correspond to the labels. Each intermediate node has a binary decision activation (e.g. sigmoid) that is trained, and predicts if we should go to the left or to the right. The probability of the output unit is then given by the product of the probabilities of intermediate nodes along the path from the root to the output unit leave.

For a detailed explanation, you can have a look on [this video](https://www.youtube.com/watch?v=B95LTf2rVWM).

In fastText, we use a Huffman tree, so that the lookup time is faster for more frequent outputs and thus the average lookup time for the output is optimal.

## Multi-label classification

When we want to assign a document to multiple labels, we can still use the softmax loss and play with the parameters for prediction, namely the number of labels to predict and the threshold for the predicted probability. However playing with these arguments can be tricky and unintuitive since the probabilities must sum to 1.

A convenient way to handle multiple labels is to use independent binary classifiers for each label. This can be done with `-loss one-vs-all` or `-loss ova`.

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext supervised -input cooking.train -output model_cooking -lr 0.5 -epoch 25 -wordNgrams 2 -bucket 200000 -dim 50 -loss one-vs-all
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   72104 lr:  0.000000 loss:  4.340807 ETA:   0h 0m
```
<!--Python-->
```py
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train", lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   72104 lr:  0.000000 loss:  4.340807 ETA:   0h 0m
```
<!--END_DOCUSAURUS_CODE_TABS-->

It is a good idea to decrease the learning rate compared to other loss functions.

Now let's have a look on our predictions, we want as many prediction as possible (argument `-1`) and we want only labels with probability higher or equal to `0.5` :
<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
>> ./fasttext predict-prob model_cooking.bin - -1 0.5
```
and then type the sentence:

*Which baking dish is best to bake a banana bread ?*

we get:
```
__label__baking 1.00000 __label__bananas 0.939923 __label__bread 0.592677
```
<!--Python-->
```py
>>> model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
((u''__label__baking, u'__label__bananas', u'__label__bread'), array([1.00000, 0.939923, 0.592677]))
```
<!--END_DOCUSAURUS_CODE_TABS-->



<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
<br />
We can also evaluate our results with the `test` command :
```bash
>> ./fasttext test model_cooking.bin cooking.valid -1 0.5
N 3000
P@-1  0.702
R@-1  0.2
Number of examples: 3000
```
and play with the threshold to obtain desired precision/recall metrics :

```bash
>> ./fasttext test model_cooking.bin cooking.valid -1 0.1
N 3000
P@-1  0.591
R@-1  0.272
Number of examples: 3000
```
<!--Python-->
<br />
We can also evaluate our results with the `test` function:
```py
>>> model.test("cooking.valid", k=-1)
(3000L, 0.702, 0.2)
```
<!--END_DOCUSAURUS_CODE_TABS-->


## Conclusion

In this tutorial, we gave a brief overview of how to use fastText to train powerful text classifiers. We had a light overview of some of the most important options to tune.
