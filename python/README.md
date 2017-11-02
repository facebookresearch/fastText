# fastText

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

## Requirements

**fastText** builds on modern Mac OS and Linux distributions.
Since it uses C\++11 features, it requires a compiler with good C++11 support.
These include :

* (gcc-4.8 or newer) or (clang-3.3 or newer)

You will need

* python 2.7 or newer
* numpy & scipy
* [pybind11](https://github.com/pybind/pybind11)

## Building fastTextpy

In order to build `fastTextpy`, do the following:

```
$ python setup.py install
```

This will add the module fastTextpy to your python interpreter.
Depending on your system you might need to use 'sudo', for example

```
$ sudo python setup.py install
```

Now you can import this library with

```
import fastText
```


## Examples

If you're already largely familiar with fastText you could skip this section 
and take a look at the examples within the doc folder.

## Using models

First, you'll need to train a model with fastText. For example

```
./fasttext skipgram -input data/fil9 -output result/fil9
```

You can see more examples within the scripts in the [fastText repository](https://github.com/facebookresearch/fastText).

Next, you can load this model from Python and query it. 

```
from fastText import load_model

f = load_model('result/model.bin')
words, frequency = f.get_words()
subwords = f.get_subwords("Paris")
```

If you trained an unsupervised model, you can get word vectors with

```
vector = f.get_word_vector("London")
```

If you trained a supervised model, you can get the top k labels and get their probabilities with

```
k = 5
labels, probabilities = f.predict("I like this Product", k)
```

A more advanced application might look like this:

Getting the word vectors of all words:

```
words, frequency = f.get_words()
for w in words:
    print((w, f.get_word_vector(w))
```

## Training models

Training a model is easy. For example

```
from fastText import train_supervised
from fastText import train_unsupervised

model_unsup = train_unsupervised(
    input=<data>,
    epoch=1,
    model="cbow",
    thread=10
)
model_unsup.save_model(<path>)

model_sup = train_supervised(
    input=<labeled_data>
    epoch=1,
    thread=10
)
```

You can then use the model objects just as exemplified above.

To get extended help on these functions use the python help functions.

For example

```
Help on function train_unsupervised in module fastText.FastText:

train_unsupervised(input, model=u'skipgram', lr=0.05, dim=100, ws=5, epoch=5, minCount=5, minCountLabel=0, minn=3, maxn=6, neg=5, wordNgrams=1, loss=u'ns', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001, label=u'__label__', verbose=2, pretrainedVectors=u'', saveOutput=0)
    Train an unsupervised model and return a model object.

    input must be a filepath. The input text does not need to be tokenized
    as per the tokenize function, but it must be preprocessed and encoded
    as UTF-8. You might want to consult standard preprocessing scripts such
    as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html

    The input fiel must not contain any labels or use the specified label prefix
    unless it is ok for those words to be ignored. For an example consult the
    dataset pulled by the example script word-vector-example.sh, which is
    part of the fastText repository.
```

## Processing data

You can tokenize using the fastText Dictionary method readWord.

This will give you a list of tokens split on the same whitespace characters that fastText splits on.

It will also add the EOS character as necessary, which is exposed via fastText.EOS

Then resulting text is then stored entirely in memory.

For example:

```
from fastText import tokenize
with open(<PATH>, 'r') as f:
    tokens = tokenize(f.read())
```
