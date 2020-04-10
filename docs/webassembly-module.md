---
id: webassembly-module
title: WebAssembly module
---

In this document we present how to use fastText in javascript with WebAssembly.

## Table of contents

* [Requirements](#requirements)
* [Building WebAssembly binaries](#building-webassembly-binaries)
* [Build a webpage that uses fastText](#build-a-webpage-that-uses-fasttext)
* [Load a model](#load-a-model)
* [Train a model](#train-a-model)
   * [Disclaimer](#disclaimer)
   * [Text classification](#text-classification)
   * [Word representations](#word-representations)
* [Quantized models](#quantized-models)
* [API](#api)
   * [`model` object](#model-object)
   * [`loadModel`](#loadmodel)
   * [`trainSupervised`](#trainsupervised)
   * [`trainUnsupervised`](#trainunsupervised)

# Requirements

For building [fastText](https://fasttext.cc/) with WebAssembly bindings, we will need:
 - a compiler with good C++11 support, since it uses C\++11 features,
 - [emscripten](https://emscripten.org/),
 - a [browser that supports WebAssembly](https://caniuse.com/#feat=wasm).


# Building WebAssembly binaries

First, download and install emscripten sdk as [described here](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions).


We need to make sure we activated the PATH for emscripten:
```bash
$ source /path/to/emsdk/emsdk_env.sh
```

Clone [fastText repository](https://github.com/facebookresearch/fastText/):

```bash
$ git clone git@github.com:facebookresearch/fastText.git
```

Build WebAssembly binaries:
```bash
$ cd fastText
$ make wasm
```

This will create `fasttext_wasm.wasm` and `fasttext_wasm.js` in the `webassembly` folder.

- `fasttext_wasm.wasm` is the binary file that will be loaded in the webassembly's virtual machine.
- `fasttext_wasm.js` is a javascript file built by emscripten, that helps to load `fasttext_wasm.wasm` file in the virtual machine and provides some helper functions.
- `fasttext.js` is the wrapper that provides a nice API for fastText. 

As the user of the library, we will interact with classes and methods defined in `fasttext.js`. We won't deal with `fasttext_wasm.*` files, but they are necessary to run fastText in the javascript's VM.

# Build a webpage that uses fastText

In this section we are going to build a minimal HTML page that loads fastText WebAssembly module.

At the root of the repository, create a folder `webassembly-test`, and copy the files mentioned in the previous section:

```bash
$ mkdir webassembly-test
$ cp webassembly/fasttext_wasm.wasm webassembly-test/
$ cp webassembly/fasttext_wasm.js webassembly-test/
$ cp webassembly/fasttext.js webassembly-test/
```

Inside that folder, create `test.html` file containing:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no">
</head>
<body>
    <script type="module">
        import {FastText, addOnPostRun} from "./fasttext.js";

        addOnPostRun(() => {
            let ft = new FastText();
            console.log(ft);
        });

    </script>
</body>
</html>
```

It is important to add the attribute `type="module"` to the script tag, because we use ES6 style imports. `addOnPostRun` is a function that helps to provide a handler that is called when fastText is successfully loaded in the virtual machine. Once we are called inside that function, we can create an instance of `FastText`, that we will use to access the api.


Let's test it.

Opening `test.html` directly in the browser won't work since we are dynamically loading webassembly resources. The `test.html` file must be served from a webserver. The easiest way to achieve this is to use python's simple http server module:

```bash
$ cd webassembly-test
$ python -m SimpleHTTPServer
```

Then browse `http://localhost:8000/test.html` in your browser. If everything worked as expected, you should see `FastText {f: FastText}` in the javascript console.


# Load a model

In order to load a fastText model that was already trained, we can use `loadModel` function. In the example below we use `lid.176.ftz` that you can download from [here](/docs/en/language-identification.html).

Place the model file you want to load inside the same directory than the HTML file, and inside the script part:
```javascript
import {FastText, addOnPostRun} from "./fasttext.js";

const printVector = function(predictions) {
    for (let i=0; i<predictions.size(); i++){
        let prediction = predictions.get(i);
        console.log(predictions.get(i));
    }
}

addOnPostRun(() => {
    let ft = new FastText();

    const url = "lid.176.ftz";
    ft.loadModel(url).then(model => {
        
        console.log("Model loaded.")

        let text = "Bonjour à tous. Ceci est du français";
        console.log(text);
        printVector(model.predict(text, 5, 0.0));

        text = "Hello, world. This is english";
        console.log(text);
        printVector(model.predict(text, 5, 0.0));

        text = "Merhaba dünya. Bu da türkçe"
        console.log(text);
        printVector(model.predict(text, 5, 0.0));
    });
});
```

`loadModel` function returns a [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) that resolves to a `model` object. We can then use [`model` object](#model-object) to call various methods, such as `predict`.

We define `printVector` function that loops through a representation of `std::vector` in javascript, and displays the items. Here, we use it to display prediction results.

You can also refer to `webassembly/doc/examples/predict.html` in the source code.

# Calling other methods

Once the model is loaded, you can call any method like `model.getDimension()` or `model.getSubwords(word)`. You can refer to [this](#api) section of the document for a complete API. You can also have a look to `webassembly/doc/examples/misc.html` file in the source code for further examples.


# Train a model

### Disclaimer

It is also possible to train a model inside the browser with fastText's WebAssembly API. The training can be slow because at the time of writing, it is not possible to use multithreading in WebAssembly (along with dynamic memory growth). So most of the time, we would train a model with the python or command line tool, eventually quantize it, and load it in the WebAssembly module. However, training a model inside the browser can be useful for creating animations or educational tools.
 
### Text classification

Place the `cooking.train` file (as described [here](/docs/en/supervised-tutorial.html)) inside the same directory:

```javascript
import {FastText, addOnPostRun} from "./fasttext.js";

const trainCallback = (progress, loss, wst, lr, eta) => {
    console.log([progress, loss, wst, lr, eta]);
};

addOnPostRun(() => {
    let ft = new FastText();

    ft.trainSupervised("cooking.train", {
        'lr':1.0,
        'epoch':10,
        'loss':'hs',
        'wordNgrams':2,
        'dim':50,
        'bucket':200000
    }, trainCallback).then(model => {
        console.log('Trained.');
    });
});
```

`trainCallback` function is called by the module to show progress, average training cost, number of words per second (per thread, but there is only one thread), learning rate, estimated remaining time.


### Word representations

Place the `fil9` file (as described [here](/docs/en/unsupervised-tutorial.html)) inside the same directory:

```javascript
import {FastText, addOnPostRun} from "./fasttext.js";

const trainCallback = (progress, loss, wst, lr, eta) => {
    console.log([progress, loss, wst, lr, eta]);
};

addOnPostRun(() => {
    let ft = new FastText();

    ft.trainUnsupervised("fil9", 'skipgram', {
        'lr':0.1,
        'epoch':1,
        'loss':'ns',
        'wordNgrams':2,
        'dim':50,
        'bucket':200000
    }, trainCallback).then(model => {
        console.log('Trained.');
    });
});
```

# Quantized models

Quantization is a technique that reduces the size of your models. You can quantize your model as [described here](/docs/en/faqs.html#how-can-i-reduce-the-size-of-my-fasttext-models).

You can load a quantized model in fastText's WebAssembly module, as we did in ["Load a model" section](#load-a-model).


In the context of web, it is particularly useful to have smaller models since they can be downloaded much faster. You can use our autotune feature as [described here](/docs/en/autotune.html#constrain-model-size) in order to find the best trade-off between accuracy and model size that fits your needs.


# API

## `model` object

`trainSupervised`, `trainUnsupervised` and `loadModel` functions return a Promise that resolves to an instance of `FastTextModel` class, that we generaly name `model` object.

This object exposes several functions:

```javascript
isQuant                  // true if the model is quantized.
getDimension             // the dimension (size) of a lookup vector (hidden layer).
getWordVector(word)      // the vector representation of `word`.
getSentenceVector(text)  // the vector representation of `text`.
getNearestNeighbors(word, k=10)      // nearest `k` neighbors of `word`.
getAnalogies(wordA, wordB, wordC, k) // nearest `k` neighbors of the operation `wordA - wordB + wordC`.
getWordId(word)          // get the word id within the dictionary.
getSubwordId(subword)    // the index (within input matrix) a subword hashes to.
getSubwords(word)        // the subwords and their indicies.
getInputVector(ind)      // given an index, get the corresponding vector of the Input Matrix.
predict(text, k = 1, threshold = 0.0) // Given a string, get a list of labels and a list of corresponding
                                      // probabilities. k controls the number of returned labels.
getInputMatrix()         // get a reference to the full input matrix of a (non-quantized) Model.
getOutputMatrix()        // get a reference to the full output matrix of a (non-quantized) Model.
getWords()               // get the entire list of words of the dictionary including the frequency
                         // of the individual words. This does not include any subwords. For that
                         // please consult the function get_subwords.
getLabels()              // get the entire list of labels of the dictionary including the frequency
getLine(text)            // split a line of text into words and labels.
saveModel()              // saves the model file in WebAssembly's in-memory FS and returns a blob
test(url, k, threshold)  // downloads the test file from the specified url, evaluates the supervised model with it.
```

You can also have a look to `webassembly/doc/examples/misc.html` file in the source code for further examples.

## `loadModel`

You can load a model as follows:

`ft.loadModel(url);`

`loadModel` returns a [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) that resolves to a [`model` object](#model-object).


## `trainSupervised` 

You can train a text classification model with fastText's WebAssembly API as follows:

`ft.trainSupervised(trainFile, args, trainCallback);`

- `trainFile`:  the url of the input file
- `args`: a dictionary with following keys:
```javascript
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
```
- `trainCallback` is the name of the function that will be called during training to provide various information. Set this argument to `null` if you don't need a callback, or provide a function that has the following signature: `function myCallback(progress, loss, wst, lr, eta){ ... }`

`trainSupervised` returns a [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) that resolves to a [`model` object](#model-object).

## `trainUnsupervised` 

You can train a word representation model with fastText's WebAssembly API as follows:

`ft.trainUnsupervised(trainFile, modelname, args, trainCallback);`

- `trainFile`:  the url of the input file
- `modelName`: must be `"cbow"` or `"skipgram"`
- `args`: a dictionary with following keys:
```javascript
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
```
- `trainCallback` is the name of the function that will be called during training to provide various information. Set this argument to `null` if you don't need a callback, or provide a function that has the following signature: `function myCallback(progress, loss, wst, lr, eta){ ... }`

`trainUnsupervised` returns a [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) that resolves to a [`model` object](#model-object).



