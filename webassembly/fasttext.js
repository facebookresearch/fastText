/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import fastTextModularized from './fasttext_wasm.js';
const fastTextModule = fastTextModularized();

let postRunFunc = null;
const addOnPostRun = function(func) {
  postRunFunc = func;
};

fastTextModule.addOnPostRun(() => {
  if (postRunFunc) {
    postRunFunc();
  }
});

const thisModule = this;
const trainFileInWasmFs = 'train.txt';
const testFileInWasmFs = 'test.txt';
const modelFileInWasmFs = 'model.bin';

const getFloat32ArrayFromHeap = (len) => {
  const dataBytes = len * Float32Array.BYTES_PER_ELEMENT;
  const dataPtr = fastTextModule._malloc(dataBytes);
  const dataHeap = new Uint8Array(fastTextModule.HEAPU8.buffer,
    dataPtr,
    dataBytes);
  return {
    'ptr':dataHeap.byteOffset,
    'size':len,
    'buffer':dataHeap.buffer
  };
};

const heapToFloat32 = (r) => new Float32Array(r.buffer, r.ptr, r.size);

class FastText {
  constructor() {
    this.f = new fastTextModule.FastText();
  }

  /**
   * loadModel
   *
   * Loads the model file from the specified url, and returns the
   * corresponding `FastTextModel` object.
   *
   * @param {string}     url
   *     the url of the model file.
   *
   * @return {Promise}   promise object that resolves to a `FastTextModel`
   *
   */
  loadModel(url) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;

    const fastTextNative = this.f;
    return new Promise(function(resolve, reject) {
      fetchFunc(url).then(response => {
        return response.arrayBuffer();
      }).then(bytes => {
        const byteArray = new Uint8Array(bytes);
        const FS = fastTextModule.FS;
        FS.writeFile(modelFileInWasmFs, byteArray);
      }).then(() =>  {
        fastTextNative.loadModel(modelFileInWasmFs);
        resolve(new FastTextModel(fastTextNative));
      }).catch(error => {
        reject(error);
      });
    });
  }

  _train(url, modelName, kwargs = {}, callback = null) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;

    return new Promise(function(resolve, reject) {
      fetchFunc(url).then(response => {
        return response.arrayBuffer();
      }).then(bytes => {
        const byteArray = new Uint8Array(bytes);
        const FS = fastTextModule.FS;
        FS.writeFile(trainFileInWasmFs, byteArray);
      }).then(() =>  {
        const argsList = ['lr', 'lrUpdateRate', 'dim', 'ws', 'epoch',
          'minCount', 'minCountLabel', 'neg', 'wordNgrams', 'loss',
          'model', 'bucket', 'minn', 'maxn', 't', 'label', 'verbose',
          'pretrainedVectors', 'saveOutput', 'seed', 'qout', 'retrain',
          'qnorm', 'cutoff', 'dsub', 'qnorm', 'autotuneValidationFile',
          'autotuneMetric', 'autotunePredictions', 'autotuneDuration',
          'autotuneModelSize'];
        const args = new fastTextModule.Args();
        argsList.forEach(k => {
          if (k in kwargs) {
            args[k] = kwargs[k];
          }
        });
        args.model = fastTextModule.ModelName[modelName];
        args.loss = ('loss' in kwargs) ?
          fastTextModule.LossName[kwargs['loss']] : 'hs';
        args.thread = 1;
        args.input = trainFileInWasmFs;

        fastTextNative.train(args, callback);

        resolve(new FastTextModel(fastTextNative));
      }).catch(error => {
        reject(error);
      });
    });
  }

  /**
   * trainSupervised
   *
   * Downloads the input file from the specified url, trains a supervised
   * model and returns a `FastTextModel` object.
   *
   * @param {string}     url
   *     the url of the input file.
   *     The input file must must contain at least one label per line. For an
   *     example consult the example datasets which are part of the fastText
   *     repository such as the dataset pulled by classification-example.sh.
   *
   * @param {dict}       kwargs
   *     train parameters.
   *     For example {'lr': 0.5, 'epoch': 5}
   *
   * @param {function}   callback
   *     train callback function
   *     `callback` function is called regularly from the train loop:
   *     `callback(progress, loss, wordsPerSec, learningRate, eta)`
   *
   * @return {Promise}   promise object that resolves to a `FastTextModel`
   *
   */
  trainSupervised(url, kwargs = {}, callback) {
    const self = this;
    return new Promise(function(resolve, reject) {
      self._train(url, 'supervised', kwargs, callback).then(model => {
        resolve(model);
      }).catch(error => {
        reject(error);
      });
    });
  }

  /**
   * trainUnsupervised
   *
   * Downloads the input file from the specified url, trains an unsupervised
   * model and returns a `FastTextModel` object.
   *
   * @param {string}     url
   *     the url of the input file.
   *     The input file must not contain any labels or use the specified label
   *     prefixunless it is ok for those words to be ignored. For an example
   *     consult the dataset pulled by the example script word-vector-example.sh
   *     which is part of the fastText repository.
   *
   * @param {string}     modelName
   *     Model to be used for unsupervised learning. `cbow` or `skipgram`.
   *
   * @param {dict}       kwargs
   *     train parameters.
   *     For example {'lr': 0.5, 'epoch': 5}
   *
   * @param {function}   callback
   *     train callback function
   *     `callback` function is called regularly from the train loop:
   *     `callback(progress, loss, wordsPerSec, learningRate, eta)`
   *
   * @return {Promise}   promise object that resolves to a `FastTextModel`
   *
   */
  trainUnsupervised(url, modelName, kwargs = {}, callback) {
    const self = this;
    return new Promise(function(resolve, reject) {
      self._train(url, modelName, kwargs, callback).then(model => {
        resolve(model);
      }).catch(error => {
        reject(error);
      });
    });
  }

}


class FastTextModel {
  /**
     * `FastTextModel` represents a trained model.
     *
     * @constructor
     *
     * @param {object}       fastTextNative
     *     webassembly object that makes the bridge between js and C++
     */
  constructor(fastTextNative) {
    this.f = fastTextNative;
  }

  /**
     * isQuant
     *
     * @return {bool}   true if the model is quantized
     *
     */
  isQuant() {
    return this.f.isQuant;
  }

  /**
     * getDimension
     *
     * @return {int}    the dimension (size) of a lookup vector (hidden layer)
     *
     */
  getDimension() {
    return this.f.args.dim;
  }

  /**
     * getWordVector
     *
     * @param {string}          word
     *
     * @return {Float32Array}   the vector representation of `word`.
     *
     */
  getWordVector(word) {
    const b = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getWordVector(b, word);

    return heapToFloat32(b);
  }

  /**
     * getSentenceVector
     *
     * @param {string}          text
     *
     * @return {Float32Array}   the vector representation of `text`.
     *
     */
  getSentenceVector(text) {
    if (text.indexOf('\n') != -1) {
      "sentence vector processes one line at a time (remove '\\n')";
    }
    text += '\n';
    const b = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getSentenceVector(b, text);

    return heapToFloat32(b);
  }

  /**
     * getNearestNeighbors
     *
     * returns the nearest `k` neighbors of `word`.
     *
     * @param {string}          word
     * @param {int}             k
     *
     * @return {Array.<Pair.<number, string>>}
     *     words and their corresponding cosine similarities.
     *
     */
  getNearestNeighbors(word, k = 10) {
    return this.f.getNN(word, k);
  }

  /**
     * getAnalogies
     *
     * returns the nearest `k` neighbors of the operation
     * `wordA - wordB + wordC`.
     *
     * @param {string}          wordA
     * @param {string}          wordB
     * @param {string}          wordC
     * @param {int}             k
     *
     * @return {Array.<Pair.<number, string>>}
     *     words and their corresponding cosine similarities
     *
     */
  getAnalogies(wordA, wordB, wordC, k) {
    return this.f.getAnalogies(k, wordA, wordB, wordC);
  }

  /**
     * getWordId
     *
     * Given a word, get the word id within the dictionary.
     * Returns -1 if word is not in the dictionary.
     *
     * @return {int}    word id
     *
     */
  getWordId(word) {
    return this.f.getWordId(word);
  }

  /**
     * getSubwordId
     *
     * Given a subword, return the index (within input matrix) it hashes to.
     *
     * @return {int}    subword id
     *
     */
  getSubwordId(subword) {
    return this.f.getSubwordId(subword);
  }

  /**
     * getSubwords
     *
     * returns the subwords and their indicies.
     *
     * @param {string}          word
     *
     * @return {Pair.<Array.<string>, Array.<int>>}
     *     words and their corresponding indicies
     *
     */
  getSubwords(word) {
    return this.f.getSubwords(word);
  }

  /**
     * getInputVector
     *
     * Given an index, get the corresponding vector of the Input Matrix.
     *
     * @param {int}             ind
     *
     * @return {Float32Array}   the vector of the `ind`'th index
     *
     */
  getInputVector(ind) {
    const b = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getInputVector(b, ind);

    return heapToFloat32(b);
  }

  /**
     * predict
     *
     * Given a string, get a list of labels and a list of corresponding
     * probabilities. k controls the number of returned labels.
     *
     * @param {string}          text
     * @param {int}             k, the number of predictions to be returned
     * @param {number}          probability threshold
     *
     * @return {Array.<Pair.<number, string>>}
     *     labels and their probabilities
     *
     */
  predict(text, k = 1, threshold = 0.0) {
    return this.f.predict(text, k, threshold);
  }

  /**
     * getInputMatrix
     *
     * Get a reference to the full input matrix of a Model. This only
     * works if the model is not quantized.
     *
     * @return {DenseMatrix}
     *     densematrix with functions: `rows`, `cols`, `at(i,j)`
     *
     * example:
     *     let inputMatrix = model.getInputMatrix();
     *     let value = inputMatrix.at(1, 2);
     */
  getInputMatrix() {
    if (this.isQuant()) {
      throw new Error("Can't get quantized Matrix");
    }
    return this.f.getInputMatrix();
  }

  /**
     * getOutputMatrix
     *
     * Get a reference to the full input matrix of a Model. This only
     * works if the model is not quantized.
     *
     * @return {DenseMatrix}
     *     densematrix with functions: `rows`, `cols`, `at(i,j)`
     *
     * example:
     *     let outputMatrix = model.getOutputMatrix();
     *     let value = outputMatrix.at(1, 2);
     */
  getOutputMatrix() {
    if (this.isQuant()) {
      throw new Error("Can't get quantized Matrix");
    }
    return this.f.getOutputMatrix();
  }

  /**
     * getWords
     *
     * Get the entire list of words of the dictionary including the frequency
     * of the individual words. This does not include any subwords. For that
     * please consult the function get_subwords.
     *
     * @return {Pair.<Array.<string>, Array.<int>>}
     *     words and their corresponding frequencies
     *
     */
  getWords() {
    return this.f.getWords();
  }

  /**
     * getLabels
     *
     * Get the entire list of labels of the dictionary including the frequency
     * of the individual labels.
     *
     * @return {Pair.<Array.<string>, Array.<int>>}
     *     labels and their corresponding frequencies
     *
     */
  getLabels() {
    return this.f.getLabels();
  }

  /**
     * getLine
     *
     * Split a line of text into words and labels. Labels must start with
     * the prefix used to create the model (__label__ by default).
     *
     * @param {string}          text
     *
     * @return {Pair.<Array.<string>, Array.<string>>}
     *     words and labels
     *
     */
  getLine(text) {
    return this.f.getLine(text);
  }

  /**
     * saveModel
     *
     * Saves the model file in web assembly in-memory FS and returns a blob
     *
     * @return {Blob}           blob data of the file saved in web assembly FS
     *
     */
  saveModel() {
    this.f.saveModel(modelFileInWasmFs);
    const content = fastTextModule.FS.readFile(modelFileInWasmFs,
      { encoding: 'binary' });
    return new Blob(
      [new Uint8Array(content, content.byteOffset, content.length)],
      { type: ' application/octet-stream' }
    );
  }

  /**
     * test
     *
     * Downloads the test file from the specified url, evaluates the supervised
     * model with it.
     *
     * @param {string}          url
     * @param {int}             k, the number of predictions to be returned
     * @param {number}          probability threshold
     *
     * @return {Promise}   promise object that resolves to a `Meter` object
     *
     * example:
     * model.test("/absolute/url/to/test.txt", 1, 0.0).then((meter) => {
     *     console.log(meter.precision);
     *     console.log(meter.recall);
     *     console.log(meter.f1Score);
     *     console.log(meter.nexamples());
     * });
     *
     */
  test(url, k, threshold) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;

    return new Promise(function(resolve, reject) {
      fetchFunc(url).then(response => {
        return response.arrayBuffer();
      }).then(bytes => {
        const byteArray = new Uint8Array(bytes);
        const FS = fastTextModule.FS;
        FS.writeFile(testFileInWasmFs, byteArray);
      }).then(() =>  {
        const meter = fastTextNative.test(testFileInWasmFs, k, threshold);
        resolve(meter);
      }).catch(error => {
        reject(error);
      });
    });
  }
}


export {FastText, addOnPostRun};
