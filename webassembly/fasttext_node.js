const Module = require('./fasttext_wasm');
const fs = require('fs/promises');

class FastText {
  static _modulePromise = Module();

  /**
   * Loads the model file from the specified path.
   *
   * @param {string} modelPath the path of the model file.
   * @return {Promise<FastText>} promise object that resolves to a `FastText` instance.
   *
   */
  static async from(modelPath) {
    const module = await this._modulePromise;
    const wasmModelPath = 'model.bin';
    // https://emscripten.org/docs/api_reference/Filesystem-API.html
    await module.FS.writeFile(wasmModelPath, new Uint8Array(await fs.readFile(modelPath)));
    try {
      const model = new module.FastText();
      await model.loadModel(wasmModelPath);
      return new FastText(model);
    } finally {
      await module.FS.unlink(wasmModelPath);
    }
  } 

  constructor(model) {
    this._model = model;
  }
  
  /**
     * Given a string, get a list of labels and a list of corresponding
     * probabilities. k controls the number of returned labels.
     *
     * @param {string} text
     * @param {int} k, the number of predictions to be returned
     * @param {number} probability threshold
     * @param {string} labelPrefix prefix used by the labels
     *
     * @return {Array<[string, number]>} labels and their probabilities
     *
     */
  predict(text, k = 1, threshold = 0.0, labelPrefix = '__label__') {
    const vector = this._model.predict(text, k, threshold);
    const result = new Array(vector.size());
    for (let i = 0; i < vector.size(); i++) {
      const [prob, label] = vector.get(i);
      result[i] = [label.replace(labelPrefix, ''), prob];
    }
    return result;
  }
}

exports.FastText = FastText;
