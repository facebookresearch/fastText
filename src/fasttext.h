/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "meter.h"
#include "model.h"
#include "qmatrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class FastText {
 protected:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;

  std::shared_ptr<Matrix> input_;
  std::shared_ptr<Matrix> output_;

  std::shared_ptr<QMatrix> qinput_;
  std::shared_ptr<QMatrix> qoutput_;

  std::shared_ptr<Model> model_;

  std::atomic<int64_t> tokenCount_{};
  std::atomic<real> loss_{};

  std::chrono::steady_clock::time_point start_;
  void signModel(std::ostream&);
  bool checkModel(std::istream&);
  void startThreads();
  void addInputVector(Vector&, int32_t) const;
  void trainThread(int32_t);
  std::vector<std::pair<real, std::string>> getNN(
      const Matrix& wordVectors,
      const Vector& queryVec,
      int32_t k,
      const std::set<std::string>& banSet);
  void lazyComputeWordVectors();
  void printInfo(real, real, std::ostream&);

  bool quant_;
  int32_t version;
  std::unique_ptr<Matrix> wordVectors_;

 public:
  FastText();

  int32_t getWordId(const std::string& word) const;

  int32_t getSubwordId(const std::string& subword) const;

  void getWordVector(Vector& vec, const std::string& word) const;

  void getSubwordVector(Vector& vec, const std::string& subword) const;

  inline void getInputVector(Vector& vec, int32_t ind) {
    vec.zero();
    addInputVector(vec, ind);
  }

  const Args getArgs() const;

  std::shared_ptr<const Dictionary> getDictionary() const;

  std::shared_ptr<const Matrix> getInputMatrix() const;

  std::shared_ptr<const Matrix> getOutputMatrix() const;

  void saveVectors(const std::string& filename);

  void saveModel(const std::string& filename);

  void saveOutput(const std::string& filename);

  void loadModel(std::istream& in);

  void loadModel(const std::string& filename);

  void getSentenceVector(std::istream& in, Vector& vec);

  void quantize(const Args& qargs);

  std::tuple<int64_t, double, double>
  test(std::istream& in, int32_t k, real threshold = 0.0);

  void test(std::istream& in, int32_t k, real threshold, Meter& meter) const;

  void predict(
      int32_t k,
      const std::vector<int32_t>& words,
      std::vector<std::pair<real, int32_t>>& predictions,
      real threshold = 0.0) const;

  bool predictLine(
      std::istream& in,
      std::vector<std::pair<real, std::string>>& predictions,
      int32_t k,
      real threshold) const;

  std::vector<std::pair<std::string, Vector>> getNgramVectors(
      const std::string& word) const;

  std::vector<std::pair<real, std::string>> getNN(
      const std::string& word,
      int32_t k);

  std::vector<std::pair<real, std::string>> getAnalogies(
      int32_t k,
      const std::string& wordA,
      const std::string& wordB,
      const std::string& wordC);

  void train(const Args& args);

  void loadVectors(const std::string& filename);

  int getDimension() const;

  bool isQuant() const;

  FASTTEXT_DEPRECATED(
      "getVector is being deprecated and replaced by getWordVector.")
  void getVector(Vector& vec, const std::string& word) const;

  FASTTEXT_DEPRECATED(
      "ngramVectors is being deprecated and replaced by getNgramVectors.")
  void ngramVectors(std::string word);

  FASTTEXT_DEPRECATED(
      "analogies is being deprecated and replaced by getAnalogies.")
  void analogies(int32_t k);

  FASTTEXT_DEPRECATED("supervised is being deprecated.")
  void supervised(
      Model& model,
      real lr,
      const std::vector<int32_t>& line,
      const std::vector<int32_t>& labels);

  FASTTEXT_DEPRECATED("cbow is being deprecated.")
  void cbow(Model& model, real lr, const std::vector<int32_t>& line);

  FASTTEXT_DEPRECATED("skipgram is being deprecated.")
  void skipgram(Model& model, real lr, const std::vector<int32_t>& line);

  FASTTEXT_DEPRECATED("selectEmbeddings is being deprecated.")
  std::vector<int32_t> selectEmbeddings(int32_t cutoff) const;

  FASTTEXT_DEPRECATED(
      "saveVectors is being deprecated, please use the other signature.")
  void saveVectors();

  FASTTEXT_DEPRECATED(
      "saveOutput is being deprecated, please use the other signature.")
  void saveOutput();

  FASTTEXT_DEPRECATED(
      "saveModel is being deprecated, please use the other signature.")
  void saveModel();

  FASTTEXT_DEPRECATED("precomputeWordVectors is being deprecated.")
  void precomputeWordVectors(Matrix& wordVectors);

  FASTTEXT_DEPRECATED("findNN is being deprecated and replaced by getNN.")
  void findNN(
      const Matrix& wordVectors,
      const Vector& query,
      int32_t k,
      const std::set<std::string>& banSet,
      std::vector<std::pair<real, std::string>>& results);
};
} // namespace fasttext
