/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
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

  int32_t getWordId(const std::string&) const;

  int32_t getSubwordId(const std::string&) const;

  void getWordVector(Vector&, const std::string&) const;

  void getSubwordVector(Vector&, const std::string&) const;

  inline void getInputVector(Vector& vec, int32_t ind) {
    vec.zero();
    addInputVector(vec, ind);
  }

  const Args getArgs() const;

  std::shared_ptr<const Dictionary> getDictionary() const;

  std::shared_ptr<const Matrix> getInputMatrix() const;

  std::shared_ptr<const Matrix> getOutputMatrix() const;

  void saveVectors(const std::string&);

  void saveModel(const std::string&);

  void saveOutput(const std::string&);

  void loadModel(std::istream&);

  void loadModel(const std::string&);

  void getSentenceVector(std::istream&, Vector&);

  void quantize(const Args);

  std::tuple<int64_t, double, double> test(std::istream&, int32_t, real = 0.0);

  void test(std::istream&, int32_t, real, Meter&) const;

  void predict(
      int32_t,
      const std::vector<int32_t>&,
      std::vector<std::pair<real, int32_t>>&,
      real = 0.0) const;

  bool predictLine(
      std::istream&,
      std::vector<std::pair<real, std::string>>&,
      int32_t,
      real) const;

  std::vector<std::pair<std::string, Vector>> getNgramVectors(
      const std::string& word) const;

  std::vector<std::pair<real, std::string>> getNN(const std::string&, int32_t);

  std::vector<std::pair<real, std::string>> getAnalogies(
      int32_t,
      const std::string&,
      const std::string&,
      const std::string&);

  void train(const Args);

  void loadVectors(std::string);

  int getDimension() const;

  bool isQuant() const;

  FASTTEXT_DEPRECATED(
      "getVector is being deprecated and replaced by getWordVector.")
  void getVector(Vector&, const std::string&) const;

  FASTTEXT_DEPRECATED(
      "ngramVectors is being deprecated and replaced by getNgramVectors.")
  void ngramVectors(std::string);

  FASTTEXT_DEPRECATED(
      "analogies is being deprecated and replaced by getAnalogies.")
  void analogies(int32_t);

  FASTTEXT_DEPRECATED("supervised is being deprecated.")
  void supervised(
      Model&,
      real,
      const std::vector<int32_t>&,
      const std::vector<int32_t>&);

  FASTTEXT_DEPRECATED("cbow is being deprecated.")
  void cbow(Model&, real, const std::vector<int32_t>&);

  FASTTEXT_DEPRECATED("skipgram is being deprecated.")
  void skipgram(Model&, real, const std::vector<int32_t>&);

  FASTTEXT_DEPRECATED("selectEmbeddings is being deprecated.")
  std::vector<int32_t> selectEmbeddings(int32_t) const;

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
  void precomputeWordVectors(Matrix&);

  FASTTEXT_DEPRECATED("findNN is being deprecated and replaced by getNN.")
  void findNN(
      const Matrix&,
      const Vector&,
      int32_t,
      const std::set<std::string>&,
      std::vector<std::pair<real, std::string>>& results);
};
} // namespace fasttext
