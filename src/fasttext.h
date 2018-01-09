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
#include <memory>
#include <set>
#include <chrono>
#include <iostream>
#include <queue>
#include <tuple>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
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

  std::atomic<int64_t> tokenCount_;
  std::atomic<real> loss_;

  clock_t start_;
  void signModel(std::ostream&);
  bool checkModel(std::istream&);

  bool quant_;
  int32_t version;

  void startThreads();

 public:
  FastText();

  int32_t getWordId(const std::string&) const;
  int32_t getSubwordId(const std::string&) const;
  FASTTEXT_DEPRECATED(
    "getVector is being deprecated and replaced by getWordVector.")
  void getVector(Vector&, const std::string&) const;
  void getWordVector(Vector&, const std::string&) const;
  void getSubwordVector(Vector&, const std::string&) const;
  void addInputVector(Vector&, int32_t) const;
  inline void getInputVector(Vector& vec, int32_t ind) {
    vec.zero();
    addInputVector(vec, ind);
  }

  const Args getArgs() const;
  std::shared_ptr<const Dictionary> getDictionary() const;
  std::shared_ptr<const Matrix> getInputMatrix() const;
  std::shared_ptr<const Matrix> getOutputMatrix() const;
  void saveVectors();
  void saveModel(const std::string);
  void saveOutput();
  void saveModel();
  void loadModel(std::istream&);
  void loadModel(const std::string&);
  void printInfo(real, real, std::ostream&);

  void supervised(
      Model&,
      real,
      const std::vector<int32_t>&,
      const std::vector<int32_t>&);
  void cbow(Model&, real, const std::vector<int32_t>&);
  void skipgram(Model&, real, const std::vector<int32_t>&);
  std::vector<int32_t> selectEmbeddings(int32_t) const;
  void getSentenceVector(std::istream&, Vector&);
  void quantize(const Args);
  std::tuple<int64_t, double, double> test(std::istream&, int32_t, real = 0.0);
  void predict(std::istream&, int32_t, bool, real = 0.0);
  void predict(
      std::istream&,
      int32_t,
      std::vector<std::pair<real, std::string>>&,
      real = 0.0) const;
  void ngramVectors(std::string);
  void precomputeWordVectors(Matrix&);
  void findNN(
      const Matrix&,
      const Vector&,
      int32_t,
      const std::set<std::string>&,
      std::vector<std::pair<real, std::string>>& results);
  void analogies(int32_t);
  void trainThread(int32_t);
  void train(const Args);

  void loadVectors(std::string);
  int getDimension() const;
  bool isQuant() const;
};
}
