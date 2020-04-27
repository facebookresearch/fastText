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
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>

#include "args.h"
#include "densematrix.h"
#include "dictionary.h"
#include "matrix.h"
#include "meter.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class FastText {
 public:
  using TrainCallback =
      std::function<void(float, float, double, double, int64_t)>;

 protected:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<Matrix> input_;
  std::shared_ptr<Matrix> output_;
  std::shared_ptr<Model> model_;
  std::atomic<int64_t> tokenCount_{};
  std::atomic<real> loss_{};
  std::chrono::steady_clock::time_point start_;
  bool quant_;
  int32_t version;
  std::unique_ptr<DenseMatrix> wordVectors_;
  std::exception_ptr trainException_;

  void signModel(std::ostream&);
  bool checkModel(std::istream&);
  void startThreads(const TrainCallback& callback = {});
  void addInputVector(Vector&, int32_t) const;
  void trainThread(int32_t, const TrainCallback& callback);
  std::vector<std::pair<real, std::string>> getNN(
      const DenseMatrix& wordVectors,
      const Vector& queryVec,
      int32_t k,
      const std::set<std::string>& banSet);
  void lazyComputeWordVectors();
  void printInfo(real, real, std::ostream&);
  std::shared_ptr<Matrix> getInputMatrixFromFile(const std::string&) const;
  std::shared_ptr<Matrix> createRandomMatrix() const;
  std::shared_ptr<Matrix> createTrainOutputMatrix() const;
  std::vector<int64_t> getTargetCounts() const;
  std::shared_ptr<Loss> createLoss(std::shared_ptr<Matrix>& output);
  void supervised(
      Model::State& state,
      real lr,
      const std::vector<int32_t>& line,
      const std::vector<int32_t>& labels);
  void cbow(Model::State& state, real lr, const std::vector<int32_t>& line);
  void skipgram(Model::State& state, real lr, const std::vector<int32_t>& line);
  std::vector<int32_t> selectEmbeddings(int32_t cutoff) const;
  void precomputeWordVectors(DenseMatrix& wordVectors);
  bool keepTraining(const int64_t ntokens) const;
  void buildModel();
  std::tuple<int64_t, double, double> progressInfo(real progress);

 public:
  FastText();

  int32_t getWordId(const std::string& word) const;

  int32_t getSubwordId(const std::string& subword) const;

  int32_t getLabelId(const std::string& label) const;

  void getWordVector(Vector& vec, const std::string& word) const;

  void getSubwordVector(Vector& vec, const std::string& subword) const;

  inline void getInputVector(Vector& vec, int32_t ind) {
    vec.zero();
    addInputVector(vec, ind);
  }

  const Args getArgs() const;

  std::shared_ptr<const Dictionary> getDictionary() const;

  std::shared_ptr<const DenseMatrix> getInputMatrix() const;

  void setMatrices(
      const std::shared_ptr<DenseMatrix>& inputMatrix,
      const std::shared_ptr<DenseMatrix>& outputMatrix);

  std::shared_ptr<const DenseMatrix> getOutputMatrix() const;

  void saveVectors(const std::string& filename);

  void saveModel(const std::string& filename);

  void saveOutput(const std::string& filename);

  void loadModel(std::istream& in);

  void loadModel(const std::string& filename);

  void getSentenceVector(std::istream& in, Vector& vec);

  void quantize(const Args& qargs, const TrainCallback& callback = {});

  std::tuple<int64_t, double, double>
  test(std::istream& in, int32_t k, real threshold = 0.0);

  void test(std::istream& in, int32_t k, real threshold, Meter& meter) const;

  void predict(
      int32_t k,
      const std::vector<int32_t>& words,
      Predictions& predictions,
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

  void train(const Args& args, const TrainCallback& callback = {});

  void abort();

  int getDimension() const;

  bool isQuant() const;

  class AbortError : public std::runtime_error {
   public:
    AbortError() : std::runtime_error("Aborted.") {}
  };
};
} // namespace fasttext
