/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "matrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"
#include "args.h"
#include "dictionary.h"

namespace fasttext {

class Loss;

class Model {
 protected:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<Loss> loss_;
  bool normalizeGradient_;

 public:
  Model(
      std::shared_ptr<Matrix> wi,
      std::shared_ptr<Matrix> wo,
      std::shared_ptr<Loss> loss,
      bool normalizeGradient);
  Model(const Model& model) = delete;
  Model(Model&& model) = delete;
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;
  inline std::shared_ptr<Loss>& getLoss() { return loss_; }
  class State {
   private:
    real lossValue_;
    int64_t nexamples_;

   public:
    virtual ~State() {}
    Vector hidden;
    Vector output;
    Vector grad;
    std::vector<int32_t> line;
    std::vector<int32_t> labels;
    std::minstd_rand rng;

    State(int32_t hiddenSize, int32_t outputSize, int32_t seed);
    real getLoss() const;
    void incrementNExamples(real loss);
    virtual int64_t getLine(std::ifstream& ifs, std::shared_ptr<Dictionary> dict, model_name modelname);
  };

  void predict(
      const std::vector<int32_t>& input,
      int32_t k,
      real threshold,
      Predictions& heap,
      State& state) const;
  void update(
      const std::vector<int32_t>& input,
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      real lr,
      State& state);
  void computeHidden(const std::vector<int32_t>& input, State& state) const;

  real std_log(real) const;

  static const int32_t kUnlimitedPredictions = -1;
  static const int32_t kAllLabelsAsTarget = -1;
};

} // namespace fasttext
