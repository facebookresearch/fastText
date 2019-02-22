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

#include "args.h"
#include "loss.h"
#include "matrix.h"
#include "real.h"
#include "vector.h"

namespace fasttext {

class Model {
 protected:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<Args> args_;
  std::shared_ptr<Loss> loss_;
  Vector hidden_;
  Vector output_;
  Vector grad_;
  int32_t hsz_;
  int32_t osz_;
  real lossValue_;
  int64_t nexamples_;

 public:
  Model(
      std::shared_ptr<Matrix> wi,
      std::shared_ptr<Matrix> wo,
      std::shared_ptr<Args> args,
      std::shared_ptr<Loss> loss,
      int32_t seed);
  Model(const Model& model, int32_t seed);
  Model(const Model& model) = delete;
  Model(Model&& model) = delete;
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;

  void predict(
      const std::vector<int32_t>&,
      int32_t,
      real,
      Predictions&,
      Vector&,
      Vector&) const;
  void predict(const std::vector<int32_t>&, int32_t, real, Predictions&);
  void update(
      const std::vector<int32_t>&,
      const std::vector<int32_t>&,
      int32_t,
      real);
  void computeHidden(const std::vector<int32_t>&, Vector&) const;

  real getLoss() const;
  real std_log(real) const;

  std::minstd_rand rng;

  static const int32_t kUnlimitedPredictions = -1;
  static const int32_t kAllLabelsAsTarget = -1;
};

} // namespace fasttext
