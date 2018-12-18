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
#include "matrix.h"
#include "qmatrix.h"
#include "real.h"
#include "vector.h"

namespace fasttext {

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t count;
  bool binary;
};

class Model {
 protected:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<QMatrix> qwi_;
  std::shared_ptr<QMatrix> qwo_;
  std::shared_ptr<Args> args_;
  Vector hidden_;
  Vector output_;
  Vector grad_;
  int32_t hsz_;
  int32_t osz_;
  real loss_;
  int64_t nexamples_;
  std::vector<real> t_sigmoid_;
  std::vector<real> t_log_;
  // used for negative sampling:
  std::vector<int32_t> negatives_;
  size_t negpos;
  // used for hierarchical softmax:
  std::vector<std::vector<int32_t>> paths;
  std::vector<std::vector<bool>> codes;
  std::vector<Node> tree;

  static bool comparePairs(
      const std::pair<real, int32_t>&,
      const std::pair<real, int32_t>&);

  int32_t getNegative(int32_t target);
  void initSigmoid();
  void initLog();
  void computeOutput(Vector&, Vector&) const;

  static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

 public:
  Model(
      std::shared_ptr<Matrix>,
      std::shared_ptr<Matrix>,
      std::shared_ptr<Args>,
      int32_t);

  real binaryLogistic(int32_t, bool, real);
  real negativeSampling(int32_t, real);
  real hierarchicalSoftmax(int32_t, real);
  real softmax(int32_t, real);
  real oneVsAll(const std::vector<int32_t>&, real);

  void predict(
      const std::vector<int32_t>&,
      int32_t,
      real,
      std::vector<std::pair<real, int32_t>>&,
      Vector&,
      Vector&) const;
  void predict(
      const std::vector<int32_t>&,
      int32_t,
      real,
      std::vector<std::pair<real, int32_t>>&);
  void dfs(
      int32_t,
      real,
      int32_t,
      real,
      std::vector<std::pair<real, int32_t>>&,
      Vector&) const;
  void findKBest(
      int32_t,
      real,
      std::vector<std::pair<real, int32_t>>&,
      Vector&,
      Vector&) const;
  void update(
      const std::vector<int32_t>&,
      const std::vector<int32_t>&,
      int32_t,
      real);
  real computeLoss(const std::vector<int32_t>&, int32_t, real);
  void computeHidden(const std::vector<int32_t>&, Vector&) const;
  void computeOutputSigmoid(Vector&, Vector&) const;
  void computeOutputSoftmax(Vector&, Vector&) const;
  void computeOutputSoftmax();

  void setTargetCounts(const std::vector<int64_t>&);
  void initTableNegatives(const std::vector<int64_t>&);
  void buildTree(const std::vector<int64_t>&);
  real getLoss() const;
  real sigmoid(real) const;
  real log(real) const;
  real std_log(real) const;

  std::minstd_rand rng;
  bool quant_;
  void
  setQuantizePointer(std::shared_ptr<QMatrix>, std::shared_ptr<QMatrix>, bool);

  static const int32_t kUnlimitedPredictions = -1;
  static const int32_t kAllLabelsAsTarget = -1;
};

} // namespace fasttext
