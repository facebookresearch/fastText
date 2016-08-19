/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_MODEL_H
#define FASTTEXT_MODEL_H

#include <vector>
#include <random>
#include <utility>

#include "matrix.h"
#include "vector.h"
#include "real.h"

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t count;
  bool binary;
};

class Model {
  private:
    Matrix& wi_;
    Matrix& wo_;
    Vector hidden_;
    Vector output_;
    Vector grad_;
    int32_t hsz_;
    int32_t isz_;
    int32_t osz_;

    static real lr_;

    static bool comparePairs(const std::pair<real, int32_t>&,
                             const std::pair<real, int32_t>&);

    std::vector<int32_t> negatives;
    size_t negpos;
    std::vector< std::vector<int32_t> > paths;
    std::vector< std::vector<bool> > codes;
    std::vector<Node> tree;

    static const int32_t NEGATIVE_TABLE_SIZE = 10000000;
    static constexpr real MIN_LR = 0.000001;

  public:
    Model(Matrix&, Matrix&, int32_t, real, int32_t);

    void setLearningRate(real);
    real getLearningRate();

    real binaryLogistic(int32_t, bool);
    real negativeSampling(int32_t);
    real hierarchicalSoftmax(int32_t);
    real softmax(int32_t);

    void predict(const std::vector<int32_t>&, int32_t,
                 std::vector<std::pair<real, int32_t>>&);
    void dfs(int32_t, int32_t, real, std::vector<std::pair<real, int32_t>>&);
    void findKBest(int32_t, std::vector<std::pair<real, int32_t>>&);
    real update(const std::vector<int32_t>&, int32_t);
    void computeHidden(const std::vector<int32_t>&);

    void setTargetCounts(const std::vector<int64_t>&);
    void initTableNegatives(const std::vector<int64_t>&);
    int32_t getNegative(int32_t target);
    void buildTree(const std::vector<int64_t>&);

    std::minstd_rand rng;
};

#endif
