/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"
#include "Vector.h"
#include "Real.h"
#include <vector>
#include <random>

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t freq;
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

    std::vector<int32_t> negatives;
    size_t npos;
    std::vector< std::vector<int32_t> > paths;
    std::vector< std::vector<bool> > codes;
    std::vector<Node> tree;

    static const int32_t NEGATIVE_TABLE_SIZE = 10000000;
    static constexpr real MIN_LR = 0.000001;

  public:
    Model(Matrix&, Matrix&, int32_t, real, int32_t);
    ~Model();

    void setLearningRate(real);
    real getLearningRate();

    void binaryLogistic(int32_t, int32_t, double&);
    void negativeSampling(int32_t, double&, int32_t&);
    void hierarchicalSoftmax(int32_t, double&, int32_t&);
    void softmax(int32_t, double&, int32_t&);

    int32_t predict(const std::vector<int32_t>&);
    void dfs(int32_t, real, real&, int32_t&);
    void update(const std::vector<int32_t>&, int32_t, double&, int32_t&);

    void setLabelFreq(const std::vector<int64_t>&);
    void initTableNegatives(const std::vector<int64_t>&);
    int32_t getNegative(int32_t target);
    void buildTree(const std::vector<int64_t>&);

    std::minstd_rand rng;
};

#endif
