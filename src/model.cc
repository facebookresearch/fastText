/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"
#include <assert.h>
#include <algorithm>
#include "args.h"
#include "utils.h"

extern Args args;

real Model::lr_ = MIN_LR;

Model::Model(Matrix& wi, Matrix& wo, int32_t hsz, real lr, int32_t seed)
            : wi_(wi), wo_(wo), hidden_(hsz), output_(wo.m_),
              grad_(hsz), rng(seed) {
  isz_ = wi.m_;
  osz_ = wo.m_;
  hsz_ = hsz;
  lr_ = lr;
  npos = 0;
}

Model::~Model() {
}

void Model::setLearningRate(real lr) {
  lr_ = (lr < MIN_LR) ? MIN_LR : lr;
}

real Model::getLearningRate() {
  return lr_;
}

void Model::binaryLogistic(int32_t target, bool label, double& loss) {
  real score = utils::sigmoid(wo_.dotRow(hidden_, target));
  real alpha = lr_ * (real(label) - score);
  grad_.addRow(wo_, target, alpha);
  wo_.addRow(hidden_, target, alpha);
  if (label) {
    loss -= utils::log(score);
  } else {
    loss -= utils::log(1.0 - score);
  }
}

void Model::negativeSampling(int32_t target, double& loss, int32_t& N) {
  grad_.zero();
  for (int32_t n = 0; n <= args.neg; n++) {
    if (n == 0) {
      binaryLogistic(target, true, loss);
    } else {
      binaryLogistic(getNegative(target), false, loss);
    }
    N += 1;
  }
}

void Model::hierarchicalSoftmax(int32_t target, double& loss, int32_t& N) {
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    binaryLogistic(pathToRoot[i], binaryCode[i], loss);
  }
  N += 1;
}

void Model::softmax(int32_t target, double& loss, int32_t& N) {
  grad_.zero();
  output_.mul(wo_, hidden_);
  real max = 0.0, z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output_[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output_[i] = exp(output_[i] - max);
    z += output_[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    output_[i] /= z;
    real alpha = lr_ * (label - output_[i]);
    grad_.addRow(wo_, i, alpha);
    wo_.addRow(hidden_, i, alpha);
  }
  loss -= utils::log(output_[target]);
  N++;
}

int32_t Model::predict(const std::vector<int32_t>& input) {
  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  if (args.loss == loss_name::hs) {
    real max = -1e10;
    int32_t argmax = -1;
    dfs(2 * osz_ - 2, 0.0, max, argmax);
    return argmax;
  } else {
    output_.mul(wo_, hidden_);
    return output_.argmax();
  }
}

void Model::dfs(int32_t node, real score, real& max, int32_t& argmax) {
  if (score < max) return;
  if (tree[node].left == -1 && tree[node].right == -1) {
    max = score;
    argmax = node;
    return;
  }
  real f = utils::sigmoid(wo_.dotRow(hidden_, node - osz_));
  dfs(tree[node].left, score + utils::log(1.0 - f), max, argmax);
  dfs(tree[node].right, score + utils::log(f), max, argmax);
}

void Model::update(const std::vector<int32_t>& input, int32_t target, double& loss,
                   int32_t& N) {
  assert(target >= 0 && target < osz_);
  if (input.size() == 0) return;

  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  if (args.loss == loss_name::ns) {
    negativeSampling(target, loss, N);
  } else if (args.loss == loss_name::hs) {
    hierarchicalSoftmax(target, loss, N);
  } else {
    softmax(target, loss, N);
  }

  if (args.model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_.addRow(grad_, *it, 1.0);
  }
}

void Model::setLabelFreq(const std::vector<int64_t>& freq) {
  assert(freq.size() == osz_);
  if (args.loss == loss_name::ns) {
    initTableNegatives(freq);
  }
  if (args.loss == loss_name::hs) {
    buildTree(freq);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& freq) {
  real N = 0.0;
  for (int32_t i = 0; i < freq.size(); i++) {
    if (args.sampling == sampling_name::log) {
      N += log(freq[i]);
    } else if (args.sampling == sampling_name::sqrt) {
      N += sqrt(freq[i]);
    } else {
      N += 1.0;
    }
  }
  for (int32_t i = 0; i < freq.size(); i++) {
    real c = 0.0;
    if (args.sampling == sampling_name::log) {
      c = log(freq[i]);
    } else if (args.sampling == sampling_name::sqrt) {
      c = sqrt(freq[i]);
    } else {
      c = 1.0;
    }
    int32_t n = (int32_t)ceil(c * ((real)NEGATIVE_TABLE_SIZE / N));
    for (int32_t j = 0; j < n; j++) {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[npos];
    npos = (npos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& freq) {
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].freq = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].freq = freq[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].freq < tree[node].freq) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].freq = tree[mini[0]].freq + tree[mini[1]].freq;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}
