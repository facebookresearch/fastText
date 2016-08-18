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

TopIndexScoresCollector::TopIndexScoresCollector(int64_t top_k)
  : top_k_(top_k),
    result_(top_k == 1 ? some_index_score_t::single : some_index_score_t::multi) {
  assert(top_k_ >= 1);
  if (top_k_ == 1) {
    result_.datum.first = -1;
    result_.datum.second = -1e10;
  } else {
    result_.data.reserve(top_k_ + 1);
  }
}

bool TopIndexScoresCollector::shouldAdd(real score) {
  if (top_k_ == 1) {
    return score > result_.datum.second;
  } else {
    return
      result_.data.size() < static_cast<size_t>(top_k_) ||
      score > result_.data.front().second;
  }
}

bool TopIndexScoresCollector::compIndexScorePairs(
    const index_score_t &l,
    const index_score_t &r
  ) {
  return l.second > r.second;
}

void TopIndexScoresCollector::add(int64_t index, real score) {
  assert(shouldAdd(score));
  if (top_k_ == 1) {
    result_.datum.first = index;
    result_.datum.second = score;
    return;
  }
  result_.data.push_back(std::make_pair(index, score));
  std::push_heap(
    result_.data.begin(),
    result_.data.end(),
    TopIndexScoresCollector::compIndexScorePairs
  );
  if (result_.data.size() > static_cast<size_t>(top_k_)) {
    std::pop_heap(
      result_.data.begin(),
      result_.data.end(),
      TopIndexScoresCollector::compIndexScorePairs
    );
    result_.data.pop_back();
  }
}

some_index_score_t TopIndexScoresCollector::result() {
  if (top_k_ > 1) {
    std::sort_heap(
      result_.data.begin(),
      result_.data.end(),
      TopIndexScoresCollector::compIndexScorePairs
    );
  }

  return std::move(result_);
}

real Model::lr_ = MIN_LR;

Model::Model(Matrix& wi, Matrix& wo, int32_t hsz, real lr, int32_t seed)
            : wi_(wi), wo_(wo), hidden_(hsz), output_(wo.m_),
              grad_(hsz), rng(seed) {
  isz_ = wi.m_;
  osz_ = wo.m_;
  hsz_ = hsz;
  lr_ = lr;
  negpos = 0;
}

void Model::setLearningRate(real lr) {
  lr_ = (lr < MIN_LR) ? MIN_LR : lr;
}

real Model::getLearningRate() {
  return lr_;
}

real Model::binaryLogistic(int32_t target, bool label) {
  real score = utils::sigmoid(wo_.dotRow(hidden_, target));
  real alpha = lr_ * (real(label) - score);
  grad_.addRow(wo_, target, alpha);
  wo_.addRow(hidden_, target, alpha);
  if (label) {
    return -utils::log(score);
  } else {
    return -utils::log(1.0 - score);
  }
}

real Model::negativeSampling(int32_t target) {
  real loss = 0.0;
  grad_.zero();
  for (int32_t n = 0; n <= args.neg; n++) {
    if (n == 0) {
      loss += binaryLogistic(target, true);
    } else {
      loss += binaryLogistic(getNegative(target), false);
    }
  }
  return loss;
}

real Model::hierarchicalSoftmax(int32_t target) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i]);
  }
  return loss;
}

real Model::softmax(int32_t target) {
  grad_.zero();
  output_.mul(wo_, hidden_);
  real max = output_[0], z = 0.0;
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
  return -utils::log(output_[target]);
}

some_index_score_t Model::predictOneOrMore(int32_t top_k, const std::vector<int32_t>& input) {
  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  TopIndexScoresCollector collector(top_k);
  if (args.loss == loss_name::hs) {
    dfs(2 * osz_ - 2, 0.0, collector);
  } else {
    output_.mul(wo_, hidden_);
    for (int64_t i = 0; i < output_.m_; i += 1) {
      if (collector.shouldAdd(output_[i])) {
        collector.add(i, output_[i]);
      }
    }
  }

  return collector.result();
}

int32_t Model::predict(const std::vector<int32_t>& input) {
  auto result = predictOneOrMore(1, input);
  return result.datum.first;
}

std::vector<int64_t> Model::predict(int32_t top_k, const std::vector<int32_t>& input) {
  assert(top_k > 1);
  auto result = predictOneOrMore(top_k, input);
  std::vector<int64_t> label_indices;
  label_indices.reserve(top_k);
  for (const auto& pair : result.data) {
    label_indices.push_back(pair.first);
  }
  return label_indices;
}

void Model::dfs(int32_t node, real score, TopIndexScoresCollector& collector) {
  if (!collector.shouldAdd(score)) {
    return;
  }
  if (tree[node].left == -1 && tree[node].right == -1) {
    collector.add(node, score);
    return;
  }

  real f = utils::sigmoid(wo_.dotRow(hidden_, node - osz_));
  dfs(tree[node].left, score + utils::log(1.0 - f), collector);
  dfs(tree[node].right, score + utils::log(f), collector);
}

real Model::update(const std::vector<int32_t>& input, int32_t target) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return 0.0;
  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  real loss;
  if (args.loss == loss_name::ns) {
    loss = negativeSampling(target);
  } else if (args.loss == loss_name::hs) {
    loss = hierarchicalSoftmax(target);
  } else {
    loss = softmax(target);
  }

  if (args.model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_.addRow(grad_, *it, 1.0);
  }
  return loss;
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  assert(counts.size() == osz_);
  if (args.loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args.loss == loss_name::hs) {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& counts) {
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
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
