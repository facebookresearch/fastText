/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "utils.h"

#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace fasttext {

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Args> args,
    std::shared_ptr<Loss> loss,
    int32_t seed)
    : hidden_(args->dim), output_(wo->size(0)), grad_(args->dim), rng(seed) {
  wi_ = wi;
  wo_ = wo;
  args_ = args;
  loss_ = loss;
  osz_ = wo->size(0);
  hsz_ = args->dim;
  lossValue_ = 0.0;
  nexamples_ = 1;
}

Model::Model(const Model& other, int32_t seed)
    : wi_(other.wi_),
      wo_(other.wo_),
      args_(other.args_),
      loss_(other.loss_),
      hidden_(other.hidden_),
      output_(other.output_),
      grad_(other.grad_),
      hsz_(other.hsz_),
      osz_(other.osz_),
      lossValue_(other.lossValue_),
      nexamples_(other.nexamples_),
      rng(seed) {}

void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden)
    const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap,
    Vector& hidden,
    Vector& output) const {
  if (k == Model::kUnlimitedPredictions) {
    k = osz_;
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  if (args_->model != model_name::sup) {
    throw std::invalid_argument("Model needs to be supervised for prediction!");
  }
  heap.reserve(k + 1);
  computeHidden(input, hidden);

  loss_->predict(k, threshold, heap, hidden, output);
}

void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap) {
  predict(input, k, threshold, heap, hidden_, output_);
}

void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr) {
  if (input.size() == 0) {
    return;
  }
  computeHidden(input, hidden_);

  grad_.zero();
  lossValue_ += loss_->forward(
      targets, targetIndex, hidden_, output_, grad_, lr, rng, true);

  nexamples_ += 1;

  if (args_->model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad_, *it, 1.0);
  }
}

real Model::getLoss() const {
  return lossValue_ / nexamples_;
}

real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

} // namespace fasttext
