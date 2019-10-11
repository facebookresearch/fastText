/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace fasttext {

Model::State::State(int32_t hiddenSize, int32_t outputSize, int32_t seed)
    : lossValue_(0.0),
      nexamples_(0),
      hidden(hiddenSize),
      output(outputSize),
      grad(hiddenSize),
      rng(seed) {}

real Model::State::getLoss() const {
  return lossValue_ / nexamples_;
}

void Model::State::incrementNExamples(real loss) {
  lossValue_ += loss;
  nexamples_++;
}

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Loss> loss,
    bool normalizeGradient)
    : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
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
    State& state) const {
  if (k == Model::kUnlimitedPredictions) {
    k = wo_->size(0); // output size
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  heap.reserve(k + 1);
  computeHidden(input, state);

  loss_->predict(k, threshold, heap, state);
}

void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state) {
  if (input.size() == 0) {
    return;
  }
  computeHidden(input, state);

  Vector& grad = state.grad;
  grad.zero();
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
  state.incrementNExamples(lossValue);

  if (normalizeGradient_) {
    grad.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad, *it, 1.0);
  }
}

real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

} // namespace fasttext
