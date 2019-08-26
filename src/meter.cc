/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meter.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

namespace fasttext {

void Meter::log(
    const std::vector<int32_t>& labels,
    const Predictions& predictions) {
  nexamples_++;
  metrics_.gold += labels.size();
  metrics_.predicted += predictions.size();

  for (const auto& prediction : predictions) {
    labelMetrics_[prediction.second].predicted++;

    real score = std::exp(prediction.first);
    real gold = 0.0;
    if (utils::contains(labels, prediction.second)) {
      labelMetrics_[prediction.second].predictedGold++;
      metrics_.predictedGold++;
      gold = 1.0;
    }
    labelMetrics_[prediction.second].scoreVsTrue.emplace_back(score, gold);
  }

  for (const auto& label : labels) {
    labelMetrics_[label].gold++;
  }
}

double Meter::precision(int32_t i) {
  return labelMetrics_[i].precision();
}

double Meter::recall(int32_t i) {
  return labelMetrics_[i].recall();
}

double Meter::f1Score(int32_t i) {
  return labelMetrics_[i].f1Score();
}

double Meter::precision() const {
  return metrics_.precision();
}

double Meter::recall() const {
  return metrics_.recall();
}

double Meter::f1Score() const {
  const double precision = this->precision();
  const double recall = this->recall();
  if (precision + recall != 0) {
    return 2 * precision * recall / (precision + recall);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

void Meter::writeGeneralMetrics(std::ostream& out, int32_t k) const {
  out << "N"
      << "\t" << nexamples_ << std::endl;
  out << std::setprecision(3);
  out << "P@" << k << "\t" << metrics_.precision() << std::endl;
  out << "R@" << k << "\t" << metrics_.recall() << std::endl;
}

} // namespace fasttext
