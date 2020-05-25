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

constexpr int32_t kAllLabels = -1;
constexpr real falseNegativeScore = -1.0;

void Meter::log(
    const std::vector<int32_t>& labels,
    const Predictions& predictions) {
  nexamples_++;
  metrics_.gold += labels.size();
  metrics_.predicted += predictions.size();

  for (const auto& prediction : predictions) {
    labelMetrics_[prediction.second].predicted++;

    real score = std::min(std::exp(prediction.first), 1.0f);
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
    if (falseNegativeLabels_) {
      if (!utils::containsSecond(predictions, label)) {
        labelMetrics_[label].scoreVsTrue.emplace_back(falseNegativeScore, 1.0);
      }
    }
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

std::vector<std::pair<uint64_t, uint64_t>> Meter::getPositiveCounts(
    int32_t labelId) const {
  std::vector<std::pair<uint64_t, uint64_t>> positiveCounts;

  const auto& v = scoreVsTrue(labelId);
  uint64_t truePositives = 0;
  uint64_t falsePositives = 0;
  double lastScore = falseNegativeScore - 1.0;

  for (auto it = v.rbegin(); it != v.rend(); ++it) {
    double score = it->first;
    double gold = it->second;
    if (score < 0) { // only reachable recall
      break;
    }
    if (gold == 1.0) {
      truePositives++;
    } else {
      falsePositives++;
    }
    if (score == lastScore && positiveCounts.size()) { // squeeze tied scores
      positiveCounts.back() = {truePositives, falsePositives};
    } else {
      positiveCounts.emplace_back(truePositives, falsePositives);
    }
    lastScore = score;
  }

  return positiveCounts;
}

double Meter::precisionAtRecall(double recallQuery) const {
  return precisionAtRecall(kAllLabels, recallQuery);
}

double Meter::precisionAtRecall(int32_t labelId, double recallQuery) const {
  const auto& precisionRecall = precisionRecallCurve(labelId);
  double bestPrecision = 0.0;
  std::for_each(
      precisionRecall.begin(),
      precisionRecall.end(),
      [&bestPrecision, recallQuery](const std::pair<double, double>& element) {
        if (element.second >= recallQuery) {
          bestPrecision = std::max(bestPrecision, element.first);
        };
      });
  return bestPrecision;
}

double Meter::recallAtPrecision(double precisionQuery) const {
  return recallAtPrecision(kAllLabels, precisionQuery);
}

double Meter::recallAtPrecision(int32_t labelId, double precisionQuery) const {
  const auto& precisionRecall = precisionRecallCurve(labelId);
  double bestRecall = 0.0;
  std::for_each(
      precisionRecall.begin(),
      precisionRecall.end(),
      [&bestRecall, precisionQuery](const std::pair<double, double>& element) {
        if (element.first >= precisionQuery) {
          bestRecall = std::max(bestRecall, element.second);
        };
      });
  return bestRecall;
}

std::vector<std::pair<double, double>> Meter::precisionRecallCurve() const {
  return precisionRecallCurve(kAllLabels);
}

std::vector<std::pair<double, double>> Meter::precisionRecallCurve(
    int32_t labelId) const {
  std::vector<std::pair<double, double>> precisionRecallCurve;
  const auto& positiveCounts = getPositiveCounts(labelId);
  if (positiveCounts.empty()) {
    return precisionRecallCurve;
  }

  uint64_t golds =
      (labelId == kAllLabels) ? metrics_.gold : labelMetrics_.at(labelId).gold;

  auto fullRecall = std::lower_bound(
      positiveCounts.begin(),
      positiveCounts.end(),
      golds,
      utils::compareFirstLess);

  if (fullRecall != positiveCounts.end()) {
    fullRecall = std::next(fullRecall);
  }

  for (auto it = positiveCounts.begin(); it != fullRecall; it++) {
    double precision = 0.0;
    double truePositives = it->first;
    double falsePositives = it->second;
    if (truePositives + falsePositives != 0.0) {
      precision = truePositives / (truePositives + falsePositives);
    }
    double recall = golds != 0 ? (truePositives / double(golds))
                               : std::numeric_limits<double>::quiet_NaN();
    precisionRecallCurve.emplace_back(precision, recall);
  }
  precisionRecallCurve.emplace_back(1.0, 0.0);

  return precisionRecallCurve;
}

std::vector<std::pair<real, real>> Meter::scoreVsTrue(int32_t labelId) const {
  std::vector<std::pair<real, real>> ret;
  if (labelId == kAllLabels) {
    for (const auto& k : labelMetrics_) {
      auto& labelScoreVsTrue = labelMetrics_.at(k.first).scoreVsTrue;
      ret.insert(ret.end(), labelScoreVsTrue.begin(), labelScoreVsTrue.end());
    }
  } else {
    if (labelMetrics_.count(labelId)) {
      ret = labelMetrics_.at(labelId).scoreVsTrue;
    }
  }
  sort(ret.begin(), ret.end());

  return ret;
}

} // namespace fasttext
