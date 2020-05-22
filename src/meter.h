/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "dictionary.h"
#include "real.h"
#include "utils.h"

namespace fasttext {

class Meter {
  struct Metrics {
    uint64_t gold;
    uint64_t predicted;
    uint64_t predictedGold;
    mutable std::vector<std::pair<real, real>> scoreVsTrue;

    Metrics() : gold(0), predicted(0), predictedGold(0), scoreVsTrue() {}

    double precision() const {
      if (predicted == 0) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      return predictedGold / double(predicted);
    }
    double recall() const {
      if (gold == 0) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      return predictedGold / double(gold);
    }
    double f1Score() const {
      if (predicted + gold == 0) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      return 2 * predictedGold / double(predicted + gold);
    }

    std::vector<std::pair<real, real>> getScoreVsTrue() {
      return scoreVsTrue;
    }
  };
  std::vector<std::pair<uint64_t, uint64_t>> getPositiveCounts(
      int32_t labelId) const;

 public:
  Meter() = delete;
  explicit Meter(bool falseNegativeLabels)
      : metrics_(),
        nexamples_(0),
        labelMetrics_(),
        falseNegativeLabels_(falseNegativeLabels) {}

  void log(const std::vector<int32_t>& labels, const Predictions& predictions);

  double precision(int32_t);
  double recall(int32_t);
  double f1Score(int32_t);
  std::vector<std::pair<real, real>> scoreVsTrue(int32_t labelId) const;
  double precisionAtRecall(int32_t labelId, double recall) const;
  double precisionAtRecall(double recall) const;
  double recallAtPrecision(int32_t labelId, double recall) const;
  double recallAtPrecision(double recall) const;
  std::vector<std::pair<double, double>> precisionRecallCurve(
      int32_t labelId) const;
  std::vector<std::pair<double, double>> precisionRecallCurve() const;
  double precision() const;
  double recall() const;
  double f1Score() const;
  uint64_t nexamples() const {
    return nexamples_;
  }
  void writeGeneralMetrics(std::ostream& out, int32_t k) const;

 private:
  Metrics metrics_{};
  uint64_t nexamples_;
  std::unordered_map<int32_t, Metrics> labelMetrics_;
  bool falseNegativeLabels_;
};

} // namespace fasttext
