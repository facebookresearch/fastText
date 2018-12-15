/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "dictionary.h"
#include "real.h"

namespace fasttext {

class Meter {
  struct Metrics {
    uint64_t gold;
    uint64_t predicted;
    uint64_t predictedGold;

    Metrics() : gold(0), predicted(0), predictedGold(0) {}

    double precision() const {
      return predictedGold / double(predicted);
    }
    double recall() const {
      return predictedGold / double(gold);
    }
    double f1Score() const {
      return 2 * predictedGold / double(predicted + gold);
    }
  };

 public:
  Meter() : metrics_(), nexamples_(0), labelMetrics_() {}

  void log(
      const std::vector<int32_t>& labels,
      const std::vector<std::pair<real, int32_t>>& predictions);

  double precision(int32_t);
  double recall(int32_t);
  double f1Score(int32_t);
  double precision() const;
  double recall() const;
  uint64_t nexamples() const {
    return nexamples_;
  }
  void writeGeneralMetrics(std::ostream& out, int32_t k) const;

 private:
  Metrics metrics_{};
  uint64_t nexamples_;
  std::unordered_map<int32_t, Metrics> labelMetrics_;
};

} // namespace fasttext
