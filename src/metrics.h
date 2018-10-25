#pragma once

#include <vector>
#include <unordered_map>

#include "real.h"
#include "dictionary.h"

namespace fasttext {

struct Metrics {
  uint64_t numExamples;
  uint64_t numLabels;
  uint64_t numPredictions;
  uint64_t numTruePositives;

  double precision() const { return numTruePositives / double(numPredictions); }
  double recall() const { return numTruePositives / double(numLabels); }
  double f1Score() const { return 2 * numTruePositives / double(numPredictions + numLabels); }
};

class MetricsAccumulator {
 public:
  virtual ~MetricsAccumulator() = default;
  virtual void log(const std::vector<int32_t>& labels, const std::vector<std::pair<real, int32_t>>& predictions);
  Metrics metrics() const { return metrics_; }

 private:
  Metrics metrics_{};
};

class LabelMetricsAccumulator : public MetricsAccumulator {
 public:
  void log(const std::vector<int32_t>& labels, const std::vector<std::pair<real, int32_t>>& predictions) override;
  void write(std::ostream&, std::shared_ptr<const Dictionary>) const;

 private:
  std::unordered_map<int32_t, Metrics> labelMetrics_;
};

}