/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace fasttext {

enum class model_name : int { cbow = 1, sg, sup };
enum class loss_name : int { hs = 1, ns, softmax, ova };
enum class metric_name : int {
  f1score = 1,
  f1scoreLabel,
  precisionAtRecall,
  precisionAtRecallLabel,
  recallAtPrecision,
  recallAtPrecisionLabel
};

class Args {
 protected:
  std::string boolToString(bool) const;
  std::string modelToString(model_name) const;
  std::string metricToString(metric_name) const;
  std::unordered_set<std::string> manualArgs_;

 public:
  Args();
  std::string input;
  std::string output;
  double lr;
  int lrUpdateRate;
  int dim;
  int ws;
  int epoch;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  loss_name loss;
  model_name model;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  std::string label;
  int verbose;
  std::string pretrainedVectors;
  bool saveOutput;
  int seed;

  bool qout;
  bool retrain;
  bool qnorm;
  size_t cutoff;
  size_t dsub;

  std::string autotuneValidationFile;
  std::string autotuneMetric;
  int autotunePredictions;
  int autotuneDuration;
  std::string autotuneModelSize;

  void parseArgs(const std::vector<std::string>& args);
  void printHelp();
  void printBasicHelp();
  void printDictionaryHelp();
  void printTrainingHelp();
  void printAutotuneHelp();
  void printQuantizationHelp();
  void save(std::ostream&);
  void load(std::istream&);
  void dump(std::ostream&) const;
  bool hasAutotune() const;
  bool isManual(const std::string& argName) const;
  void setManual(const std::string& argName);
  std::string lossToString(loss_name) const;
  metric_name getAutotuneMetric() const;
  std::string getAutotuneMetricLabel() const;
  double getAutotuneMetricValue() const;
  int64_t getAutotuneModelSize() const;

  static constexpr double kUnlimitedModelSize = -1.0;
};
} // namespace fasttext
