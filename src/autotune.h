/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "args.h"
#include "fasttext.h"

namespace fasttext {

class AutotuneStrategy {
 private:
  Args bestArgs_;
  int maxDuration_;
  std::minstd_rand rng_;
  int trials_;
  int bestMinnIndex_;
  int bestDsubExponent_;
  int bestNonzeroBucket_;
  int originalBucket_;
  std::vector<int> minnChoices_;
  int getIndex(int val, const std::vector<int>& choices);

 public:
  explicit AutotuneStrategy(
      const Args& args,
      std::minstd_rand::result_type seed);
  Args ask(double elapsed);
  void updateBest(const Args& args);
};

class Autotune {
 protected:
  std::shared_ptr<FastText> fastText_;
  double elapsed_;
  double bestScore_;
  int32_t trials_;
  int32_t sizeConstraintFailed_;
  std::atomic<bool> continueTraining_;
  std::unique_ptr<AutotuneStrategy> strategy_;
  std::thread timer_;

  bool keepTraining(double maxDuration) const;
  void printInfo(double maxDuration);
  void timer(
      const std::chrono::steady_clock::time_point& start,
      double maxDuration);
  void abort();
  void startTimer(const Args& args);
  double getMetricScore(
      Meter& meter,
      const metric_name& metricName,
      const std::string& metricLabel) const;
  void printArgs(const Args& args, const Args& autotuneArgs);
  void printSkippedArgs(const Args& autotuneArgs);
  bool quantize(Args& args, const Args& autotuneArgs);
  int getCutoffForFileSize(bool qout, bool qnorm, int dsub, int64_t fileSize)
      const;

  class TimeoutError : public std::runtime_error {
   public:
    TimeoutError() : std::runtime_error("Autotune timed out.") {}
  };


 public:
  Autotune() = delete;
  explicit Autotune(const std::shared_ptr<FastText>& fastText);
  Autotune(const Autotune&) = delete;
  Autotune(Autotune&&) = delete;
  Autotune& operator=(const Autotune&) = delete;
  Autotune& operator=(Autotune&&) = delete;
  ~Autotune() noexcept = default;

  void train(const Args& args);
};

} // namespace fasttext
