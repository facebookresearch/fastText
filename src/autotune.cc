/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "autotune.h"

#include <algorithm>
#include <csignal>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

#define LOG_VAL(name, val)                        \
  if (autotuneArgs.verbose > 2) {                 \
    std::cout << #name " = " << val << std::endl; \
  }
#define LOG_VAL_NAN(name, val)                      \
  if (autotuneArgs.verbose > 2) {                   \
    if (std::isnan(val)) {                          \
      std::cout << #name " = NaN" << std::endl;     \
    } else {                                        \
      std::cout << #name " = " << val << std::endl; \
    }                                               \
  }

namespace {

std::function<void()> interruptSignalHandler;

void signalHandler(int signal) {
  if (signal == SIGINT) {
    interruptSignalHandler();
  }
}

class ElapsedTimeMarker {
  std::chrono::steady_clock::time_point start_;

 public:
  ElapsedTimeMarker() {
    start_ = std::chrono::steady_clock::now();
  }
  double getElapsed() {
    return fasttext::utils::getDuration(
        start_, std::chrono::steady_clock::now());
  }
};

} // namespace

namespace fasttext {

constexpr double kUnknownBestScore = -1.0;
constexpr int kCutoffLimit = 256;

template <typename T>
T getArgGauss(
    T val,
    std::minstd_rand& rng,
    double startSigma,
    double endSigma,
    double t,
    bool linear) {
  T returnValue;
  const double stddev = startSigma -
      ((startSigma - endSigma) / 0.5) *
          std::min(0.5, std::max((t - 0.25), 0.0));

  std::normal_distribution<double> normal(0.0, stddev);

  const double coeff = normal(rng);
  double updateCoeff = 0.0;

  if (linear) {
    updateCoeff = coeff;
    returnValue = static_cast<T>(updateCoeff + val);
  } else {
    updateCoeff = std::pow(2.0, coeff);
    returnValue = static_cast<T>(updateCoeff * val);
  }

  return returnValue;
}

template <typename T>
T updateArgGauss(
    T val,
    T min,
    T max,
    double startSigma,
    double endSigma,
    double t,
    bool linear,
    std::minstd_rand& rng) {
  T retVal = getArgGauss(val, rng, startSigma, endSigma, t, linear);
  if (retVal > max) {
    retVal = max;
  }
  if (retVal < min) {
    retVal = min;
  }
  return retVal;
}

AutotuneStrategy::AutotuneStrategy(
    const Args& originalArgs,
    std::minstd_rand::result_type seed)
    : bestArgs_(),
      maxDuration_(originalArgs.autotuneDuration),
      rng_(seed),
      trials_(0),
      bestMinnIndex_(0),
      bestDsubExponent_(1),
      bestNonzeroBucket_(2000000),
      originalBucket_(originalArgs.bucket) {
  minnChoices_ = {0, 2, 3};
  updateBest(originalArgs);
}

Args AutotuneStrategy::ask(double elapsed) {
  const double t = std::min(1.0, elapsed / maxDuration_);
  trials_++;

  if (trials_ == 1) {
    return bestArgs_;
  }

  Args args = bestArgs_;

  if (!args.isManual("epoch")) {
    args.epoch = updateArgGauss(args.epoch, 1, 100, 2.8, 2.5, t, false, rng_);
  }
  if (!args.isManual("lr")) {
    args.lr = updateArgGauss(args.lr, 0.01, 5.0, 1.9, 1.0, t, false, rng_);
  };
  if (!args.isManual("dim")) {
    args.dim = updateArgGauss(args.dim, 1, 1000, 1.4, 0.3, t, false, rng_);
  }
  if (!args.isManual("wordNgrams")) {
    args.wordNgrams =
        updateArgGauss(args.wordNgrams, 1, 5, 4.3, 2.4, t, true, rng_);
  }
  if (!args.isManual("dsub")) {
    int dsubExponent =
        updateArgGauss(bestDsubExponent_, 1, 4, 2.0, 1.0, t, true, rng_);
    args.dsub = (1 << dsubExponent);
  }
  if (!args.isManual("minn")) {
    int minnIndex = updateArgGauss(
        bestMinnIndex_,
        0,
        static_cast<int>(minnChoices_.size() - 1),
        4.0,
        1.4,
        t,
        true,
        rng_);
    args.minn = minnChoices_[minnIndex];
  }
  if (!args.isManual("maxn")) {
    if (args.minn == 0) {
      args.maxn = 0;
    } else {
      args.maxn = args.minn + 3;
    }
  }
  if (!args.isManual("bucket")) {
    int nonZeroBucket = updateArgGauss(
        bestNonzeroBucket_, 10000, 10000000, 2.0, 1.5, t, false, rng_);
    args.bucket = nonZeroBucket;
  } else {
    args.bucket = originalBucket_;
  }
  if (args.wordNgrams <= 1 && args.maxn == 0) {
    args.bucket = 0;
  }
  if (!args.isManual("loss")) {
    args.loss = loss_name::softmax;
  }

  return args;
}

int AutotuneStrategy::getIndex(int val, const std::vector<int>& choices) {
  auto found = std::find(choices.begin(), choices.end(), val);
  int ind = 0;
  if (found != choices.end()) {
    ind = std::distance(choices.begin(), found);
  }
  return ind;
}

void AutotuneStrategy::updateBest(const Args& args) {
  bestArgs_ = args;
  bestMinnIndex_ = getIndex(args.minn, minnChoices_);
  bestDsubExponent_ = log2(args.dsub);
  if (args.bucket != 0) {
    bestNonzeroBucket_ = args.bucket;
  }
}

Autotune::Autotune(const std::shared_ptr<FastText>& fastText)
    : fastText_(fastText),
      elapsed_(0.),
      bestScore_(0.),
      trials_(0),
      sizeConstraintFailed_(0),
      continueTraining_(false),
      strategy_(),
      timer_() {}

void Autotune::printInfo(double maxDuration) {
  double progress = elapsed_ * 100 / maxDuration;
  progress = std::min(progress, 100.0);

  std::cerr << "\r";
  std::cerr << std::fixed;
  std::cerr << "Progress: ";
  std::cerr << std::setprecision(1) << std::setw(5) << progress << "%";
  std::cerr << " Trials: " << std::setw(4) << trials_;
  std::cerr << " Best score: " << std::setw(9) << std::setprecision(6);
  if (bestScore_ == kUnknownBestScore) {
    std::cerr << "unknown";
  } else {
    std::cerr << bestScore_;
  }
  std::cerr << " ETA: "
            << utils::ClockPrint(std::max(maxDuration - elapsed_, 0.0));
  std::cerr << std::flush;
}

void Autotune::timer(
    const std::chrono::steady_clock::time_point& start,
    double maxDuration) {
  elapsed_ = 0.0;
  while (keepTraining(maxDuration)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    elapsed_ = utils::getDuration(start, std::chrono::steady_clock::now());
    printInfo(maxDuration);
  }
  abort();
}

bool Autotune::keepTraining(double maxDuration) const {
  return continueTraining_ && elapsed_ < maxDuration;
}

void Autotune::abort() {
  if (continueTraining_) {
    continueTraining_ = false;
    fastText_->abort();
  }
}

void Autotune::startTimer(const Args& args) {
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  timer_ = std::thread([=]() { timer(start, args.autotuneDuration); });
  bestScore_ = kUnknownBestScore;
  trials_ = 0;
  continueTraining_ = true;

  auto previousSignalHandler = std::signal(SIGINT, signalHandler);
  interruptSignalHandler = [&]() {
    std::signal(SIGINT, previousSignalHandler);
    std::cerr << std::endl << "Aborting autotune..." << std::endl;
    abort();
  };
}

double Autotune::getMetricScore(
    Meter& meter,
    const metric_name& metricName,
    const std::string& metricLabel) const {
  double score = 0.0;
  if (metricName == metric_name::f1score) {
    score = meter.f1Score();
  } else if (metricName == metric_name::labelf1score) {
    int32_t labelId = fastText_->getDictionary()->getId(metricLabel);
    if (labelId == -1) {
      throw std::runtime_error("Unknown autotune metric label");
    }
    labelId = labelId - fastText_->getDictionary()->nwords();
    score = meter.f1Score(labelId);
  } else {
    throw std::runtime_error("Unknown metric");
  }
  return score;
}

void Autotune::printArgs(const Args& args, const Args& autotuneArgs) {
  LOG_VAL(epoch, args.epoch)
  LOG_VAL(lr, args.lr)
  LOG_VAL(dim, args.dim)
  LOG_VAL(minCount, args.minCount)
  LOG_VAL(wordNgrams, args.wordNgrams)
  LOG_VAL(minn, args.minn)
  LOG_VAL(maxn, args.maxn)
  LOG_VAL(bucket, args.bucket)
  LOG_VAL(dsub, args.dsub)
  LOG_VAL(loss, args.lossToString(args.loss))
}

int Autotune::getCutoffForFileSize(
    bool qout,
    bool qnorm,
    int dsub,
    int64_t fileSize) const {
  int64_t outModelSize = 0;
  const int64_t outM = fastText_->getOutputMatrix()->size(0);
  const int64_t outN = fastText_->getOutputMatrix()->size(1);
  if (qout) {
    const int64_t outputPqSize = 16 + 4 * (outN * (1 << 8));
    outModelSize =
        21 + (outM * ((outN + 2 - 1) / 2)) + outputPqSize + (qnorm ? outM : 0);
  } else {
    outModelSize = 16 + 4 * (outM * outN);
  }
  const int64_t dim = fastText_->getInputMatrix()->size(1);

  int target = (fileSize - (107) - 4 * (1 << 8) * dim - outModelSize);
  int cutoff = target / ((dim + dsub - 1) / dsub + (qnorm ? 1 : 0) + 10);

  return std::max(cutoff, kCutoffLimit);
}

bool Autotune::quantize(Args& args, const Args& autotuneArgs) {
  if (autotuneArgs.getAutotuneModelSize() == Args::kUnlimitedModelSize) {
    return true;
  }
  auto outputSize = fastText_->getOutputMatrix()->size(0);

  args.qnorm = true;
  args.qout = (outputSize >= kCutoffLimit);
  args.retrain = true;
  args.cutoff = getCutoffForFileSize(
      args.qout, args.qnorm, args.dsub, autotuneArgs.getAutotuneModelSize());
  LOG_VAL(cutoff, args.cutoff);
  if (args.cutoff == kCutoffLimit) {
    return false;
  }
  fastText_->quantize(args);

  return true;
}

void Autotune::printSkippedArgs(const Args& autotuneArgs) {
  std::unordered_set<std::string> argsToCheck = {"epoch",
                                                 "lr",
                                                 "dim",
                                                 "wordNgrams",
                                                 "loss",
                                                 "bucket",
                                                 "minn",
                                                 "maxn",
                                                 "dsub"};
  for (const auto& arg : argsToCheck) {
    if (autotuneArgs.isManual(arg)) {
      std::cerr << "Warning : " << arg
                << " is manually set to a specific value. "
                << "It will not be automatically optimized." << std::endl;
    }
  }
}

void Autotune::train(const Args& autotuneArgs) {
  std::ifstream validationFileStream(autotuneArgs.autotuneValidationFile);
  if (!validationFileStream.is_open()) {
    throw std::invalid_argument("Validation file cannot be opened!");
  }
  printSkippedArgs(autotuneArgs);

  bool sizeConstraintWarning = false;
  int verbose = autotuneArgs.verbose;
  Args bestTrainArgs(autotuneArgs);
  Args trainArgs(autotuneArgs);
  trainArgs.verbose = 0;
  strategy_ = std::unique_ptr<AutotuneStrategy>(
      new AutotuneStrategy(trainArgs, autotuneArgs.seed));
  startTimer(autotuneArgs);

  while (keepTraining(autotuneArgs.autotuneDuration)) {
    trials_++;

    trainArgs = strategy_->ask(elapsed_);
    LOG_VAL(Trial, trials_)
    printArgs(trainArgs, autotuneArgs);
    ElapsedTimeMarker elapsedTimeMarker;
    double currentScore = std::numeric_limits<double>::quiet_NaN();
    try {
      fastText_->train(trainArgs);
      bool sizeConstraintOK = quantize(trainArgs, autotuneArgs);
      if (sizeConstraintOK) {
        Meter meter;
        fastText_->test(
            validationFileStream, autotuneArgs.autotunePredictions, 0.0, meter);

        currentScore = getMetricScore(
            meter,
            autotuneArgs.getAutotuneMetric(),
            autotuneArgs.getAutotuneMetricLabel());

        if (bestScore_ == kUnknownBestScore ||
            (currentScore > bestScore_)) {
          bestTrainArgs = trainArgs;
          bestScore_ = currentScore;
          strategy_->updateBest(bestTrainArgs);
        }
      } else {
        sizeConstraintFailed_++;
        if (!sizeConstraintWarning && trials_ > 10 &&
            sizeConstraintFailed_ > (trials_ / 2)) {
          sizeConstraintWarning = true;
          std::cerr
              << std::endl
              << "Warning : requested model size is probably too small. You may want to increase `autotune-modelsize`."
              << std::endl;
        }
      }
    } catch (DenseMatrix::EncounteredNaNError&) {
      // ignore diverging loss and go on
    } catch (std::bad_alloc&) {
      // ignore parameter samples asking too much memory
    } catch (TimeoutError&) {
      break;
    } catch (FastText::AbortError&) {
      break;
    }
    LOG_VAL_NAN(currentScore, currentScore)
    LOG_VAL(train took, elapsedTimeMarker.getElapsed())
  }
  if (timer_.joinable()) {
    timer_.join();
  }

  if (bestScore_ == kUnknownBestScore) {
    std::string errorMessage;
    if (sizeConstraintWarning) {
      errorMessage =
          "Couldn't fulfil model size constraint: please increase `autotune-modelsize`.";
    } else {
      errorMessage =
          "Didn't have enough time to train once: please increase `autotune-duration`.";
    }
    throw std::runtime_error(errorMessage);
  } else {
    std::cerr << std::endl;
    std::cerr << "Training again with best arguments" << std::endl;
    bestTrainArgs.verbose = verbose;
    LOG_VAL(Best selected args, 0)
    printArgs(bestTrainArgs, autotuneArgs);
    fastText_->train(bestTrainArgs);
    quantize(bestTrainArgs, autotuneArgs);
  }
}

} // namespace fasttext
