/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "args.h"

#include <cstdlib>
#include <cstdint>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fasttext {

Args::Args() {
  lr = 0.05;
  dim = 100;
  ws = 5;
  epoch = 5;
  minCount = 5;
  minCountLabel = 0;
  neg = 5;
  wordNgrams = 1;
  loss = loss_name::ns;
  model = model_name::sg;
  bucket = 2000000;
  minn = 3;
  maxn = 6;
  thread = 12;
  lrUpdateRate = 100;
  t = 1e-4;
  label = "__label__";
  verbose = 2;
  pretrainedVectors = "";
  saveOutput = false;
  seed = 0;

  qout = false;
  retrain = false;
  qnorm = false;
  cutoff = 0;
  dsub = 2;

  autotuneValidationFile = "";
  autotuneMetric = "f1";
  autotunePredictions = 1;
  autotuneDuration = 60 * 5; // 5 minutes
  autotuneModelSize = "";
}

std::string Args::lossToString(loss_name ln) const {
  switch (ln) {
    case loss_name::hs:
      return "hs";
    case loss_name::ns:
      return "ns";
    case loss_name::softmax:
      return "softmax";
    case loss_name::ova:
      return "one-vs-all";
  }
  return "Unknown loss!"; // should never happen
}

std::string Args::boolToString(bool b) const {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

std::string Args::modelToString(model_name mn) const {
  switch (mn) {
    case model_name::cbow:
      return "cbow";
    case model_name::sg:
      return "sg";
    case model_name::sup:
      return "sup";
  }
  return "Unknown model name!"; // should never happen
}

std::string Args::metricToString(metric_name mn) const {
  switch (mn) {
    case metric_name::f1score:
      return "f1score";
    case metric_name::f1scoreLabel:
      return "f1scoreLabel";
    case metric_name::precisionAtRecall:
      return "precisionAtRecall";
    case metric_name::precisionAtRecallLabel:
      return "precisionAtRecallLabel";
    case metric_name::recallAtPrecision:
      return "recallAtPrecision";
    case metric_name::recallAtPrecisionLabel:
      return "recallAtPrecisionLabel";
  }
  return "Unknown metric name!"; // should never happen
}

void Args::parseArgs(const std::vector<std::string>& args) {
  const std::string& command(args[1]);
  if (command == "supervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    minCount = 1;
    minn = 0;
    maxn = 0;
    lr = 0.1;
  } else if (command == "cbow") {
    model = model_name::cbow;
  }
  for (int ai = 2; ai < args.size(); ai += 2) {
    if (args[ai][0] != '-') {
      std::cerr << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    try {
      setManual(args[ai].substr(1));

      if (args[ai] == "-h") {
        std::cerr << "Here is the help! Usage:" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      } else if (args[ai] == "-input") {
        input = std::string(args.at(ai + 1));
      } else if (args[ai] == "-output") {
        output = std::string(args.at(ai + 1));
      } else if (args[ai] == "-lr") {
        lr = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-lrUpdateRate") {
        lrUpdateRate = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dim") {
        dim = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-ws") {
        ws = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-epoch") {
        epoch = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCount") {
        minCount = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCountLabel") {
        minCountLabel = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-neg") {
        neg = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-wordNgrams") {
        wordNgrams = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-loss") {
        if (args.at(ai + 1) == "hs") {
          loss = loss_name::hs;
        } else if (args.at(ai + 1) == "ns") {
          loss = loss_name::ns;
        } else if (args.at(ai + 1) == "softmax") {
          loss = loss_name::softmax;
        } else if (
            args.at(ai + 1) == "one-vs-all" || args.at(ai + 1) == "ova") {
          loss = loss_name::ova;
        } else {
          std::cerr << "Unknown loss: " << args.at(ai + 1) << std::endl;
          printHelp();
          exit(EXIT_FAILURE);
        }
      } else if (args[ai] == "-bucket") {
        bucket = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minn") {
        minn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxn") {
        maxn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-thread") {
        thread = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-t") {
        t = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-label") {
        label = std::string(args.at(ai + 1));
      } else if (args[ai] == "-verbose") {
        verbose = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-pretrainedVectors") {
        pretrainedVectors = std::string(args.at(ai + 1));
      } else if (args[ai] == "-saveOutput") {
        saveOutput = true;
        ai--;
      } else if (args[ai] == "-seed") {
        seed = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-qnorm") {
        qnorm = true;
        ai--;
      } else if (args[ai] == "-retrain") {
        retrain = true;
        ai--;
      } else if (args[ai] == "-qout") {
        qout = true;
        ai--;
      } else if (args[ai] == "-cutoff") {
        cutoff = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dsub") {
        dsub = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-autotune-validation") {
        autotuneValidationFile = std::string(args.at(ai + 1));
      } else if (args[ai] == "-autotune-metric") {
        autotuneMetric = std::string(args.at(ai + 1));
        getAutotuneMetric(); // throws exception if not able to parse
        getAutotuneMetricLabel(); // throws exception if not able to parse
      } else if (args[ai] == "-autotune-predictions") {
        autotunePredictions = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-autotune-duration") {
        autotuneDuration = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-autotune-modelsize") {
        autotuneModelSize = std::string(args.at(ai + 1));
      } else {
        std::cerr << "Unknown argument: " << args[ai] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } catch (std::out_of_range) {
      std::cerr << args[ai] << " is missing an argument" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
  }
  if (input.empty() || output.empty()) {
    std::cerr << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  if (wordNgrams <= 1 && maxn == 0 && !hasAutotune()) {
    bucket = 0;
  }
}

void Args::printHelp() {
  printBasicHelp();
  printDictionaryHelp();
  printTrainingHelp();
  printAutotuneHelp();
  printQuantizationHelp();
}

void Args::printBasicHelp() {
  std::cerr << "\nThe following arguments are mandatory:\n"
            << "  -input              training file path\n"
            << "  -output             output file path\n"
            << "\nThe following arguments are optional:\n"
            << "  -verbose            verbosity level [" << verbose << "]\n";
}

void Args::printDictionaryHelp() {
  std::cerr << "\nThe following arguments for the dictionary are optional:\n"
            << "  -minCount           minimal number of word occurences ["
            << minCount << "]\n"
            << "  -minCountLabel      minimal number of label occurences ["
            << minCountLabel << "]\n"
            << "  -wordNgrams         max length of word ngram [" << wordNgrams
            << "]\n"
            << "  -bucket             number of buckets [" << bucket << "]\n"
            << "  -minn               min length of char ngram [" << minn
            << "]\n"
            << "  -maxn               max length of char ngram [" << maxn
            << "]\n"
            << "  -t                  sampling threshold [" << t << "]\n"
            << "  -label              labels prefix [" << label << "]\n";
}

void Args::printTrainingHelp() {
  std::cerr
      << "\nThe following arguments for training are optional:\n"
      << "  -lr                 learning rate [" << lr << "]\n"
      << "  -lrUpdateRate       change the rate of updates for the learning "
         "rate ["
      << lrUpdateRate << "]\n"
      << "  -dim                size of word vectors [" << dim << "]\n"
      << "  -ws                 size of the context window [" << ws << "]\n"
      << "  -epoch              number of epochs [" << epoch << "]\n"
      << "  -neg                number of negatives sampled [" << neg << "]\n"
      << "  -loss               loss function {ns, hs, softmax, one-vs-all} ["
      << lossToString(loss) << "]\n"
      << "  -thread             number of threads (set to 1 to ensure "
         "reproducible results) ["
      << thread << "]\n"
      << "  -pretrainedVectors  pretrained word vectors for supervised "
         "learning ["
      << pretrainedVectors << "]\n"
      << "  -saveOutput         whether output params should be saved ["
      << boolToString(saveOutput) << "]\n"
      << "  -seed               random generator seed  [" << seed << "]\n";
}

void Args::printAutotuneHelp() {
  std::cerr << "\nThe following arguments are for autotune:\n"
            << "  -autotune-validation            validation file to be used "
               "for evaluation\n"
            << "  -autotune-metric                metric objective {f1, "
               "f1:labelname} ["
            << autotuneMetric << "]\n"
            << "  -autotune-predictions           number of predictions used "
               "for evaluation  ["
            << autotunePredictions << "]\n"
            << "  -autotune-duration              maximum duration in seconds ["
            << autotuneDuration << "]\n"
            << "  -autotune-modelsize             constraint model file size ["
            << autotuneModelSize << "] (empty = do not quantize)\n";
}

void Args::printQuantizationHelp() {
  std::cerr
      << "\nThe following arguments for quantization are optional:\n"
      << "  -cutoff             number of words and ngrams to retain ["
      << cutoff << "]\n"
      << "  -retrain            whether embeddings are finetuned if a cutoff "
         "is applied ["
      << boolToString(retrain) << "]\n"
      << "  -qnorm              whether the norm is quantized separately ["
      << boolToString(qnorm) << "]\n"
      << "  -qout               whether the classifier is quantized ["
      << boolToString(qout) << "]\n"
      << "  -dsub               size of each sub-vector [" << dsub << "]\n";
}

void Args::save(std::ostream& out) {
  out.write((char*)&(dim), sizeof(int));
  out.write((char*)&(ws), sizeof(int));
  out.write((char*)&(epoch), sizeof(int));
  out.write((char*)&(minCount), sizeof(int));
  out.write((char*)&(neg), sizeof(int));
  out.write((char*)&(wordNgrams), sizeof(int));
  out.write((char*)&(loss), sizeof(loss_name));
  out.write((char*)&(model), sizeof(model_name));
  out.write((char*)&(bucket), sizeof(int));
  out.write((char*)&(minn), sizeof(int));
  out.write((char*)&(maxn), sizeof(int));
  out.write((char*)&(lrUpdateRate), sizeof(int));
  out.write((char*)&(t), sizeof(double));
}

void Args::load(std::istream& in) {
  in.read((char*)&(dim), sizeof(int));
  in.read((char*)&(ws), sizeof(int));
  in.read((char*)&(epoch), sizeof(int));
  in.read((char*)&(minCount), sizeof(int));
  in.read((char*)&(neg), sizeof(int));
  in.read((char*)&(wordNgrams), sizeof(int));
  in.read((char*)&(loss), sizeof(loss_name));
  in.read((char*)&(model), sizeof(model_name));
  in.read((char*)&(bucket), sizeof(int));
  in.read((char*)&(minn), sizeof(int));
  in.read((char*)&(maxn), sizeof(int));
  in.read((char*)&(lrUpdateRate), sizeof(int));
  in.read((char*)&(t), sizeof(double));
}

void Args::dump(std::ostream& out) const {
  out << "dim"
      << " " << dim << std::endl;
  out << "ws"
      << " " << ws << std::endl;
  out << "epoch"
      << " " << epoch << std::endl;
  out << "minCount"
      << " " << minCount << std::endl;
  out << "neg"
      << " " << neg << std::endl;
  out << "wordNgrams"
      << " " << wordNgrams << std::endl;
  out << "loss"
      << " " << lossToString(loss) << std::endl;
  out << "model"
      << " " << modelToString(model) << std::endl;
  out << "bucket"
      << " " << bucket << std::endl;
  out << "minn"
      << " " << minn << std::endl;
  out << "maxn"
      << " " << maxn << std::endl;
  out << "lrUpdateRate"
      << " " << lrUpdateRate << std::endl;
  out << "t"
      << " " << t << std::endl;
}

bool Args::hasAutotune() const {
  return !autotuneValidationFile.empty();
}

bool Args::isManual(const std::string& argName) const {
  return (manualArgs_.count(argName) != 0);
}

void Args::setManual(const std::string& argName) {
  manualArgs_.emplace(argName);
}

metric_name Args::getAutotuneMetric() const {
  if (autotuneMetric.substr(0, 3) == "f1:") {
    return metric_name::f1scoreLabel;
  } else if (autotuneMetric == "f1") {
    return metric_name::f1score;
  } else if (autotuneMetric.substr(0, 18) == "precisionAtRecall:") {
    size_t semicolon = autotuneMetric.find(':', 18);
    if (semicolon != std::string::npos) {
      return metric_name::precisionAtRecallLabel;
    }
    return metric_name::precisionAtRecall;
  } else if (autotuneMetric.substr(0, 18) == "recallAtPrecision:") {
    size_t semicolon = autotuneMetric.find(':', 18);
    if (semicolon != std::string::npos) {
      return metric_name::recallAtPrecisionLabel;
    }
    return metric_name::recallAtPrecision;
  }
  throw std::runtime_error("Unknown metric : " + autotuneMetric);
}

std::string Args::getAutotuneMetricLabel() const {
  metric_name metric = getAutotuneMetric();
  std::string label;
  if (metric == metric_name::f1scoreLabel) {
    label = autotuneMetric.substr(3);
  } else if (
      metric == metric_name::precisionAtRecallLabel ||
      metric == metric_name::recallAtPrecisionLabel) {
    size_t semicolon = autotuneMetric.find(':', 18);
    label = autotuneMetric.substr(semicolon + 1);
  } else {
    return label;
  }

  if (label.empty()) {
    throw std::runtime_error("Empty metric label : " + autotuneMetric);
  }
  return label;
}

double Args::getAutotuneMetricValue() const {
  metric_name metric = getAutotuneMetric();
  double value = 0.0;
  if (metric == metric_name::precisionAtRecallLabel ||
      metric == metric_name::precisionAtRecall ||
      metric == metric_name::recallAtPrecisionLabel ||
      metric == metric_name::recallAtPrecision) {
    size_t firstSemicolon = 18; // semicolon position in "precisionAtRecall:"
    size_t secondSemicolon = autotuneMetric.find(':', firstSemicolon);
    const std::string valueStr =
        autotuneMetric.substr(firstSemicolon, secondSemicolon - firstSemicolon);
    value = std::stof(valueStr) / 100.0;
  }
  return value;
}

int64_t Args::getAutotuneModelSize() const {
  std::string modelSize = autotuneModelSize;
  if (modelSize.empty()) {
    return Args::kUnlimitedModelSize;
  }
  std::unordered_map<char, int> units = {
      {'k', 1000},
      {'K', 1000},
      {'m', 1000000},
      {'M', 1000000},
      {'g', 1000000000},
      {'G', 1000000000},
  };
  uint64_t multiplier = 1;
  char lastCharacter = modelSize.back();
  if (units.count(lastCharacter)) {
    multiplier = units[lastCharacter];
    modelSize = modelSize.substr(0, modelSize.size() - 1);
  }
  uint64_t size = 0;
  size_t nonNumericCharacter = 0;
  bool parseError = false;
  try {
    size = std::stol(modelSize, &nonNumericCharacter);
  } catch (std::invalid_argument&) {
    parseError = true;
  }
  if (!parseError && nonNumericCharacter != modelSize.size()) {
    parseError = true;
  }
  if (parseError) {
    throw std::invalid_argument(
        "Unable to parse model size " + autotuneModelSize);
  }

  return size * multiplier;
}

} // namespace fasttext
