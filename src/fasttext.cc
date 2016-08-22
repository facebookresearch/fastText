/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <fenv.h>
#include <time.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"

Args args;

namespace info {
  clock_t start;
  std::atomic<int64_t> allWords(0);
  std::atomic<int64_t> allN(0);
  double allLoss(0.0);
}

void getVector(Dictionary& dict, Matrix& input, Vector& vec, std::string word) {
  const std::vector<int32_t>& ngrams = dict.getNgrams(word);
  vec.zero();
  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
    vec.addRow(input, *it);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void saveVectors(Dictionary& dict, Matrix& input, Matrix& output) {
  std::ofstream ofs(args.output + ".vec");
  if (!ofs.is_open()) {
    std::cout << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict.nwords() << " " << args.dim << std::endl;
  Vector vec(args.dim);
  for (int32_t i = 0; i < dict.nwords(); i++) {
    std::string word = dict.getWord(i);
    getVector(dict, input, vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void printVectors(Dictionary& dict, Matrix& input) {
  std::string word;
  Vector vec(args.dim);
  while (std::cin >> word) {
    getVector(dict, input, vec, word);
    std::cout << word << " " << vec << std::endl;
  }
}

void saveModel(Dictionary& dict, Matrix& input, Matrix& output) {
  std::ofstream ofs(args.output + ".bin");
  if (!ofs.is_open()) {
    std::cerr << "Model file cannot be opened for saving!" << std::endl;
    exit(EXIT_FAILURE);
  }
  args.save(ofs);
  dict.save(ofs);
  input.save(ofs);
  output.save(ofs);
  ofs.close();
}

void loadModel(std::string filename, Dictionary& dict,
               Matrix& input, Matrix& output) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  args.load(ifs);
  dict.load(ifs);
  input.load(ifs);
  output.load(ifs);
  ifs.close();
}

void printInfo(Model& model, real progress) {
  real loss = info::allLoss / info::allN;
  real t = real(clock() - info::start) / CLOCKS_PER_SEC;
  real wst = real(info::allWords) / t;
  int eta = int(t / progress * (1 - progress) / args.thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cout << std::fixed;
  std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cout << "  lr: " << std::setprecision(6) << model.getLearningRate();
  std::cout << "  loss: " << std::setprecision(6) << loss;
  std::cout << "  eta: " << etah << "h" << etam << "m ";
  std::cout << std::flush;
}

void supervised(Model& model,
                const std::vector<int32_t>& line,
                const std::vector<int32_t>& labels,
                double& loss, int32_t& nexamples) {
  if (labels.size() == 0 || line.size() == 0) return;
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  loss += model.update(line, labels[i]);
  nexamples++;
}

void cbow(Dictionary& dict, Model& model,
          const std::vector<int32_t>& line,
          double& loss, int32_t& nexamples) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args.ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict.getNgrams(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    loss += model.update(bow, line[w]);
    nexamples++;
  }
}

void skipgram(Dictionary& dict, Model& model,
              const std::vector<int32_t>& line,
              double& loss, int32_t& nexamples) {
  std::uniform_int_distribution<> uniform(1, args.ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict.getNgrams(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        loss += model.update(ngrams, line[w + c]);
        nexamples++;
      }
    }
  }
}

void test(Dictionary& dict, Model& model, std::string filename, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Test file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  while (ifs.peek() != EOF) {
    dict.getLine(ifs, line, labels, model.rng);
    dict.addNgrams(line, args.wordNgrams);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> predictions;
      model.predict(line, k, predictions);
      for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  ifs.close();
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << ": " << precision / nlabels << std::endl;
  std::cout << "Number of examples: " << nexamples << std::endl;
}

void predict(Dictionary& dict, Model& model, std::string filename, int32_t k) {
  std::vector<int32_t> line, labels;
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Test file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  while (ifs.peek() != EOF) {
    dict.getLine(ifs, line, labels, model.rng);
    dict.addNgrams(line, args.wordNgrams);
    if (line.empty()) {
      std::cout << "n/a" << std::endl;
      continue;
    }
    std::vector<std::pair<real, int32_t>> predictions;
    model.predict(line, k, predictions);
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << ' ';
      }
      std::cout << dict.getLabel(it->second);
    }
    std::cout << std::endl;
  }
  ifs.close();
}

void trainThread(Dictionary& dict, Matrix& input, Matrix& output,
                 int32_t threadId) {
  std::ifstream ifs(args.input);
  utils::seek(ifs, threadId * utils::size(ifs) / args.thread);

  Model model(input, output, args.dim, args.lr, threadId);
  if (args.model == model_name::sup) {
    model.setTargetCounts(dict.getCounts(entry_type::label));
  } else {
    model.setTargetCounts(dict.getCounts(entry_type::word));
  }

  real progress;
  const int64_t ntokens = dict.ntokens();
  int64_t tokenCount = 0, printCount = 0, deltaCount = 0;
  double loss = 0.0;
  int32_t nexamples = 0;
  std::vector<int32_t> line, labels;
  while (info::allWords < args.epoch * ntokens) {
    deltaCount = dict.getLine(ifs, line, labels, model.rng);
    tokenCount += deltaCount;
    printCount += deltaCount;
    if (args.model == model_name::sup) {
      dict.addNgrams(line, args.wordNgrams);
      supervised(model, line, labels, loss, nexamples);
    } else if (args.model == model_name::cbow) {
      cbow(dict, model, line, loss, nexamples);
    } else if (args.model == model_name::sg) {
      skipgram(dict, model, line, loss, nexamples);
    }
    if (tokenCount > args.lrUpdateRate) {
      info::allWords += tokenCount;
      info::allLoss += loss;
      info::allN += nexamples;
      tokenCount = 0;
      loss = 0.0;
      nexamples = 0;
      progress = real(info::allWords) / (args.epoch * ntokens);
      model.setLearningRate(args.lr * (1.0 - progress));
      if (threadId == 0) {
        printInfo(model, progress);
      }
    }
  }
  if (threadId == 0) {
    printInfo(model, 1.0);
    std::cout << std::endl;
  }
  ifs.close();
}

void printUsage() {
  std::cout
    << "usage: fasttext <command> <args>\n\n"
    << "The commands supported by fasttext are:\n\n"
    << "  supervised       train a supervised classifier\n"
    << "  test             evaluate a supervised classifier\n"
    << "  predict          predict most likely label\n"
    << "  skipgram         train a skipgram model\n"
    << "  cbow             train a cbow model\n"
    << "  print-vectors    print vectors given a trained model\n"
    << std::endl;
}

void printTestUsage() {
  std::cout
    << "usage: fasttext test <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPredictUsage() {
  std::cout
    << "usage: fasttext predict <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPrintVectorsUsage() {
  std::cout
    << "usage: fasttext print-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void test(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  Dictionary dict;
  Matrix input, output;
  loadModel(std::string(argv[2]), dict, input, output);
  Model model(input, output, args.dim, args.lr, 1);
  model.setTargetCounts(dict.getCounts(entry_type::label));
  test(dict, model, std::string(argv[3]), k);
  exit(0);
}

void predict(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  Dictionary dict;
  Matrix input, output;
  loadModel(std::string(argv[2]), dict, input, output);
  Model model(input, output, args.dim, args.lr, 1);
  model.setTargetCounts(dict.getCounts(entry_type::label));
  predict(dict, model, std::string(argv[3]), k);
  exit(0);
}

void printVectors(int argc, char** argv) {
  if (argc != 3) {
    printPrintVectorsUsage();
    exit(EXIT_FAILURE);
  }
  Dictionary dict;
  Matrix input, output;
  loadModel(std::string(argv[2]), dict, input, output);
  printVectors(dict, input);
  exit(0);
}

void train(int argc, char** argv) {
  args.parseArgs(argc, argv);

  Dictionary dict;
  std::ifstream ifs(args.input);
  if (!ifs.is_open()) {
    std::cerr << "Input file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  dict.readFromFile(ifs);
  ifs.close();

  Matrix input(dict.nwords() + args.bucket, args.dim);
  Matrix output;
  if (args.model == model_name::sup) {
    output = Matrix(dict.nlabels(), args.dim);
  } else {
    output = Matrix(dict.nwords(), args.dim);
  }
  input.uniform(1.0 / args.dim);
  output.zero();

  info::start = clock();
  time_t t0 = time(nullptr);
  std::vector<std::thread> threads;
  for (int32_t i = 0; i < args.thread; i++) {
    threads.push_back(std::thread(&trainThread, std::ref(dict),
                                  std::ref(input), std::ref(output), i));
  }
  for (auto it = threads.begin(); it != threads.end(); ++it) {
    it->join();
  }
  double trainTime = difftime(time(nullptr), t0);
  std::cout << "Train time: " << trainTime << " sec" << std::endl;

  saveModel(dict, input, output);
  if (args.model != model_name::sup) {
    saveVectors(dict, input, output);
  }
}

int main(int argc, char** argv) {
  utils::initTables();
  if (argc < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(argv[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised") {
    train(argc, argv);
  } else if (command == "test") {
    test(argc, argv);
  } else if (command == "print-vectors") {
    printVectors(argc, argv);
  } else if (command == "predict") {
    predict(argc, argv);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  utils::freeTables();
  return 0;
}
