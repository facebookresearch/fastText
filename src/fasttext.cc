/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <time.h>
#include <string>
#include <math.h>
#include <vector>
#include <atomic>
#include <algorithm>
#include <fenv.h>

Args args;

namespace info {
  clock_t start;
  std::atomic<int64_t> allWords(0);
  std::atomic<int64_t> allN(0);
  double allLoss(0.0);
}

void saveVectors(Dictionary& dict, Matrix& input, Matrix& output) {
  int32_t N = dict.getNumWords();
  std::wofstream ofs(args.output + ".vec");
  if (ofs.is_open()) {
    ofs << N << ' ' << args.dim << std::endl;
    for (int32_t i = 0; i < N; i++) {
      ofs << dict.getWord(i) << ' ';
      Vector vec(args.dim);
      vec.zero();
      const std::vector<int32_t>& ngrams = dict.getNgrams(i);
      for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
        vec.addRow(input, *it);
      }
      vec.mul(1.0 / ngrams.size());
      vec.writeToStream(ofs);
      ofs << std::endl;
    }
    ofs.close();
  } else {
    std::wcout << "Error opening file for saving vectors." << std::endl;
  }
}

void printVectors(Dictionary& dict, Matrix& input, Matrix& output) {
  std::wstring ws;
  while (std::wcin >> ws) {
    std::wcout << ws << " ";
    Vector vec(args.dim);
    vec.zero();
    std::vector<int32_t> ngrams = dict.getNgrams(ws);
    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
      vec.addRow(input, *it);
    }
    vec.mul(1.0 / ngrams.size());
    vec.writeToStream(std::wcout);
    std::wcout << std::endl;
  }
}

void saveModel(Dictionary& dict, Matrix& input, Matrix& output) {
  std::ofstream ofs(args.output + ".bin");
  args.save(ofs);
  dict.save(ofs);
  input.save(ofs);
  output.save(ofs);
  ofs.close();
}

void loadModel(std::string filename, Dictionary& dict,
               Matrix& input, Matrix& output) {
  std::ifstream ifs(filename);
  args.load(ifs);
  dict.load(ifs);
  input.load(ifs);
  output.load(ifs);
  ifs.close();
}

void printInfo(Model& model, int64_t numTokens) {
  real progress = real(info::allWords) / (args.epoch * numTokens);
  real avLoss = info::allLoss / info::allN;
  float time = float(clock() - info::start) / CLOCKS_PER_SEC;
  float wst = float(info::allWords) / time;
  int eta = int(time / progress * (1 - progress) / args.thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;

  std::wcout << std::fixed;
  std::wcout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::wcout << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::wcout << "  lr: " << std::setprecision(6) << model.getLearningRate();
  std::wcout << "  loss: " << std::setprecision(6) << avLoss;
  std::wcout << "  eta: " << etah << "h" << etam << "m  ";
  std::wcout << std::flush;
}

void supervised(Model& model,
                const std::vector<int32_t>& line,
                const std::vector<int32_t>& labels,
                double& loss, int32_t& N) {
    if (labels.size() == 0 || line.size() == 0) return;
    std::uniform_int_distribution<> uniform(0, labels.size() - 1);
    int32_t i = uniform(model.rng);
    model.update(line, labels[i], loss, N);
}

void cbow(Dictionary& dict, Model& model,
          const std::vector<int32_t>& line,
          double& loss, int32_t& N) {
  int32_t n = line.size();
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args.ws);
  for (int32_t w = 0; w < n; w++) {
    int32_t wb = uniform(model.rng);
    bow.clear();
    for (int32_t c = -wb; c <= wb; c++) {
      if (c != 0 && w + c >= 0 && w + c < n) {
        const std::vector<int32_t>& ngrams = dict.getNgrams(line[w + c]);
        for (auto it = ngrams.cbegin(); it != ngrams.cend(); ++it) {
          bow.push_back(*it);
        }
      }
    }
    model.update(bow, line[w], loss, N);
  }
}

void skipGram(Dictionary& dict, Model& model,
              const std::vector<int32_t>& line,
              double& loss, int32_t& N) {
  int32_t n = line.size();
  std::uniform_int_distribution<> uniform(1, args.ws);
  for (int32_t w = 0; w < n; w++) {
    int32_t wb = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict.getNgrams(line[w]);
    for (int32_t c = -wb; c <= wb; c++) {
      if (c != 0 && w + c >= 0 && w + c < n) {
        int32_t target = line[w + c];
        model.update(ngrams, target, loss, N);
      }
    }
  }
}

void test(Dictionary& dict, Model& model, std::string filename) {
  int32_t nexamples = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;
  std::wifstream ifs(filename);
  while (!ifs.eof()) {
    dict.getLine(ifs, line, labels, model.rng);
    dict.addNgrams(line, args.wordNgrams);
    if (labels.size() > 0 && line.size() > 0) {
      int32_t i = model.predict(line);
      if (std::find(labels.begin(), labels.end(), i) != labels.end()) {
        precision += 1.0;
      }
      nexamples++;
    }
  }
  ifs.close();
  std::wcout << std::setprecision(3);
  std::wcout << "P@1: " << precision / nexamples << std::endl;
  std::wcout << "Number of examples: " << nexamples << std::endl;
}

void trainThread(Dictionary& dict, Matrix& input, Matrix& output,
                 int32_t threadId) {
  std::wifstream ifs(args.input);
  utils::seek(ifs, threadId * utils::size(ifs) / args.thread);

  Model model(input, output, args.dim, args.lr, threadId);
  if (args.model == model_name::sup) {
    model.setLabelFreq(dict.getLabelFreq());
  } else {
    model.setLabelFreq(dict.getWordFreq());
  }

  const int64_t ntokens = dict.getNumTokens();
  int64_t tokenCount = 0;
  double loss = 0.0;
  int32_t N = 0;
  std::vector<int32_t> line, labels;

  while (info::allWords < args.epoch * ntokens) {
    tokenCount += dict.getLine(ifs, line, labels, model.rng);
    if (args.model == model_name::sup) {
      dict.addNgrams(line, args.wordNgrams);
      supervised(model, line, labels, loss, N);
    } else if (args.model == model_name::cbow) {
      cbow(dict, model, line, loss, N);
    } else if (args.model == model_name::sg) {
      skipGram(dict, model, line, loss, N);
    }

    if (tokenCount > 10000) {
      info::allWords += tokenCount;
      info::allLoss += loss;
      info::allN += N;
      tokenCount = 0;
      loss = 0.0;
      N = 0;
      real progress = real(info::allWords) / (args.epoch * ntokens);
      model.setLearningRate(args.lr * (1.0 - progress));
      if (threadId == 0) printInfo(model, ntokens);
    }
  }
  if (threadId == 0) {
    printInfo(model, ntokens);
    std::wcout << std::endl;
  }
  if (args.model == model_name::sup && threadId == 0) {
    test(dict, model, args.test);
  }
  ifs.close();
}

void printUsage() {
  std::wcout
    << "usage: fasttext <command> <args>\n\n"
    << "The commands supported by fasttext are:\n\n"
    << "  supervised       train a supervised classifier\n"
    << "  test             evaluate a supervised classifier\n"
    << "  skipgram         train a skipgram model\n"
    << "  cbow             train a cbow model\n"
    << "  print-vectors    print vectors given a trained model\n"
    << std::endl;
}

void printTestUsage() {
  std::wcout
    << "usage: fasttext test <model> <test-data>\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename\n"
    << std::endl;
}

void printPrintVectorsUsage() {
  std::wcout
    << "usage: fasttext print-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void test(int argc, char** argv) {
  if (argc != 4) {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  Dictionary dict;
  Matrix input, output;
  loadModel(std::string(argv[2]), dict, input, output);
  Model model(input, output, args.dim, args.lr, 1);
  model.setLabelFreq(dict.getLabelFreq());
  test(dict, model, std::string(argv[3]));
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
  printVectors(dict, input, output);
  exit(0);
}

void train(int argc, char** argv) {
  args.parseArgs(argc, argv);

  Dictionary dict;
  dict.readFromFile(args.input);
  Matrix input(dict.getNumWords() + args.bucket, args.dim);
  Matrix output;
  if (args.model == model_name::sup) {
    output = Matrix(dict.getNumLabels(), args.dim);
  } else {
    output = Matrix(dict.getNumWords(), args.dim);
  }
  input.uniform(1.0 / args.dim);
  output.zero();

  info::start = clock();
  std::vector<std::thread> threads;
  for (int32_t i = 0; i < args.thread; i++) {
    threads.push_back(std::thread(&trainThread, std::ref(dict),
                                  std::ref(input), std::ref(output), i));
  }
  for (auto it = threads.begin(); it != threads.end(); ++it) {
    it->join();
  }
  float trainTime = float(clock() - info::start) / CLOCKS_PER_SEC / args.thread;
  std::wcout << "training took: " << trainTime << " sec" << std::endl;

  if (args.output.size() != 0) {
    saveModel(dict, input, output);
    saveVectors(dict, input, output);
  }
}

int main(int argc, char** argv) {
  std::locale::global(std::locale(""));
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
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  utils::freeTables();
  return 0;
}
