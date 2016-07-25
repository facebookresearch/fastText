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
#include <fenv.h>

Args args;

void loadModel(std::string path, Dictionary& dict,
               Matrix& input, Matrix& output) {
  std::ifstream ifs(path);
  args.load(ifs);
  dict.load(ifs);
  input.load(ifs);
  output.load(ifs);
  ifs.close();
}

void test(Dictionary& dict, Matrix& input, Matrix& output, std::string fname) {
  Model model(input, output, args.dim, args.lr, 1);
  model.setLabelFreq(dict.getLabelFreq());
  int32_t N = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;
  std::wifstream ifs(fname);
  while (!ifs.eof()) {
    dict.getLine(ifs, line, labels, model.rng);
    dict.addNgrams(line, args.wordNgrams);
    if (labels.size() > 0 && line.size() > 0) {
      int32_t i = model.predict(line);
      for (auto& t : labels) {
        if (i == t) {
          precision += 1.0;
          break;
        }
      }
      N++;
    }
  }
  ifs.close();
  std::wcout << std::setprecision(3) << "P@1: " << precision / N << std::endl;
  std::wcout << std::setprecision(3) << "Sentences: " << N << std::endl;
}

int main(int argc, char** argv) {
  std::locale::global(std::locale(""));
  utils::initTables();

  std::string modelPath(argv[1]);
  std::string testPath(argv[2]);

  Dictionary dict;
  Matrix input;
  Matrix output;
  loadModel(modelPath, dict, input, output);
  test(dict, input, output, testPath);

  utils::freeTables();

  return 0;
}
