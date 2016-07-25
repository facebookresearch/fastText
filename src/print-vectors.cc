/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>
#include <string>
#include <vector>
#include "args.h"
#include "matrix.h"
#include "vector.h"
#include "dictionary.h"

Args args;

void getEmbeddings(Dictionary& dict, Matrix& input, Matrix& output) {
  std::wstring ws;
  while (std::wcin >> ws) {
    std::wcout << ws << " ";
    Vector embedding(args.dim);
    embedding.zero();
    std::vector<int32_t> ngrams = dict.getNgrams(ws);
    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
      embedding.addRow(input, *it);
    }
    embedding.mul(1.0 / ngrams.size());
    embedding.writeToStream(std::wcout);
    std::wcout << std::endl;
  }
}

void loadModel(std::string path, Dictionary& dict, Matrix& input,
                Matrix& output) {
  std::ifstream ifs(path);
  if (ifs.good()) {
    args.load(ifs);
    dict.load(ifs);
    input.load(ifs);
    output.load(ifs);
  } else {
    std::wcout << "Invalid model name!" << std::endl;
    exit(EXIT_FAILURE);
  }
  ifs.close();
}

int main(int argc, char** argv) {
  std::locale::global(std::locale(""));
  std::cout << argc << std::endl;
  if (argc != 2) {
    std::wcout << "Please provide path to .bin file" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string modelPath(argv[1]);

  Dictionary dict;
  Matrix input;
  Matrix output;
  loadModel(modelPath, dict, input, output);

  std::wcout << dict.getNumWords() << std::endl;

  getEmbeddings(dict, input, output);

  return 0;
}
