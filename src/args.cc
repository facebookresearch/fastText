/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "args.h"
#include "stdlib.h"
#include <string.h>
#include <iostream>
#include <fstream>

Args::Args() {
  lr = 0.025;
  dim = 100;
  ws = 5;
  epoch = 5;
  minCount = 5;
  neg = 5;
  wordNgrams = 0;
  sampling = sampling_name::sqrt;
  loss = loss_name::ns;
  model = model_name::sg;
  bucket = 2000000;
  minn = 3;
  maxn = 6;
  onlyWord = 0;
  thread = 12;
  verbose = 1000;
  t = 1e-4;
  label = L"__label__";
}

void Args::parseArgs(int argc, char** argv) {
  if (argc == 1) {
    std::wcout << "No arguments were provided! Usage:" << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  int ai = 1;
  while (ai < argc) {
    if (argv[ai][0] != '-') {
      std::wcout << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    if (strcmp(argv[ai], "-h") == 0) {
      std::wcout << "Here is the help! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    } else if (strcmp(argv[ai], "-input") == 0) {
      input = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-test") == 0) {
      test = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-output") == 0) {
      output = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lr") == 0) {
      lr = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dim") == 0) {
      dim = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-ws") == 0) {
      ws = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-epoch") == 0) {
      epoch = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minCount") == 0) {
      minCount = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-neg") == 0) {
      neg = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-wordNgrams") == 0) {
      wordNgrams = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-sampling") == 0) {
      if (strcmp(argv[ai + 1], "sqrt") == 0) {
        sampling = sampling_name::sqrt;
      } else if (strcmp(argv[ai + 1], "log") == 0) {
        sampling = sampling_name::log;
      } else if (strcmp(argv[ai + 1], "uni") == 0) {
        sampling = sampling_name::uni;
      } else {
        std::wcout << "Invalid sampling!" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } else if (strcmp(argv[ai], "-loss") == 0) {
      if (strcmp(argv[ai + 1], "hs") == 0) {
        loss = loss_name::hs;
      } else if (strcmp(argv[ai + 1], "ns") == 0) {
        loss = loss_name::ns;
      } else if (strcmp(argv[ai + 1], "softmax") == 0) {
        loss = loss_name::softmax;
      } else {
        std::wcout << "Invalid loss!" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } else if (strcmp(argv[ai], "-bucket") == 0) {
      bucket = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minn") == 0) {
      minn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-maxn") == 0) {
      maxn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-onlyWord") == 0) {
      onlyWord = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-thread") == 0) {
      thread = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-verbose") == 0) {
      verbose = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-t") == 0) {
      t = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-model") == 0) {
      if (strcmp(argv[ai + 1], "cbow") == 0) {
        model = model_name::cbow;
      } else if (strcmp(argv[ai + 1], "sg") == 0) {
        model = model_name::sg;
      } else if (strcmp(argv[ai + 1], "sup") == 0) {
        model = model_name::sup;
      } else {
        std::wcout << "Invalid model!" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } else if (strcmp(argv[ai], "-label") == 0) {
      std::string str = std::string(argv[ai + 1]);
      label = std::wstring(str.begin(), str.end());
    } else {
      std::wcout << "Unknown argument!" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    ai += 2;
  }
  if (!checkArgs()) {
    std::wcout << "Empty input or output path!" << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
}

bool Args::checkArgs() {
  return input.length() != 0 && output.length() != 0;
}

void Args::printHelp() {
  std::wcout << "The following arguments are mandatory:" << std::endl;
  std::wcout << "\t-input:       training file path" << std::endl;
  std::wcout << "\t-output:   output file path" << std::endl;
  std::wcout << "The following arguments are optional "
              << "and have a default value:" << std::endl;
  std::wcout << "\t-lr:               learning rate, default="
              << lr << std::endl;
  std::wcout << "\t-dim:             size of the word vector, default="
              << dim << std::endl;
  std::wcout << "\t-ws:               size of the context window, default="
              << ws << std::endl;
  std::wcout << "\t-epoch:           number of epochs, default="
              << epoch << std::endl;
  std::wcout << "\t-minCount:       minimal number of word occurences, "
              << "default=" << minCount << std::endl;
  std::wcout << "\t-neg:           number of negatives sampled, default="
              << neg << std::endl;
  std::wcout << "\t-wordNgrams:       n for word ngrams to use in the "
              << "supervised setup, default=" << wordNgrams << std::endl;
  std::wcout << "\t-sampling:         sampling strategy used {sqrt, log, uni}, "
              << "default=log" << std::endl;
  std::wcout << "\t-loss:             loss function {ns, hs}, "
              << "default=ns" << std::endl;
  std::wcout << "\t-bucket:        number of ngrams used, default="
              << bucket << std::endl;
  std::wcout << "\t-minn:             length of shortest n-gram, default="
              << minn << std::endl;
  std::wcout << "\t-maxn:             length of longest n-gram, default="
              << maxn << std::endl;
  std::wcout << "\t-onlyWord:         number of words with no n-grams, "
              << "default=" << onlyWord << std::endl;
  std::wcout << "\t-thread:       number of threads, default="
              << thread << std::endl;
  std::wcout << "\t-verbose:      how often to print to stdout, default="
              << verbose << std::endl;
  std::wcout << "\t-t:                sampling threshold, default="
              << t << std::endl;
  std::wcout << "\t-model:            {sg, cbow}, default=sg" << std::endl;
  std::wcout << "\t-label:           labels prefix, default=__label__";
  std::wcout << std::endl;
}

void Args::save(std::ofstream& ofs) {
  if (ofs.is_open()) {
    ofs.write((char*) &(dim), sizeof(int));
    ofs.write((char*) &(ws), sizeof(int));
    ofs.write((char*) &(epoch), sizeof(int));
    ofs.write((char*) &(minCount), sizeof(int));
    ofs.write((char*) &(neg), sizeof(int));
    ofs.write((char*) &(wordNgrams), sizeof(int));
    ofs.write((char*) &(sampling), sizeof(sampling_name));
    ofs.write((char*) &(loss), sizeof(loss_name));
    ofs.write((char*) &(model), sizeof(model_name));
    ofs.write((char*) &(bucket), sizeof(int));
    ofs.write((char*) &(minn), sizeof(int));
    ofs.write((char*) &(maxn), sizeof(int));
    ofs.write((char*) &(onlyWord), sizeof(int));
    ofs.write((char*) &(verbose), sizeof(int));
    ofs.write((char*) &(t), sizeof(double));
  }
}

void Args::load(std::ifstream& ifs) {
  if (ifs.is_open()) {
    ifs.read((char*) &(dim), sizeof(int));
    ifs.read((char*) &(ws), sizeof(int));
    ifs.read((char*) &(epoch), sizeof(int));
    ifs.read((char*) &(minCount), sizeof(int));
    ifs.read((char*) &(neg), sizeof(int));
    ifs.read((char*) &(wordNgrams), sizeof(int));
    ifs.read((char*) &(sampling), sizeof(sampling_name));
    ifs.read((char*) &(loss), sizeof(loss_name));
    ifs.read((char*) &(model), sizeof(model_name));
    ifs.read((char*) &(bucket), sizeof(int));
    ifs.read((char*) &(minn), sizeof(int));
    ifs.read((char*) &(maxn), sizeof(int));
    ifs.read((char*) &(onlyWord), sizeof(int));
    ifs.read((char*) &(verbose), sizeof(int));
    ifs.read((char*) &(t), sizeof(double));
  }
}
