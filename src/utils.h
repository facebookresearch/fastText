/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_UTILS_H
#define FASTTEXT_UTILS_H

#include <fstream>

#include "real.h"

#define SIGMOID_TABLE_SIZE 512
#define MAX_SIGMOID 8
#define LOG_TABLE_SIZE 512

namespace utils {

  real log(real);
  real sigmoid(real);

  void initTables();
  void initSigmoid();
  void initLog();
  void freeTables();

  int64_t size(std::ifstream&);
  void seek(std::ifstream&, int64_t);
}

#endif
