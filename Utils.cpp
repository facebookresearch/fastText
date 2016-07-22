/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "Utils.h"
#include <cmath>
#include <ios>

namespace utils {
  real* t_sigmoid;
  real* t_log;

  real uniRand() {
    return real(rand()) / RAND_MAX;
  }

  real log(real x) {
    if (x > 1.0) {
      return 0.0;
    }
    int i = int(x * LOG_TABLE_SIZE);
    return t_log[i];
  }

  real sigmoid(real x) {
    if (x < -MAX_SIGMOID) {
      return 0.0;
    } else if (x > MAX_SIGMOID) {
      return 1.0;
    } else {
      int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
      return t_sigmoid[i];
    }
  }

  real trueSigmoid(real x) {
    if (x < -11.5) {
      return 0.0 + 1e-5;
    } else if (x > 11.5) {
      return 1.0 - 1e-5;
    } else {
      return 1.0 / (1.0 + exp(-x));
    }
  }

  void initTables() {
    initSigmoid();
    initLog();
  }

  void initSigmoid() {
    t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
      real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
      t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
    }
  }

  void initLog() {
    t_log = new real[LOG_TABLE_SIZE + 1];
    for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
      real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
      t_log[i] = std::log(x);
    }
  }

  void freeTables() {
    delete[] t_sigmoid;
    delete[] t_log;
  }

  int64_t size(std::wifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  void seek(std::wifstream& ifs, int64_t pos) {
    wchar_t c;
    do {
      ifs.clear();
      ifs.seekg(std::streampos(pos++));
      ifs.get(c);
    } while (!ifs.good());
    while (!iswspace(c)) {
      ifs.get(c);
    }
  }
}
