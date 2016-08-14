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
#include <vector>

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

  template<typename T>
  struct OneOrMorePOD {
    enum tag_t { single, multi } tag;
    union {
      T datum;
      std::vector<T> data;
    };
    OneOrMorePOD(tag_t tag): tag(tag) {
      if (tag == single) {
        new (&datum) T();
      } else {
        new (&data) std::vector<T>();
      }
    }
    OneOrMorePOD(OneOrMorePOD&& other): tag(other.tag) {
      if (tag == single) {
        new (&datum) T(other.datum);
      } else {
        new (&data) std::vector<T>(std::move(other.data));
      }
    }
    OneOrMorePOD(const T& o): tag(single), datum(o) {
    }
    OneOrMorePOD(std::vector<T>&& v): tag(multi), data(std::move(v)) {
    }
    ~OneOrMorePOD() {
      if (tag == multi) {
        data.~vector();
      }
    }
  };
}

#endif
