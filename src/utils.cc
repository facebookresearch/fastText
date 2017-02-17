/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "utils.h"

#include <ios>
#include <iostream>

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream& ifs) {
    int64_t number_of_lines = 0;
    std::string line;
    while (std::getline(ifs, line))
      ++number_of_lines;
    ifs.clear();
    ifs.seekg(std::ios::beg);
    return number_of_lines;
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::ios::beg);
    std::string s;
    for(int i=1; i < pos; i++){
      std::getline(ifs, s);
    }
  }
}

}
