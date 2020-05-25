/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"

#include <iomanip>
#include <ios>

namespace fasttext {

namespace utils {

int64_t size(std::ifstream& ifs) {
  ifs.seekg(std::streamoff(0), std::ios::end);
  return ifs.tellg();
}

void seek(std::ifstream& ifs, int64_t pos) {
  ifs.clear();
  ifs.seekg(std::streampos(pos));
}

double getDuration(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

ClockPrint::ClockPrint(int32_t duration) : duration_(duration) {}

std::ostream& operator<<(std::ostream& out, const ClockPrint& me) {
  int32_t etah = me.duration_ / 3600;
  int32_t etam = (me.duration_ % 3600) / 60;
  int32_t etas = (me.duration_ % 3600) % 60;

  out << std::setw(3) << etah << "h" << std::setw(2) << etam << "m";
  out << std::setw(2) << etas << "s";
  return out;
}

bool compareFirstLess(const std::pair<double, double>& l, const double& r) {
  return l.first < r;
}

} // namespace utils

} // namespace fasttext
