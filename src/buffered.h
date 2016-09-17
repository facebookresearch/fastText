/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_BUFFERED_H
#define FASTTEXT_BUFFERED_H

#include <array>
#include <fstream>
#include <memory>

// istream wrapper for efficient char by char reading
class Buffered {
  private:
    static const int32_t BUF_SIZE = 512;

    std::array<char, BUF_SIZE> buf_;
    std::unique_ptr<std::istream> is_;
    int32_t cursor_;
    int32_t buf_last_;

    void fetch();

  public:
    Buffered(std::unique_ptr<std::istream>&&);

    bool eof() const;

    // if stream at eof, leave `c` unchanged and return false
    // otherwise, assign the current character to `c` and return true
    bool peek(char& c) const;

    // seek one character forward
    void advance();

    // clear flags and seek to the beginning
    void reset();
};

#endif
