/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_MATRIX_H
#define FASTTEXT_MATRIX_H

#include <cstdint>
#include <istream>
#include <ostream>

#include "real.h"

namespace fasttext {

class Vector;

class Matrix {

  public:
    real* data_;
    int64_t m_;
    int64_t n_;

    Matrix();
    Matrix(int64_t, int64_t);
    Matrix(const Matrix&);
    Matrix& operator=(const Matrix&);
    ~Matrix();

    void zero();
    void uniform(real);
    real dotRow(const Vector&, int64_t);
    void addRow(const Vector&, int64_t, real);

    void save(std::ostream&);
    void load(std::istream&);
};

}

#endif
