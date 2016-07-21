/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <fstream>
#include "Real.h"

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
    real dotRow(int64_t, const Vector&);
    void addRow(int64_t, real, const Vector&);

    void save(std::ofstream&);
    void load(std::ifstream&);
};

#endif
