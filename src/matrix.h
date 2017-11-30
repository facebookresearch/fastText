/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

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

    inline const real& at(int64_t i, int64_t j) const {return data_[i * n_ + j];};
    inline real& at(int64_t i, int64_t j) {return data_[i * n_ + j];};


    void zero();
    void uniform(real);
    real dotRow(const Vector&, int64_t) const;
    void addRow(const Vector&, int64_t, real);

    void multiplyRow(const Vector& nums, int64_t ib = 0, int64_t ie = -1);
    void divideRow(const Vector& denoms, int64_t ib = 0, int64_t ie = -1);

    real l2NormRow(int64_t i) const;
    void l2NormRow(Vector& norms) const;

    void save(std::ostream&);
    void load(std::istream&);
};

}
