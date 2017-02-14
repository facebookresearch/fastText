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
    // This addRow enables to add part of the vector passed in parameter to the matrix.
    // So the vector can be bigger than the matrix' row length. The amount of elements
    // added to the row of the matrix are the length of the matrix' row.
    // @param from: position from where to start getting the elements in the vector.
    void addRow(const Vector&, int64_t, real, int64_t from);
    
    void save(std::ostream&);
    void load(std::istream&);
};

}

#endif
