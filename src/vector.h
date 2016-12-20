/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_VECTOR_H
#define FASTTEXT_VECTOR_H

#include <cstdint>
#include <ostream>

#include "real.h"

namespace fasttext {

class Matrix;

class Vector {

  public:
    int64_t m_;
    real* data_;

    explicit Vector(int64_t);
    ~Vector();

    real& operator[](int64_t);
    const real& operator[](int64_t) const;

    int64_t size() const;
    void zero();
    void mul(real);
    void addRow(const Matrix&, int64_t);
    void addRow(const Matrix&, int64_t, real);
    void mul(const Matrix&, const Vector&);
    int64_t argmax();
};

std::ostream& operator<<(std::ostream&, const Vector&);

}

#endif
