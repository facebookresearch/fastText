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
    
    // mul method can now take two parameters.
    // These parameters are from and length, so one can multiply a real only to a specific part of the vector.
    // So for instance, from can be vec.size()/2 and length vec.size()/4, which would apply the multiplication
    // only to the elements of the vector that are in between the
    // positions vec.size()/2 (included) and [vec.size()/2 + vec.size()/4] (not included)
    // By default, from is 0 and length is (vec.size() - from).
    void mul(real, int64_t from = 0, int64_t length = -1);
    void addRow(const Matrix&, int64_t);
    void addRow(const Matrix&, int64_t, real);
    void mul(const Matrix&, const Vector&);
    int64_t argmax();

    void addVector(const Vector&, int64_t pos);
};

std::ostream& operator<<(std::ostream&, const Vector&);

}

#endif
