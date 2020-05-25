/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "vector.h"

#include <assert.h>

#include <cmath>
#include <iomanip>

#include "matrix.h"

namespace fasttext {

Vector::Vector(int64_t m) : data_(m) {}

void Vector::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

real Vector::norm() const {
  real sum = 0;
  for (int64_t i = 0; i < size(); i++) {
    sum += data_[i] * data_[i];
  }
  return std::sqrt(sum);
}

void Vector::mul(real a) {
  for (int64_t i = 0; i < size(); i++) {
    data_[i] *= a;
  }
}

void Vector::addVector(const Vector& source) {
  assert(size() == source.size());
  for (int64_t i = 0; i < size(); i++) {
    data_[i] += source.data_[i];
  }
}

void Vector::addVector(const Vector& source, real s) {
  assert(size() == source.size());
  for (int64_t i = 0; i < size(); i++) {
    data_[i] += s * source.data_[i];
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  A.addRowToVector(*this, i, a);
}

void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  A.addRowToVector(*this, i);
}

void Vector::mul(const Matrix& A, const Vector& vec) {
  assert(A.size(0) == size());
  assert(A.size(1) == vec.size());
  for (int64_t i = 0; i < size(); i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < size(); i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.size(); j++) {
    os << v[j] << ' ';
  }
  return os;
}

} // namespace fasttext
