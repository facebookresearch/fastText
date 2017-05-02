/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector.h"

#include <assert.h>

#include <iomanip>
#include <cmath>

#include "matrix.h"
#include "qmatrix.h"

namespace fasttext {

Vector::Vector(int64_t m) {
  m_ = m;
  data_ = new real[m];
}

Vector::~Vector() {
  delete[] data_;
}

int64_t Vector::size() const {
  return m_;
}

void Vector::zero() {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
  }
}

real Vector::norm() const {
  real sum = 0;
  for (int64_t i = 0; i < m_; i++) {
    sum += data_[i] * data_[i];
  }
  return std::sqrt(sum);
}

void Vector::mul(real a) {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] *= a;
  }
}

void Vector::addVector(const Vector& source) {
  assert(m_ == source.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] += source.data_[i];
  }
}

void Vector::addVector(const Vector& source, real s) {
  assert(m_ == source.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] += s * source.data_[i];
  }
}

void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += A.at(i, j);
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += a * A.at(i, j);
  }
}

void Vector::addRow(const QMatrix& A, int64_t i) {
  assert(i >= 0);
  A.addToVector(*this, i);
}

void Vector::mul(const Matrix& A, const Vector& vec) {
  assert(A.m_ == m_);
  assert(A.n_ == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

void Vector::mul(const QMatrix& A, const Vector& vec) {
  assert(A.getM() == m_);
  assert(A.getN() == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < m_; i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

real& Vector::operator[](int64_t i) {
  return data_[i];
}

const real& Vector::operator[](int64_t i) const {
  return data_[i];
}

std::ostream& operator<<(std::ostream& os, const Vector& v)
{
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.m_; j++) {
    os << v.data_[j] << ' ';
  }
  return os;
}

}
