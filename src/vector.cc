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

#include <algorithm>
#include <iomanip>
#include <utility>

#include "matrix.h"
#include "utils.h"

namespace fasttext {
Vector::Vector()
  : m_(0), data_(nullptr) {
}

Vector::Vector(int64_t m)
  : m_(m), data_(new real[m]) {
}

Vector::Vector(const Vector &other)
  : Vector(other.m_) {
  std::copy(
    other.data_.get(),
    other.data_.get() + other.m_,
    data_.get()
  );
}

Vector & Vector::operator=(Vector other)
{
  swap(other);
  return *this;
}

int64_t Vector::size() const {
  return m_;
}

void Vector::zero() {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
  }
}

void Vector::mul(real a) {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] *= a;
  }
}

void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += A.data_[i * A.n_ + j];
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += a * A.data_[i * A.n_ + j];
  }
}

void Vector::mul(const Matrix& A, const Vector& vec) {
  assert(A.m_ == m_);
  assert(A.n_ == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
    for (int64_t j = 0; j < A.n_; j++) {
      data_[i] += A.data_[i * A.n_ + j] * vec.data_[j];
    }
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

void Vector::swap(Vector &other)
{
  using std::swap;
  swap(m_, other.m_);
  swap(data_, other.data_);
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

namespace std {

template<>
void swap<fasttext::Vector>(fasttext::Vector &lhs, fasttext::Vector &rhs) {
  lhs.swap(rhs);
}

}
