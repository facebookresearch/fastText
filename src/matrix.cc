/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "matrix.h"

#include <assert.h>

#include <random>

#include "utils.h"
#include "vector.h"

namespace fasttext {

Matrix::Matrix()
  : m_(0), n_(0), data_() {
}

Matrix::Matrix(int64_t m, int64_t n)
  : m_(m), n_(n), data_(new real[m * n]) {
}

Matrix::Matrix(const Matrix &other)
  : Matrix(other.m_, other.n_) {
  std::copy(
    other.data_.get(),
    other.data_.get() + m_ * n_,
    data_.get()
  );
}

Matrix & Matrix::operator=(Matrix other)
{
  swap(other);
  return *this;
}

void Matrix::swap(Matrix &other)
{
  using std::swap;
  swap(m_, other.m_);
  swap(n_, other.n_);
  swap(data_, other.data_);
}

void Matrix::zero() {
  for (int64_t i = 0; i < (m_ * n_); i++) {
      data_[i] = 0.0;
  }
}

void Matrix::uniform(real a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = uniform(rng);
  }
}

void Matrix::addRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.m_ == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec.data_[j];
  }
}

real Matrix::dotRow(const Vector& vec, int64_t i) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.m_ == n_);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += data_[i * n_ + j] * vec.data_[j];
  }
  return d;
}

void Matrix::save(std::ostream& out) {
  out.write((char*) &m_, sizeof(int64_t));
  out.write((char*) &n_, sizeof(int64_t));
  out.write((char*) data_.get(), m_ * n_ * sizeof(real));
}

void Matrix::load(std::istream& in) {
  in.read((char*) &m_, sizeof(int64_t));
  in.read((char*) &n_, sizeof(int64_t));
  data_ = std::unique_ptr<real[]>(new real[m_ * n_]);
  in.read((char*) data_.get(), m_ * n_ * sizeof(real));
}

}

namespace std {

template<>
void swap<fasttext::Matrix>(fasttext::Matrix &lhs, fasttext::Matrix &rhs) {
  lhs.swap(rhs);
}

}
