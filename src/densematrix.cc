/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "densematrix.h"

#include <random>
#include <stdexcept>
#include <utility>
#include "utils.h"
#include "vector.h"

namespace fasttext {

DenseMatrix::DenseMatrix() : DenseMatrix(0, 0) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n) : Matrix(m, n), data_(m * n) {}

DenseMatrix::DenseMatrix(DenseMatrix&& other) noexcept
    : Matrix(other.m_, other.n_), data_(std::move(other.data_)) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n, real* dataPtr)
    : Matrix(m, n), data_(dataPtr, dataPtr + (m * n)) {}

real DenseMatrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += at(i, j) * vec[j];
  }
  if (std::isnan(d)) {
    throw EncounteredNaNError();
  }
  return d;
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += at(i, j);
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i, real a) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += a * at(i, j);
  }
}

void DenseMatrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = std::vector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

} // namespace fasttext
