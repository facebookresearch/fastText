/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "quantmatrix.h"

#include <assert.h>
#include <iostream>
#include <stdexcept>

namespace fasttext {

QuantMatrix::QuantMatrix() : Matrix(), qnorm_(false), codesize_(0) {}

QuantMatrix::QuantMatrix(DenseMatrix&& mat, int32_t dsub, bool qnorm)
    : Matrix(mat.size(0), mat.size(1)),
      qnorm_(qnorm),
      codesize_(mat.size(0) * ((mat.size(1) + dsub - 1) / dsub)) {
  codes_.resize(codesize_);
  pq_ = std::unique_ptr<ProductQuantizer>(new ProductQuantizer(n_, dsub));
  if (qnorm_) {
    norm_codes_.resize(m_);
    npq_ = std::unique_ptr<ProductQuantizer>(new ProductQuantizer(1, 1));
  }
  quantize(std::forward<DenseMatrix>(mat));
}

void QuantMatrix::quantizeNorm(const Vector& norms) {
  assert(qnorm_);
  assert(norms.size() == m_);
  auto dataptr = norms.data();
  npq_->train(m_, dataptr);
  npq_->compute_codes(dataptr, norm_codes_.data(), m_);
}

void QuantMatrix::quantize(DenseMatrix&& mat) {
  if (qnorm_) {
    Vector norms(mat.size(0));
    mat.l2NormRow(norms);
    mat.divideRow(norms);
    quantizeNorm(norms);
  }
  auto dataptr = mat.data();
  pq_->train(m_, dataptr);
  pq_->compute_codes(dataptr, codes_.data(), m_);
}

real QuantMatrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real norm = 1;
  if (qnorm_) {
    norm = npq_->get_centroids(0, norm_codes_[i])[0];
  }
  return pq_->mulcode(vec, codes_.data(), i, norm);
}

void QuantMatrix::addVectorToRow(const Vector&, int64_t, real) {
  throw std::runtime_error("Operation not permitted on quantized matrices.");
}

void QuantMatrix::addRowToVector(Vector& x, int32_t i, real a) const {
  real norm = 1;
  if (qnorm_) {
    norm = npq_->get_centroids(0, norm_codes_[i])[0];
  }
  pq_->addcode(x, codes_.data(), i, a * norm);
}

void QuantMatrix::addRowToVector(Vector& x, int32_t i) const {
  real norm = 1;
  if (qnorm_) {
    norm = npq_->get_centroids(0, norm_codes_[i])[0];
  }
  pq_->addcode(x, codes_.data(), i, norm);
}

void QuantMatrix::averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const {
  x.zero();
  for (auto it = rows.cbegin(); it != rows.cend(); ++it) {
    addRowToVector(x, *it);
  }
  x.mul(1.0 / rows.size());
}

void QuantMatrix::save(std::ostream& out) const {
  out.write((char*)&qnorm_, sizeof(qnorm_));
  out.write((char*)&m_, sizeof(m_));
  out.write((char*)&n_, sizeof(n_));
  out.write((char*)&codesize_, sizeof(codesize_));
  out.write((char*)codes_.data(), codesize_ * sizeof(uint8_t));
  pq_->save(out);
  if (qnorm_) {
    out.write((char*)norm_codes_.data(), m_ * sizeof(uint8_t));
    npq_->save(out);
  }
}

void QuantMatrix::load(std::istream& in) {
  in.read((char*)&qnorm_, sizeof(qnorm_));
  in.read((char*)&m_, sizeof(m_));
  in.read((char*)&n_, sizeof(n_));
  in.read((char*)&codesize_, sizeof(codesize_));
  codes_ = std::vector<uint8_t>(codesize_);
  in.read((char*)codes_.data(), codesize_ * sizeof(uint8_t));
  pq_ = std::unique_ptr<ProductQuantizer>(new ProductQuantizer());
  pq_->load(in);
  if (qnorm_) {
    norm_codes_ = std::vector<uint8_t>(m_);
    in.read((char*)norm_codes_.data(), m_ * sizeof(uint8_t));
    npq_ = std::unique_ptr<ProductQuantizer>(new ProductQuantizer());
    npq_->load(in);
  }
}

void QuantMatrix::dump(std::ostream&) const {
  throw std::runtime_error("Operation not permitted on quantized matrices.");
}

} // namespace fasttext
