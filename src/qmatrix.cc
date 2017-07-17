/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "qmatrix.h"

#include <assert.h>
#include <iostream>

namespace fasttext {

QMatrix::QMatrix() : qnorm_(false),
  m_(0), n_(0), codesize_(0) {}

QMatrix::QMatrix(const Matrix& mat, int32_t dsub, bool qnorm)
      : qnorm_(qnorm), m_(mat.m_), n_(mat.n_),
        codesize_(m_ * ((n_ + dsub - 1) / dsub)) {
  if (codesize_ > 0) {
    codes_ = new uint8_t[codesize_];
  }
  pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(n_, dsub));
  if (qnorm_) {
    norm_codes_ = new uint8_t[m_];
    npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(1, 1));
  }
  quantize(mat);
}

QMatrix::~QMatrix() {
  if (codesize_ > 0) {
    delete[] codes_;
  }
  if (qnorm_) { delete[] norm_codes_; }
}

void QMatrix::quantizeNorm(const Vector& norms) {
  assert(qnorm_);
  assert(norms.m_ == m_);
  auto dataptr = norms.data_;
  npq_->train(m_, dataptr);
  npq_->compute_codes(dataptr, norm_codes_, m_);
}

void QMatrix::quantize(const Matrix& matrix) {
  assert(n_ == matrix.n_);
  assert(m_ == matrix.m_);
  Matrix temp(matrix);
  if (qnorm_) {
    Vector norms(temp.m_);
    temp.l2NormRow(norms);
    temp.divideRow(norms);
    quantizeNorm(norms);
  }
  auto dataptr = temp.data_;
  pq_->train(m_, dataptr);
  pq_->compute_codes(dataptr, codes_, m_);
}

void QMatrix::addToVector(Vector& x, int32_t t) const {
  real norm = 1;
  if (qnorm_) {
    norm = npq_->get_centroids(0, norm_codes_[t])[0];
  }
  pq_->addcode(x, codes_, t, norm);
}

real QMatrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real norm = 1;
  if (qnorm_) {
    norm = npq_->get_centroids(0, norm_codes_[i])[0];
  }
  return pq_->mulcode(vec, codes_, i, norm);
}

int64_t QMatrix::getM() const {
  return m_;
}

int64_t QMatrix::getN() const {
  return n_;
}

void QMatrix::save(std::ostream& out) {
    out.write((char*) &qnorm_, sizeof(qnorm_));
    out.write((char*) &m_, sizeof(m_));
    out.write((char*) &n_, sizeof(n_));
    out.write((char*) &codesize_, sizeof(codesize_));
    out.write((char*) codes_, codesize_ * sizeof(uint8_t));
    pq_->save(out);
    if (qnorm_) {
      out.write((char*) norm_codes_, m_ * sizeof(uint8_t));
      npq_->save(out);
    }
}

void QMatrix::load(std::istream& in) {
    in.read((char*) &qnorm_, sizeof(qnorm_));
    in.read((char*) &m_, sizeof(m_));
    in.read((char*) &n_, sizeof(n_));
    in.read((char*) &codesize_, sizeof(codesize_));
    codes_ = new uint8_t[codesize_];
    in.read((char*) codes_, codesize_ * sizeof(uint8_t));
    pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
    pq_->load(in);
    if (qnorm_) {
      norm_codes_ = new uint8_t[m_];
      in.read((char*) norm_codes_, m_ * sizeof(uint8_t));
      npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
      npq_->load(in);
    }
}

}
