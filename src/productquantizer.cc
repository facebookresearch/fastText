/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "productquantizer.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fasttext {

ProductQuantizer::ProductQuantizer(int32_t dim, int32_t dsub)
    : dim_(dim),
      nsubq_(dim / dsub),
      dsub_(dsub),
      centroids_(dim * ksub_),
      rng(seed_) {
  lastdsub_ = dim_ % dsub;
  if (lastdsub_ == 0) {
    lastdsub_ = dsub_;
  } else {
    nsubq_++;
  }
}

const real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) const {
  if (m == nsubq_ - 1) {
    return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
  }
  return &centroids_[(m * ksub_ + i) * dsub_];
}

real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) {
  if (m == nsubq_ - 1) {
    return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
  }
  return &centroids_[(m * ksub_ + i) * dsub_];
}

real ProductQuantizer::mulcode(
    const Vector& x,
    const uint8_t* codes,
    int32_t t,
    real alpha) const {
  real res = 0.0;
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    for (auto n = 0; n < d; n++) {
      res += x[m * dsub_ + n] * c[n];
    }
  }
  return res * alpha;
}

void ProductQuantizer::addcode(
    Vector& x,
    const uint8_t* codes,
    int32_t t,
    real alpha) const {
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    for (auto n = 0; n < d; n++) {
      x[m * dsub_ + n] += alpha * c[n];
    }
  }
}

void ProductQuantizer::load(std::istream& in) {
  in.read((char*)&dim_, sizeof(dim_));
  in.read((char*)&nsubq_, sizeof(nsubq_));
  in.read((char*)&dsub_, sizeof(dsub_));
  in.read((char*)&lastdsub_, sizeof(lastdsub_));
  centroids_.resize(dim_ * ksub_);
  for (auto i = 0; i < centroids_.size(); i++) {
    in.read((char*)&centroids_[i], sizeof(real));
  }
}

} // namespace fasttext
