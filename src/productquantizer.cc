/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "productquantizer.h"

#include <algorithm>
#include <iostream>
#include <numeric>

namespace fasttext {

real distL2(const real* x, const real* y, int32_t d) {
  real dist = 0;
  for (auto i = 0; i < d; i++) {
    auto tmp = x[i] - y[i];
    dist += tmp * tmp;
  }
  return dist;
}

ProductQuantizer::ProductQuantizer(int32_t dim, int32_t dsub): dim_(dim),
  nsubq_(dim / dsub), dsub_(dsub), centroids_(dim * ksub_), rng(seed_) {
  lastdsub_ = dim_ % dsub;
  if (lastdsub_ == 0) {lastdsub_ = dsub_;}
  else {nsubq_++;}
}

const real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) const {
  if (m == nsubq_ - 1) {return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];}
  return &centroids_[(m * ksub_ + i) * dsub_];
}

real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) {
  if (m == nsubq_ - 1) {return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];}
  return &centroids_[(m * ksub_ + i) * dsub_];
}

real ProductQuantizer::assign_centroid(const real * x, const real* c0,
                                       uint8_t* code, int32_t d) const {
  const real* c = c0;
  real dis = distL2(x, c, d);
  code[0] = 0;
  for (auto j = 1; j < ksub_; j++) {
    c += d;
    real disij = distL2(x, c, d);
    if (disij < dis) {
      code[0] = (uint8_t) j;
      dis = disij;
    }
  }
  return dis;
}

void ProductQuantizer::Estep(const real* x, const real* centroids,
                             uint8_t* codes, int32_t d,
                             int32_t n) const {
  for (auto i = 0; i < n; i++) {
    assign_centroid(x + i * d, centroids, codes + i, d);
  }
}

void ProductQuantizer::MStep(const real* x0, real* centroids,
                             const uint8_t* codes,
                             int32_t d, int32_t n) {
  std::vector<int32_t> nelts(ksub_, 0);
  memset(centroids, 0, sizeof(real) * d * ksub_);
  const real* x = x0;
  for (auto i = 0; i < n; i++) {
    auto k = codes[i];
    real* c = centroids + k * d;
    for (auto j = 0; j < d; j++) {
      c[j] += x[j];
    }
    nelts[k]++;
    x += d;
  }

  real* c = centroids;
  for (auto k = 0; k < ksub_; k++) {
    real z = (real) nelts[k];
    if (z != 0) {
      for (auto j = 0; j < d; j++) {
        c[j] /= z;
      }
    }
    c += d;
  }

  std::uniform_real_distribution<> runiform(0,1);
  for (auto k = 0; k < ksub_; k++) {
    if (nelts[k] == 0) {
      int32_t m = 0;
      while (runiform(rng) * (n - ksub_) >= nelts[m] - 1) {
        m = (m + 1) % ksub_;
      }
      memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d);
      for (auto j = 0; j < d; j++) {
        int32_t sign = (j % 2) * 2 - 1;
        centroids[k * d + j] += sign * eps_;
        centroids[m * d + j] -= sign * eps_;
      }
      nelts[k] = nelts[m] / 2;
      nelts[m] -= nelts[k];
    }
  }
}

void ProductQuantizer::kmeans(const real *x, real* c, int32_t n, int32_t d) {
  std::vector<int32_t> perm(n,0);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (auto i = 0; i < ksub_; i++) {
    memcpy (&c[i * d], x + perm[i] * d, d * sizeof(real));
  }
  uint8_t* codes = new uint8_t[n];
  for (auto i = 0; i < niter_; i++) {
    Estep(x, c, codes, d, n);
    MStep(x, c, codes, d, n);
  }
  delete [] codes;
}

void ProductQuantizer::train(int32_t n, const real * x) {
  if (n < ksub_) {
    std::cerr<<"Matrix too small for quantization, must have > 256 rows"<<std::endl;
    exit(1);
  }
  std::vector<int32_t> perm(n, 0);
  std::iota(perm.begin(), perm.end(), 0);
  auto d = dsub_;
  auto np = std::min(n, max_points_);
  real* xslice = new real[np * dsub_];
  for (auto m = 0; m < nsubq_; m++) {
    if (m == nsubq_-1) {d = lastdsub_;}
    if (np != n) {std::shuffle(perm.begin(), perm.end(), rng);}
    for (auto j = 0; j < np; j++) {
      memcpy (xslice + j * d, x + perm[j] * dim_ + m * dsub_, d * sizeof(real));
    }
    kmeans(xslice, get_centroids(m, 0), np, d);
  }
  delete [] xslice;
}

real ProductQuantizer::mulcode(const Vector& x, const uint8_t* codes,
                               int32_t t, real alpha) const {
  real res = 0.0;
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {d = lastdsub_;}
    for(auto n = 0; n < d; n++) {
      res += x[m * dsub_ + n] * c[n];
    }
  }
  return res * alpha;
}

void ProductQuantizer::addcode(Vector& x, const uint8_t* codes,
                               int32_t t, real alpha) const {
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {d = lastdsub_;}
    for(auto n = 0; n < d; n++) {
      x[m * dsub_ + n] += alpha * c[n];
    }
  }
}

void ProductQuantizer::compute_code(const real* x, uint8_t* code) const {
  auto d = dsub_;
  for (auto m = 0; m < nsubq_; m++) {
    if (m == nsubq_ - 1) {d = lastdsub_;}
    assign_centroid(x + m * dsub_, get_centroids(m, 0), code + m, d);
  }
}

void ProductQuantizer::compute_codes(const real* x, uint8_t* codes,
                                     int32_t n) const {
  for (auto i = 0; i < n; i++) {
    compute_code(x + i * dim_, codes + i * nsubq_);
  }
}

void ProductQuantizer::save(std::ostream& out) {
  out.write((char*) &dim_, sizeof(dim_));
  out.write((char*) &nsubq_, sizeof(nsubq_));
  out.write((char*) &dsub_, sizeof(dsub_));
  out.write((char*) &lastdsub_, sizeof(lastdsub_));
  out.write((char*) centroids_.data(), centroids_.size() * sizeof(real));
}

void ProductQuantizer::load(std::istream& in) {
  in.read((char*) &dim_, sizeof(dim_));
  in.read((char*) &nsubq_, sizeof(nsubq_));
  in.read((char*) &dsub_, sizeof(dsub_));
  in.read((char*) &lastdsub_, sizeof(lastdsub_));
  centroids_.resize(dim_ * ksub_);
  for (auto i=0; i < centroids_.size(); i++) {
    in.read((char*) &centroids_[i], sizeof(real));
  }
}

}
