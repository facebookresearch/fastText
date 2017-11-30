/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstring>
#include <istream>
#include <ostream>
#include <vector>
#include <random>

#include "real.h"
#include "vector.h"

namespace fasttext {

class ProductQuantizer {
  protected:
    const int32_t nbits_ = 8;
    const int32_t ksub_ = 1 << nbits_;
    const int32_t max_points_per_cluster_ = 256;
    const int32_t max_points_ = max_points_per_cluster_ * ksub_;
    const int32_t seed_ = 1234;
    const int32_t niter_ = 25;
    const real eps_ = 1e-7;

    int32_t dim_;
    int32_t nsubq_;
    int32_t dsub_;
    int32_t lastdsub_;

    std::vector<real> centroids_;

    std::minstd_rand rng;

  public:
    ProductQuantizer() {}
    ProductQuantizer(int32_t, int32_t);

    real* get_centroids (int32_t, uint8_t);
    const real* get_centroids(int32_t, uint8_t) const;

    real assign_centroid(const real*, const real*, uint8_t*, int32_t) const;
    void Estep(const real*, const real*, uint8_t*, int32_t, int32_t) const;
    void MStep(const real*, real*, const uint8_t*, int32_t, int32_t);
    void kmeans(const real*, real*, int32_t, int32_t);
    void train(int, const real*);

    real mulcode(const Vector&, const uint8_t*, int32_t, real) const;
    void addcode(Vector&, const uint8_t*, int32_t, real) const;
    void compute_code(const real*, uint8_t*)  const;
    void compute_codes(const real*, uint8_t*, int32_t)  const;

    void save(std::ostream&);
    void load(std::istream&);
};

}
