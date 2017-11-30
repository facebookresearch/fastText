/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>

#include <vector>
#include <memory>

#include "real.h"

#include "matrix.h"
#include "vector.h"

#include "productquantizer.h"

namespace fasttext {

class QMatrix {
  protected:
    std::unique_ptr<ProductQuantizer> pq_;
    std::unique_ptr<ProductQuantizer> npq_;

    uint8_t* codes_;
    uint8_t* norm_codes_;

    bool qnorm_;

    int64_t m_;
    int64_t n_;

    int32_t codesize_;

  public:

    QMatrix();
    QMatrix(const Matrix&, int32_t, bool);
    ~QMatrix();

    int64_t getM() const;
    int64_t getN() const;

    void quantizeNorm(const Vector&);
    void quantize(const Matrix&);

    void addToVector(Vector& x, int32_t t) const;
    real dotRow(const Vector&, int64_t) const;

    void save(std::ostream&);
    void load(std::istream&);
};

}
