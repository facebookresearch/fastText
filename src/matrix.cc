/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "matrix.h"

namespace fasttext {

Matrix::Matrix() : m_(0), n_(0) {}

Matrix::Matrix(int64_t m, int64_t n) : m_(m), n_(n) {}

int64_t Matrix::size(int64_t dim) const {
  assert(dim == 0 || dim == 1);
  if (dim == 0) {
    return m_;
  }
  return n_;
}

} // namespace fasttext
