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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <random>

#include "utils.h"
#include "vector.h"

namespace fasttext {

Matrix::Matrix() {
  m_ = 0;
  n_ = 0;
  data_mem_ = nullptr;
  data_mmap_ = nullptr;
  data_ = data_mem_;
}

Matrix::Matrix(int64_t m, int64_t n) {
  m_ = m;
  n_ = n;
  data_mem_ = new real[m * n];
  data_mmap_ = nullptr;
  data_ = data_mem_;
}

Matrix::~Matrix() {
  delete[] data_mem_;
  if (data_mmap_) {
    // Don't forget to free the mmapped memory
    munmap(data_mmap_, fileInfo.st_size);
    close(file_);
  }
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
  out.write((char*) data_, m_ * n_ * sizeof(real));
}

void Matrix::load(std::istream& in) {
  data_ = data_mem_;
  in.read((char*) &m_, sizeof(int64_t));
  in.read((char*) &n_, sizeof(int64_t));
  delete[] data_;
  data_ = new real[m_ * n_];
  in.read((char*) data_, m_ * n_ * sizeof(real));
}

void Matrix::load2mmap(std::istream& in, const std::string& filename) {
  // create mmap
  file_ = open(filename.c_str(), O_RDONLY);
  fstat(file_, &fileInfo);
  data_mmap_ = mmap(NULL, fileInfo.st_size, PROT_READ, MAP_SHARED, file_, 0);

  // assign data (allow offset)
  in.read((char*) &m_, sizeof(int64_t));
  in.read((char*) &n_, sizeof(int64_t));
  data_ = (real*)((char*)data_mmap_ + (unsigned long long)in.tellg());
  in.seekg(m_ * n_ * sizeof(real), std::ios::cur);
}

}
