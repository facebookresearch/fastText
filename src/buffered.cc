/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "buffered.h"

void Buffered::fetch() {
  cursor_ = 0;
  is_->read(buf_.data(), BUF_SIZE);
  buf_last_ = is_->gcount() - 1;
}

Buffered::Buffered(std::unique_ptr<std::istream>&& is)
  : is_(std::move(is))
{
  fetch();
}

bool Buffered::eof() const {
  return cursor_ >= buf_last_ && is_->eof();
}

bool Buffered::peek(char& c) const {
  if (eof())
    return false;
  c = buf_[cursor_];
  return true;
}

void Buffered::advance() {
  if (cursor_ < buf_last_)
    cursor_++;
  else if (!is_->eof())
    fetch();
}

void Buffered::reset() {
  is_->clear();
  is_->seekg(0);
  fetch();
}
