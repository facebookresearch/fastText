// Copyright (c) 2018-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

uint64_t fnv1a_64(uint8_t *data, size_t sz, uint64_t h=14695981039346656037ull)
{
  for (size_t i = 0; i < sz; i++, data++) {
    h ^= uint64_t(*data);
    h *= 1099511628211ull;
  }
  return h;
}

int main(int argc, char** argv)
{
  uint64_t init_values[] = {
    14695981039346656037ull,
    9425296925403859339ull,
    13716263814064014149ull,
    3525492407291847033ull,
    8607404175481815707ull,
    9818874561736458749ull,
    10026508429719773353ull,
    3560712257386009938ull
  };
  size_t n = 1ull<<34, num_hashes = 2;
  std::vector<bool> seen(n);

  std::ios_base::sync_with_stdio(false);

  for (std::string line; std::getline(std::cin, line);) {
    bool b = true;
    for (size_t i = 0; i < num_hashes; i++) {
      uint64_t h = fnv1a_64((uint8_t*) line.data(), line.length(), init_values[i]) % n;
      b = b && seen[h];
      seen[h] = true;
    }
    if (!b) {
      std::cout << line << std::endl;
    }
  }
  return 0;
}
