// Copyright (c) 2018-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <iostream>
#include <string>

// Check that the next n bytes are continuation bytes.
bool continuation(uint8_t* str, int n)
{
  for (int i = 0; i < n; i++) {
    if ((str[i] & 0xc0) != 0x80) return false;
  }
  return true;
}

// Invalid UTF8 correspond to codepoints which are larger than U+10FFFF.
// This value is encoded in UTF8 as:
//  * 11110.100 10.001111 10.111111 10.111111
// We thus check if the first byte is larger than 0xf4, or if it is equal
// to 0xf4 and the second byte is larger than 0x8f.
bool invalid(uint8_t* str)
{
  return str[0] > 0xf4 || (str[0] == 0xf4 && str[1] > 0x8f);
}

// Surrogate halves corresponds to the range U+D800 through U+DFFF,
// which are encoded in UTF8 as:
//  * 1110.1101 10.100000 10.000000
//  * 1110.1101 10.111111 10.111111
// We thus check is the first byte is equal to 0xed and if the
// sixth bit of the second byte is set.
bool surrogate(uint8_t* str)
{
  return str[0] == 0xed && str[1] & 0x20;
}

// Sequences of length 2 are overlong if the leading 4 bits (noted as y)
// are equal to 0: 110.yyyyx 10xxxxxx
bool overlong_2(uint8_t* str)
{
  return (str[0] & 0x1e) == 0;
}

// Sequences of lenth 3 are overlong if the leading 5 bits (noted as y)
// are equal to 0: 1110.yyyy 10.yxxxxx 10.xxxxxx
bool overlong_3(uint8_t* str)
{
  return (str[0] & 0x0f) == 0 && (str[1] & 0x20) == 0;
}

// Sequences of length 4 are overlong if the leading 5 bits (noted as y)
// are equal to 0: 11110.yyy 10.yyxxxx 10.xxxxxx 10.xxxxxx
bool overlong_4(uint8_t* str)
{
  return (str[0] & 0x07) == 0 && (str[1] & 0x30) == 0;
}

bool valid_utf8(uint8_t* str, size_t length)
{
  uint8_t* end = str + length;
  while (str < end) {
    if (str[0] < 0x80) {
      // 0.xxxxxxx
      str += 1;
    } else if ((str[0] & 0xe0) == 0xc0) {
      // 110.xxxxx 10.xxxxxx
      if (str + 1 >= end) return false;
      if (!continuation(str + 1, 1)) return false;
      if (overlong_2(str)) return false;
      str += 2;
    } else if ((str[0] & 0xf0) == 0xe0) {
      // 1110.xxxx 10.xxxxxx 10.xxxxxx
      if (str + 2 >= end) return false;
      if (!continuation(str + 1, 2)) return false;
      if (overlong_3(str)) return false;
      if (surrogate(str)) return false;
      str += 3;
    } else if ((str[0] & 0xf8) == 0xf0) {
      // 11110.xxx 10.xxxxxx 10.xxxxxx 10.xxxxxx
      if (str + 3 >= end) return false;
      if (!continuation(str + 1, 3)) return false;
      if (overlong_4(str)) return false;
      if (invalid(str)) return false;
      str += 4;
    } else {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv)
{
  std::ios_base::sync_with_stdio(false);
  for (std::string line; std::getline(std::cin, line);) {
    if (valid_utf8((uint8_t*) line.data(), line.length())) {
      std::cout << line << std::endl;
    }
  }
  return 0;
}
