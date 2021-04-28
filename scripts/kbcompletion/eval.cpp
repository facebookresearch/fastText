/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

std::string EOS = "</s>";

bool readWord(std::istream& in, std::string& word)
{
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  in.get();
  return !word.empty();
}

int main(int argc, char** argv) {
  int k = 10;
  if (argc < 4) {
    std::cerr<<"eval <pred> <gt> <kb> [<k>]"<<std::endl;
    exit(1);
  }
  if (argc == 5) { k = atoi(argv[4]);}

  std::string predfn(argv[1]);
  std::ifstream predf(predfn);
  std::string gtfn(argv[2]);
  std::ifstream gtf(gtfn);
  std::string kbfn(argv[3]);
  std::ifstream kbf(kbfn);

  if (!predf.is_open() || !gtf.is_open() || !kbf.is_open()) {
    std::cerr << "Files cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::unordered_map< std::string,
    std::unordered_map< std::string, bool > > KB;

  while (kbf.peek() != EOF) {
    std::string label, key, word;
    while (readWord(kbf, word)) {
      if (word == EOS) {break;}
      if (word.find("__label__") == 0) {label = word;}
      else {key += "|" + word;}
    }
    KB[key][label] = true;
  }
  kbf.close();

  double precision = 0.0;
  int32_t nexamples = 0;
  while (predf.peek() != EOF || gtf.peek() != EOF) {
    if (predf.peek() == EOF || gtf.peek() == EOF) {
      std::cerr<<"pred / gt files have diff sizes"<<std::endl;
      exit(1);
    }
    std::string label, key, word;

    while (readWord(gtf, word)) {
      if (word == EOS) {break;}
      if ( word.find("__label__") == 0) {label = word;}
      else {key += "|" + word;}
    }
    if (KB.find(key) == KB.end()) {
      std::cerr<<"empty key!"<<std::endl; exit(1);
    }

    int count = 0;bool eval = true;
    while (readWord(predf, word)) {
      if (word == EOS) {break;}
      if (!eval) {continue;}
      if (label == word) {precision += 1.0; eval = false;}
      else if (KB[key].find(word) == KB[key].end()) {count++;}
      if (count == k) {eval = false;}
    }
    nexamples++;
  }
  predf.close(); gtf.close();
  std::cout << "N:\t" << nexamples << std::endl;
  std::cout << "R@" << k << "\t" << precision / nexamples << std::endl;
}
