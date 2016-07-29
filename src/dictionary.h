/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef DICTIONARY_H
#define DICTIONARY_H

#include "real.h"
#include <vector>
#include <string>
#include <fstream>
#include <random>

typedef int32_t id_type;

struct entry {
  std::string word;
  int32_t id;
  int64_t uf;
  int8_t type;
  std::vector<int32_t> subwords;
};

class Dictionary {
  private:
    static const int32_t MAX_VOCAB_SIZE = 30000000;
    static const int32_t MAX_LINE_SIZE = 1024;

    int32_t find(const std::string&);
    void initTableDiscard();
    void initNgrams();

    std::vector<int32_t> word2int_;
    std::vector<entry> words_;
    std::vector<real> pdiscard_;
    int32_t size;
    int32_t nwords;
    int32_t nlabels;
    int64_t ntokens;

  public:
    static const std::string EOS;
    static const std::string BOW;
    static const std::string EOW;
    static const std::hash<std::string> hashFn;

    Dictionary();
    ~Dictionary();
    int32_t getNumWords();
    int32_t getNumLabels();
    int64_t getNumTokens();
    int32_t getId(const std::string&);
    int8_t getType(int32_t);
    bool discard(int32_t, real);
    std::string getWord(int32_t);
    const std::vector<int32_t>& getNgrams(int32_t);
    const std::vector<int32_t> getNgrams(const std::string&);
    void computeNgrams(const std::string&, std::vector<int32_t>&);
    uint32_t hash(const std::string& str);
    void add(const std::string&);
    std::string readWord(std::ifstream&);
    void readFromFile(const std::string&);
    void save(std::ofstream&);
    void load(std::ifstream&);
    std::vector<int64_t> getWordFreq();
    std::vector<int64_t> getLabelFreq();
    void addNgrams(std::vector<int32_t>&, int32_t);
    int32_t getLine(std::ifstream&, std::vector<int32_t>&,
                    std::vector<int32_t>&, std::minstd_rand&);
};

#endif
