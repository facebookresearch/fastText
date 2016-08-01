/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include "args.h"

extern Args args;

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary() {
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  ntokens_ = 0;
  word2int_.resize(MAX_VOCAB_SIZE);
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
}

Dictionary::~Dictionary() {}

int32_t Dictionary::find(const std::string& w) {
  int32_t h = hashFn(w) % MAX_VOCAB_SIZE;
  while (word2int_[h] != -1 && words_[word2int_[h]].word != w) {
    h = (h + 1) % MAX_VOCAB_SIZE;
  }
  return h;
}

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  if (word2int_[h] == -1) {
    word2int_[h] = size_;
    entry we;
    we.word = w;
    we.id = size_;
    we.uf = 1;
    we.type = (w.find(args.label) == 0) ? 1 : 0;
    words_.push_back(we);
    size_++;
  } else {
    words_[word2int_[h]].uf++;
  }
}

int32_t Dictionary::nwords() {
  return nwords_;
}

int32_t Dictionary::nlabels() {
  return nlabels_;
}

int64_t Dictionary::ntokens() {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getNgrams(int32_t i) {
  assert(i >= 0 && i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getNgrams(const std::string& word) {
  std::vector<int32_t> ngrams;
  int32_t i = getId(word);
  if (i >= 0) {
    ngrams = words_[i].subwords;
  } else {
    computeNgrams(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

bool Dictionary::discard(int32_t id, real rand) {
  assert(id >= 0 && id < nwords_);
  if (args.model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w) {
  int32_t h = find(w);
  return word2int_[h];
}

int8_t Dictionary::getType(int32_t id) {
  assert(id >= 0 && id < size_);
  return words_[id].type;
}

std::string Dictionary::getWord(int32_t id) {
  assert(id >= 0 && id < size_);
  return words_[id].word;
}

uint32_t Dictionary::hash(const std::string& str) {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeNgrams(const std::string& word,
                               std::vector<int32_t>& ngrams) {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args.maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args.minn) {
        int32_t h = hash(ngram) % args.bucket;
        ngrams.push_back(nwords_ + h);
      }
    }
  }
}

void Dictionary::initNgrams() {
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    std::string word = BOW + it->word + EOW;
    computeNgrams(word, it->subwords);
  }
}

std::string Dictionary::readWord(std::ifstream& fin)
{
  char c;
  std::string word;
  while (!fin.eof()) {
    fin.get(c);
    if (iswspace(c)) {
      if (word.empty()) {
        if (c == '\n') return EOS;
        continue;
      } else {
        if (c == '\n') fin.unget();
        return word;
      }
    }
    word.push_back(c);
  }
  return word;
}

void Dictionary::readFromFile(const std::string& fname) {
  std::ifstream ifs(fname);
  std::string word = readWord(ifs);
  while (!word.empty()) {
    add(word);
    ntokens_++;
    if (ntokens_ % 1000000 == 0) {
      std::cout << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
    }
    word = readWord(ifs);
  }
  ifs.close();
  std::cout << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;

  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.uf > e2.uf;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [](const entry& e) {
        return e.uf < args.minCount;
      }), words_.end());
  words_.shrink_to_fit();

  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_;
    it->id = size_;
    it->subwords.push_back(size_);
    size_++;
    if (it->type == 0) nwords_++;
    if (it->type == 1) nlabels_++;
  }
  std::cout << "number of labels: " << nlabels_ << std::endl;

  initTableDiscard();
  initNgrams();
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (auto& w : words_) {
    real f = real(w.uf) / real(ntokens_);
    pdiscard_[w.id] = sqrt(args.t / f) + args.t / f;
  }
}

std::vector<int64_t> Dictionary::getWordFreq() {
  std::vector<int64_t> freq;
  for (auto& w : words_) {
    if (w.type == 0) {
      freq.push_back(w.uf);
    }
  }
  return freq;
}

std::vector<int64_t> Dictionary::getLabelFreq() {
  std::vector<int64_t> freq;
  for (auto& w : words_) {
    if (w.type == 1) {
      freq.push_back(w.uf);
    }
  }
  return freq;
}

void Dictionary::addNgrams(std::vector<int32_t>& line, int32_t n) {
  int32_t line_size = line.size();
  for (int32_t i = 0; i < line_size; i++) {
    uint64_t h = line[i];
    for (int32_t j = i + 1; j < line_size && j < i + n; j++) {
      h = h * 116049371 + line[j];
      line.push_back(nwords_ + (h % args.bucket));
    }
  }
}

int32_t Dictionary::getLine(std::ifstream& ifs,
                            std::vector<int32_t>& words,
                            std::vector<int32_t>& labels,
                            std::minstd_rand& rng) {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens_ = 0;
  words.clear();
  labels.clear();
  if (ifs.eof()) {
    ifs.clear();
    ifs.seekg(std::streampos(0));
  }
  while (!(token = readWord(ifs)).empty()) {
    ntokens_++;
    if (token == EOS) break;
    int32_t wid = getId(token);
    if (wid < 0) continue;
    int8_t type = getType(wid);
    if (type == 0 && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (type == 1) {
      labels.push_back(wid - nwords_);
    }
    if (words.size() > MAX_LINE_SIZE && args.model != model_name::sup) break;
  }
  return ntokens_;
}

std::string Dictionary::getLabel(int32_t lid) {
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ofstream& ofs) {
  char ender = 0;
  int k = 0;
  if (ofs.is_open()) {
    ofs.write((char*) &size_, sizeof(int32_t));
    ofs.write((char*) &nwords_, sizeof(int32_t));
    ofs.write((char*) &nlabels_, sizeof(int32_t));
    ofs.write((char*) &ntokens_, sizeof(int64_t));
    for (int32_t i = 0; i < size_; i++) {
      entry e = words_[i];
      std::string w = e.word;
      for (int c = 0; c < w.size(); c++) {
        ofs.write(&w[c], sizeof(char));
      }
      ofs.write(&ender, 1);
      ofs.write((char*) &(e.id), sizeof(int32_t));
      ofs.write((char*) &(e.uf), sizeof(int64_t));
      ofs.write((char*) &(e.type), sizeof(int8_t));
      int32_t nSubwords = e.subwords.size();
      ofs.write((char*) &nSubwords, sizeof(int32_t));
      for (int32_t subword : e.subwords) {
        ofs.write((char*) &(subword), sizeof(int32_t));
      }
    }
  }
}

void Dictionary::load(std::ifstream& ifs) {
  words_.clear();
  if (ifs.is_open()) {
    ifs.read((char*) &size_, sizeof(int32_t));
    ifs.read((char*) &nwords_, sizeof(int32_t));
    ifs.read((char*) &nlabels_, sizeof(int32_t));
    ifs.read((char*) &ntokens_, sizeof(int64_t));
    for (int32_t j = 0; j < size_; j++) {
      char c;
      entry we;
      ifs.read(&c, sizeof(char));
      while (c != 0) {
        we.word.push_back(c);
        ifs.read(&c, sizeof(char));
      }
      ifs.read((char*) &we.id, sizeof(int32_t));
      ifs.read((char*) &we.uf, sizeof(int64_t));
      ifs.read((char*) &we.type, sizeof(int8_t));
      int32_t nSubwords = 0;
      ifs.read((char*) &nSubwords, sizeof(int32_t));
      int32_t subword = 0;
      for (int z = 0; z < nSubwords; z++) {
        ifs.read((char*) &subword, sizeof(int32_t));
        we.subwords.push_back(subword);
      }
      words_.push_back(we);
    }
  }
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = it->id;
  }
  initTableDiscard();
}
