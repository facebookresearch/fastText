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

const std::wstring Dictionary::EOS = L"</s>";
const std::wstring Dictionary::BOW = L"<";
const std::wstring Dictionary::EOW = L">";

Dictionary::Dictionary() {
  size = 0;
  nwords = 0;
  nlabels = 0;
  ntokens = 0;
  word2int_.resize(MAX_VOCAB_SIZE);
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
}

Dictionary::~Dictionary() {}

// const std::hash<std::wstring> Dictionary::hashFn;

int32_t Dictionary::find(const std::wstring& w) {
  int32_t h = hashFn(w) % MAX_VOCAB_SIZE;
  while (word2int_[h] != -1 && words_[word2int_[h]].word != w) {
    h = (h + 1) % MAX_VOCAB_SIZE;
  }
  return h;
}

void Dictionary::add(const std::wstring& w) {
  int32_t h = find(w);
  if (word2int_[h] == -1) {
    word2int_[h] = size;
    entry we;
    we.word = w;
    we.id = size;
    we.uf = 1;
    we.type = (w.find(args.label) == 0) ? 1 : 0;
    words_.push_back(we);
    size++;
  } else {
    words_[word2int_[h]].uf++;
  }
}

int32_t Dictionary::getNumWords() {
  return nwords;
}

int32_t Dictionary::getNumLabels() {
  return nlabels;
}

int64_t Dictionary::getNumTokens() {
  return ntokens;
}

const std::vector<int32_t>& Dictionary::getNgrams(int32_t i) {
  assert(i >= 0 && i < nwords);
  return words_[i].hashes;
}

std::vector<int32_t> Dictionary::getNgrams(const std::wstring& w) {
  std::vector<int32_t> ngramList;
  int32_t i = getId(w);
  if (i >= 0) {
    ngramList = words_[i].hashes;
  } else {
    std::wstring word = BOW + w + EOW;
    for (int32_t n = args.minn; n <= args.maxn; n++) {
      for (int32_t i = 0; i+n <= word.size(); i++) {
        std::wstring ngram = word.substr(i, n);
        int32_t hash = hashFn(ngram) % args.bucket;
        ngramList.push_back(nwords + hash);
      }
    }
  }
  return ngramList;
}

bool Dictionary::discard(int32_t id, real rand) {
  assert(id >= 0 && id < nwords);
  if (args.model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::wstring& w) {
  int32_t h = find(w);
  return word2int_[h];
}

int8_t Dictionary::getType(int32_t id) {
  assert(id >= 0 && id < size);
  return words_[id].type;
}

std::wstring Dictionary::getWord(int32_t id) {
  assert(id >= 0 && id < size);
  return words_[id].word;
}

void Dictionary::buildNgrams() {
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    std::wstring word = BOW + it->word + EOW;
    for (int32_t n = args.minn; n <= args.maxn; n++) {
      for (int32_t i = 0; i+n <= word.size(); i++) {
        std::wstring ngram = word.substr(i, n);
        int32_t hash = hashFn(ngram) % args.bucket;
        it->hashes.push_back(nwords + hash);
      }
    }
  }
}

std::wstring Dictionary::readWord(std::wistream& fin)
{
  wchar_t c;
  std::wstring word;
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
  std::wifstream ifs(fname);
  std::wstring word = readWord(ifs);
  while (!word.empty()) {
    add(word);
    ntokens++;
    if (ntokens % 1000000 == 0) {
      std::wcout << "\rRead " << ntokens  / 1000000 << "M words" << std::flush;
    }
    word = readWord(ifs);
  }
  ifs.close();
  std::wcout << "\rRead " << ntokens  / 1000000 << "M words" << std::endl;

  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.uf > e2.uf;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [](const entry& e) {
        return e.uf < args.minCount;
      }), words_.end());
  words_.shrink_to_fit();

  size = 0;
  nwords = 0;
  nlabels = 0;
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size;
    it->id = size;
    it->hashes.push_back(size);
    size++;
    if (it->type == 0) nwords++;
    if (it->type == 1) nlabels++;
  }
  std::wcout << "number of labels: " << nlabels << std::endl;

  initTableDiscard();
  buildNgrams();
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size);
  for (auto& w : words_) {
    real f = real(w.uf) / real(ntokens);
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
      line.push_back(nwords + (h % args.bucket));
    }
  }
}

int32_t Dictionary::getLine(std::wifstream& ifs,
                            std::vector<int32_t>& words,
                            std::vector<int32_t>& labels,
                            std::minstd_rand& rng) {
  std::uniform_real_distribution<> uniform(0, 1);
  std::wstring token;
  int32_t ntokens = 0;
  words.clear();
  labels.clear();
  if (ifs.eof()) {
    ifs.clear();
    ifs.seekg(std::streampos(0));
  }
  while (!(token = readWord(ifs)).empty()) {
    ntokens++;
    if (token == EOS) break;
    int32_t wid = getId(token);
    if (wid < 0) continue;
    int8_t type = getType(wid);
    if (type == 0 && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (type == 1) {
      labels.push_back(wid - nwords);
    }
    if (words.size() > MAX_LINE_SIZE && args.model != model_name::sup) break;
  }
  return ntokens;
}

void Dictionary::save(std::ofstream& ofs) {
  char buffer[1024];
  char ender = 0;
  int k = 0;
  if (ofs.is_open()) {
    ofs.write((char*) &size, sizeof(int32_t));
    ofs.write((char*) &nwords, sizeof(int32_t));
    ofs.write((char*) &nlabels, sizeof(int32_t));
    ofs.write((char*) &ntokens, sizeof(int64_t));
    for (int32_t i = 0; i < size; i++) {
      entry e = words_[i];
      std::wstring w = e.word;
      for (int c = 0; c < w.size(); c++) {
        k = wctomb(buffer, w[c]);
        ofs.write(buffer, k);
      }
      ofs.write(&ender, 1);
      ofs.write((char*) &(e.id), sizeof(int32_t));
      ofs.write((char*) &(e.uf), sizeof(int64_t));
      ofs.write((char*) &(e.type), sizeof(int8_t));
      int32_t nHashes = e.hashes.size();
      ofs.write((char*) &nHashes, sizeof(int32_t));
      for (int32_t hash : e.hashes) {
        ofs.write((char*) &(hash), sizeof(int32_t));
      }
    }
  }
}

void Dictionary::load(std::ifstream& ifs) {
  char buffer[1024];
  wchar_t wbuffer[1024];
  words_.clear();
  if (ifs.is_open()) {
    ifs.read((char*) &size, sizeof(int32_t));
    ifs.read((char*) &nwords, sizeof(int32_t));
    ifs.read((char*) &nlabels, sizeof(int32_t));
    ifs.read((char*) &ntokens, sizeof(int64_t));
    for (int32_t j = 0; j < size; j++) {
      char c = 1;
      int i = 0;
      while (c != 0) {
        ifs.read(&c, 1);
        buffer[i] = c;
        i++;
      }
      mbstowcs(wbuffer, buffer, 1024);
      std::wstring currentWord(wbuffer);
      entry we;
      we.word = currentWord;
      ifs.read((char*) &we.id, sizeof(int32_t));
      ifs.read((char*) &we.uf, sizeof(int64_t));
      ifs.read((char*) &we.type, sizeof(int8_t));
      int32_t nHashes = 0;
      ifs.read((char*) &nHashes, sizeof(int32_t));
      int32_t hash = 0;
      for (int z = 0; z < nHashes; z++) {
        ifs.read((char*) &hash, sizeof(int32_t));
        we.hashes.push_back(hash);
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
