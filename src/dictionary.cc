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

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args, int maxSectionType)
  : maxSectionType_(maxSectionType)
{
  args_ = args;
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  ntokens_ = 0;

  dataSeparator_ = args_->separator;
  for(char& c : dataSeparator_) {
    if(dataSeparatorChars_.find(c) == dataSeparatorChars_.end()) {
      dataSeparatorChars_[c] = 1;
    }
  }
  
  word2int_.resize(MAX_VOCAB_SIZE);
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
}

int32_t Dictionary::find(const std::string& w) const {
  int32_t h = hash(w) % MAX_VOCAB_SIZE;
  while (word2int_[h] != -1 && words_[word2int_[h]].word != w) {
    h = (h + 1) % MAX_VOCAB_SIZE;
  }
  return h;
}

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }
}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getNgrams(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getNgrams(const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getNgrams(i);
  }
  std::vector<int32_t> ngrams;
  computeNgrams(BOW + word + EOW, ngrams);
  return ngrams;
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  if(id >= nwords_) { std::cout<<id<<" , "<<nwords_<<std::endl; }
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeNgrams(const std::string& word,
                               std::vector<int32_t>& ngrams) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);
      }
    }
  }
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.push_back(i);
    computeNgrams(word, words_[i].subwords);
  }
}

int Dictionary::readWord(std::istream& in, std::string& word) const
{
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  std::string tmp;

  // streambuf::sbumpc : Returns char at current pos of controlled
  //    input sequence, and advances pos indicator to next char.
  // Read istream until end of file EOF char is reached.  Inside while loop, if
  // char is a special character, wrap up and return to the caller.  If word is
  // empty, then if current character is a new line, then add special EOS
  // symbol, otherwise continue.  If word is not empty, we want to return the
  // word, just making sure before that we don't add the current character if
  // it's a new line.
  while ((c = sb.sbumpc()) != EOF) {
    // \n : line feed ; \r : carriage return ; \t : horizontal tab ;
    // \v : vertical tab ; \f : formfeed ; \0 : null char
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f' || c == '\0') {
      
      if (word.empty()) {
        if (c == '\n') {
          word += EOS; // Special character from class Dictionary to make sure
		       // classifier knows we reached end of sentence.
          return WORD_READ;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return WORD_READ;
      }
    }

    // Check if c is any of the characters in dataSeparator string.
    if (dataSeparatorChars_.find(c) != dataSeparatorChars_.end()) {
      tmp.push_back(c);
      if(tmp == dataSeparator_) {
	return DATA_SEPARATOR_DETECTED;
      }
    }
    else {
      tmp.clear();
      word.push_back(c);
    }
  }
  
  // trigger eofbit
  in.get();
  return !word.empty() ? 1 : EOF_DETECTED;
}

void Dictionary::toEndOfLine(std::istream& in) const {
  char c;
  std::streambuf& sb = *in.rdbuf();
  while((c = sb.sbumpc()) != EOF && c != '\n' && c != '\r') { }
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  int currentType = 0;
  int r;
  while ((r = readWord(in, word)) != EOF_DETECTED) {
    
    add(word); // increases ntokens_, adds word to dictionary after hashing it.
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cout << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
    }
    // size_ of dictionary
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      // compacts dictionary and sorts it up
      threshold(minThreshold, minThreshold);
    }

    // Check if we changed column in input file. If we have less
    // granularities than columns, we shouldn't read all columns.
    if(r == DATA_SEPARATOR_DETECTED) {
      currentType++;

      if(currentType > args_->granularities) {
	currentType = 0;
	
	if(args_->granularities < maxSectionType_) {
	  toEndOfLine(in);
	}
      }
    }
  }
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cout << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
    std::cout << "Number of words:  " << nwords_ << std::endl;
    std::cout << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    std::cerr << "Empty vocabulary. Try a smaller -minCount value." << std::endl;
    exit(EXIT_FAILURE);
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.count > e2.count;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
        return (e.type == entry_type::word && e.count < t) ||
               (e.type == entry_type::label && e.count < tl);
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
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) counts.push_back(w.count);
  }
  return counts;
}

void Dictionary::addNgrams(std::vector<int32_t>& line, int32_t n) const {
  int32_t line_size = line.size();
  for (int32_t i = 0; i < line_size; i++) {
    uint64_t h = line[i];
    for (int32_t j = i + 1; j < line_size && j < i + n; j++) {
      h = h * 116049371 + line[j];
      line.push_back(nwords_ + (h % args_->bucket));
    }
  }
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::vector<int32_t>& labels,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;
  words.clear();
  labels.clear();
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
  while (readWord(in, token)) {
    int32_t wid = getId(token);
    if (wid < 0) continue;
    entry_type type = getType(wid);
    ntokens++;
    if (type == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (type == entry_type::label) {
      labels.push_back(wid - nwords_);
    }
    if (words.size() > MAX_LINE_SIZE && args_->model != model_name::sup) break;
    if (token == EOS) break;
  }
  return ntokens;
}
  
int32_t Dictionary::getLine(std::istream& in,
			    VPtrVector& granularities,
                            std::vector<int32_t>& labels,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  labels.clear();
  for(std::vector<int32_t>* v : granularities) {
    v->clear();
  }

  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }

  int currentType = 0;
  int r;
  while ((r = readWord(in, token)) != EOF_DETECTED) {
    int32_t wid = getId(token);
    if (wid < 0) continue;
    ntokens++;

    entry_type type = getType(wid);
    if(currentType == 0 || type == entry_type::label) {
      labels.push_back(wid - nwords_);
    } else {
      if(!discard(wid, uniform(rng))) {
	granularities[currentType - 1]->push_back(wid);
      }
    }

    bool anyMaxedVector = false;
    for(std::vector<int32_t>* v : granularities) {
      anyMaxedVector = anyMaxedVector || v->size() > MAX_LINE_SIZE;
    }
    if (anyMaxedVector && args_->model != model_name::sup) break;
    if (token == EOS) break;

    if(r == DATA_SEPARATOR_DETECTED) {
      currentType++;

      // make sure that I also got a new line / data is properly formatted
      if(currentType > granularities.size()) {
	currentType = 0;

	// input data normally has all 3 granularities.  If we don't want to use
	// the 3, then we need to move the pointer in the in istream to the end of
	// the current line.
	if(args_->granularities < maxSectionType_) {
	  toEndOfLine(in);
	  break;
	}
      }
    }
  }
  return ntokens;
}

std::string Dictionary::getLabel(int32_t lid) const {
  assert(lid >= 0);
  assert(lid < nlabels_);
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*) &size_, sizeof(int32_t));
  out.write((char*) &nwords_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*) &(e.count), sizeof(int64_t));
    out.write((char*) &(e.type), sizeof(entry_type));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  in.read((char*) &size_, sizeof(int32_t));
  in.read((char*) &nwords_, sizeof(int32_t));
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*) &e.count, sizeof(int64_t));
    in.read((char*) &e.type, sizeof(entry_type));
    words_.push_back(e);
    word2int_[find(e.word)] = i;
  }
  initTableDiscard();
  initNgrams();
}

}
