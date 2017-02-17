/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_FASTTEXT_H
#define FASTTEXT_FASTTEXT_H

#include <time.h>

#include <atomic>
#include <memory>

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"

namespace fasttext {

class FastText {
 private:
  static const unsigned short maxGranularities = std::numeric_limits<unsigned short>::max();
  int granularityDimension;
  int64_t infileNbOfRows;
  
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<Matrix> input_;  // dict size + nb of buckets  x  size of category
  std::shared_ptr<Matrix> output_; // if supervised: nb of labels  x  size of hidden
  std::shared_ptr<Model> model_;
  std::atomic<int64_t> tokenCount;
  clock_t start;

 public:
  FastText(int);
  
  void getVector(Vector&, const std::string&);
  void saveVectors();
  void saveModel();
  void loadModel(const std::string&);
  void loadModel(std::istream&);
  void printInfo(real, real);
  
  void supervised(Model&, real, const std::vector<int32_t>&,
		  const std::vector<int32_t>&);
  void supervised(Model&, real, const List&,
		  const std::vector<int32_t>&);
  void cbow(Model&, real, const std::vector<int32_t>&);
  void skipgram(Model&, real, const std::vector<int32_t>&);
  void test(std::istream&, int32_t, int32_t granularityAmt = 1);
  void predict(std::istream&, int32_t, bool, int32_t granularity = 1);
  void predict(std::istream&, int32_t, int32_t, std::vector<std::pair<real,std::string>>&) const;
  void wordVectors();
  void textVectors();
  void printVectors();
  void trainThread(int32_t);
  void train(std::shared_ptr<Args>);
  
  void loadVectors(std::string);
};

}

#endif
