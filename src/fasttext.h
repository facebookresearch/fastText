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
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;
    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;
    std::shared_ptr<Model> model_;
    std::atomic<int64_t> tokenCount;
    clock_t start;

  public:
    void getVector(Vector&, const std::string&);
    void saveVectors();
    void saveOutput();
    void saveModel();
    void loadModel(const std::string&);
    void loadModel(std::istream&);
    void printInfo(real, real);

    void supervised(Model&, real, const std::vector<int32_t>&,
                    const std::vector<int32_t>&);
    void cbow(Model&, real, const std::vector<int32_t>&);
    void skipgram(Model&, real, const std::vector<int32_t>&);
    void test(std::istream&, int32_t);
    void predict(std::istream&, int32_t, bool);
    void predict(std::istream&, int32_t, std::vector<std::pair<real,std::string>>&) const;
    void wordVectors();
    void ngramVectors(std::string);
    void textVectors();
    void printVectors();
    void trainThread(int32_t);
    void train(std::shared_ptr<Args>);

    void loadVectors(std::string);
};

}

#endif
