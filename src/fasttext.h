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

#define FASTTEXT_VERSION 12 /* Version 1b */
#define FASTTEXT_FILEFORMAT_MAGIC_INT32 793712314

#include <time.h>

#ifdef __cplusplus

#include <atomic>
#include <memory>
#include <set>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "qmatrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class FastText {
  private:
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;

    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;

    std::shared_ptr<QMatrix> qinput_;
    std::shared_ptr<QMatrix> qoutput_;

    std::shared_ptr<Model> model_;

    std::atomic<int64_t> tokenCount;
    clock_t start;
    void signModel(std::ostream&);
    bool checkModel(std::istream&);

    bool quant_;
    int32_t version;

  public:
    FastText();

    void getVector(Vector&, const std::string&) const;
    std::shared_ptr<const Dictionary> getDictionary() const;
    void saveVectors();
    void saveOutput();
    void saveModel();
    void loadModel(std::istream&);
    void loadModel(const std::string&);
    void printInfo(real, real);

    void supervised(Model&, real, const std::vector<int32_t>&,
                    const std::vector<int32_t>&);
    void cbow(Model&, real, const std::vector<int32_t>&);
    void skipgram(Model&, real, const std::vector<int32_t>&);
    std::vector<int32_t> selectEmbeddings(int32_t) const;
    void quantize(std::shared_ptr<Args>);
    void test(std::istream&, int32_t);
    void predict(std::istream&, int32_t, bool);
    void predict(
        std::istream&,
        int32_t,
        std::vector<std::pair<real, std::string>>&) const;
    void wordVectors();
    void sentenceVectors();
    void ngramVectors(std::string);
    void textVectors();
    void printWordVectors();
    void printSentenceVectors();
    void precomputeWordVectors(Matrix&);
    void findNN(const Matrix&, const Vector&, int32_t,
                const std::set<std::string>&);
    void nn(int32_t);
    void analogies(int32_t);
    void trainThread(int32_t);
    void train(std::shared_ptr<Args>);

    void loadVectors(std::string);
    int getDimension() const;
};

}

extern "C" {
#endif /* __cplusplus */

#ifndef FASTTEXT_API
#   if defined(_WIN32) || defined(_WIN64)
#       define FASTTEXT_API __declspec(dllimport)
#   else
#       define FASTTEXT_API extern
#   endif /* defined(_WIN32) || defined(_WIN64) */
#endif /* FASTTEXT_API */

#define FASTTEXT_TRUE           (1)
#define FASTTEXT_FALSE          (0)

FASTTEXT_API int FastTextTest(const char* filename, const char* word, const int k, char* result);
FASTTEXT_API int FastTextPredict(const char* filename, const char* word, const int k, char* result);
FASTTEXT_API int FastTextPredictProb(const char* filename, const char* word, const int k, char* result);
FASTTEXT_API int FastTextPrintWordVectors(const char* filename, const char* word, char* result);
FASTTEXT_API int FastTextPrintSentenceVectors(const char* filename, const char* word, char* result);
FASTTEXT_API int FastTextPrintNgrams(const char* filename, const char* word, char* result);
FASTTEXT_API int FastTextNN(const char* filename, const int k, char* result);
FASTTEXT_API int FastTextAnalogies(const char* filename, const int k, char* result);
FASTTEXT_API int FastTextTrain(int argc, char** argv);

FASTTEXT_API int FastTextSupervised(const char* inputfile, const char* outputfile, const int dim,
                                    const double lr, const int wordNgrams, const int minCount, const int bucket);
FASTTEXT_API int FastTextSkipgram(const char* inputfile, const char* outputfile, const int dim,
                                  const double lr, const int wordNgrams, const int minCount, const int bucket);
FASTTEXT_API int FastTextCbow(const char* inputfile, const char* outputfile, const int dim,
                              const double lr, const int wordNgrams, const int minCount, const int bucket);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* FASTTEXT_FASTTEXT_H */
