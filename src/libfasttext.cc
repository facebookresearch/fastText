/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>
#include <sstream>

#include <string.h>

#include "args.h"
#include "fasttext.h"

using namespace fasttext;

/**
 * main.cc::test
 *
 * @access public
 * @param  const char* filename
 * @param  const char* word
 * @param  const int k
 * @param  char* result
 * @return int
 */
int FastTextTest(const char* filename, const char* word, const int k, char* result) {
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf strBuf(word);
    std::istream ifs(&strBuf);

    std::stringbuf  buf;
    std::streambuf* prev = std::cout.rdbuf(&buf);

    fasttext.test(ifs, k);

    std::cout.rdbuf(prev);

    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::predict
 *
 * @access public
 * @param  const char* filename
 * @param  const char* word
 * @param  const int k
 * @param  char* result
 * @return int
 */
int FastTextPredict(const char* filename, const char* word, const int k, char* result) {
    bool print_prob = false;
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf strBuf(word);
    std::istream ifs(&strBuf);

    std::stringbuf  buf;
    std::streambuf* prev = std::cout.rdbuf(&buf);

    fasttext.predict(ifs, k, print_prob);

    std::cout.rdbuf(prev);

    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::predict
 *
 * @access public
 * @param  const char* filename
 * @param  const char* word
 * @param  const int k
 * @param  char* result
 * @return int
 */
int FastTextPredictProb(const char* filename, const char* word, const int k, char* result) {
    bool print_prob = true;
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf strBuf(word);
    std::istream ifs(&strBuf);

    std::stringbuf  buf;
    std::streambuf* prev = std::cout.rdbuf(&buf);

    fasttext.predict(ifs, k, print_prob);

    std::cout.rdbuf(prev);

    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::printWordVectors
 *
 * @access public
 * @param  const char* filename
 * @param  const char* word
 * @param  char* result
 * @return int
 */
int FastTextPrintWordVectors(const char* filename, const char* word, char* result) {
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf  buf;
    std::streambuf* prevout = std::cout.rdbuf(&buf);

    std::stringbuf strBuf(word);
    std::streambuf* previn = std::cin.rdbuf(&strBuf);

    fasttext.printWordVectors();

    std::cin.rdbuf(previn);
    std::cout.rdbuf(prevout);
    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::printSentenceVectors
 *
 * @access public
 * @param  const char* filename
 * @param  const char* word
 * @param  char* result
 * @return int
 */
int FastTextPrintSentenceVectors(const char* filename, const char* word, char* result) {
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf  buf;
    std::streambuf* prevout = std::cout.rdbuf(&buf);

    std::stringbuf strBuf(word);
    std::streambuf* previn = std::cin.rdbuf(&strBuf);

    fasttext.printSentenceVectors();

    std::cin.rdbuf(previn);
    std::cout.rdbuf(prevout);
    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::printNgrams
 *
 * @access public
 * @param  const char* filename
 * @param  const char* word
 * @param  char* result
 * @return int
 */
int FastTextPrintNgrams(const char* filename, const char* word, char* result) {
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf  buf;
    std::streambuf* prevout = std::cout.rdbuf(&buf);

    fasttext.ngramVectors(std::string(word));

    std::cout.rdbuf(prevout);
    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::printNgrams
 *
 * @access public
 * @param  const char* filename
 * @param  const int k
 * @param  char* result
 * @return int
 */
int FastTextNN(const char* filename, const int k, char* result) {
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf  buf;
    std::streambuf* prevout = std::cout.rdbuf(&buf);

    fasttext.nn(k);

    std::cout.rdbuf(prevout);
    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::analogies
 *
 * @access public
 * @param  const char* filename
 * @param  const int k
 * @param  char* result
 * @return int
 */
int FastTextAnalogies(const char* filename, const int k, char* result) {
    FastText fasttext;
    fasttext.loadModel(filename);

    std::stringbuf  buf;
    std::streambuf* prevout = std::cout.rdbuf(&buf);

    fasttext.analogies(k);

    std::cout.rdbuf(prevout);
    strcpy(result, buf.str().c_str());

    return FASTTEXT_TRUE;
}

/**
 * main.cc::train
 *
 * @access public
 * @param  int argc
 * @param  char** argv
 * @return int
 */
int FastTextTrain(int argc, char** argv) {
    std::shared_ptr<Args> a = std::make_shared<Args>();
    a->parseArgs(argc, argv);
    FastText fasttext;
    fasttext.train(a);

    return FASTTEXT_TRUE;
}

/**
 * FastTextSupervised
 *
 * @access public
 * @param  const char* inputfile
 * @param  const char* outputfile
 * @param  const int dim
 * @param  const double lr
 * @param  const int wordNgrams
 * @param  const int minCount
 * @param  const int bucket
 * @return int
 */
int FastTextSupervised(const char* inputfile, const char* outputfile, const int dim,
                       const double lr, const int wordNgrams, const int minCount, const int bucket) {
    int argc = 8;
    char argv[8][1024];

    snprintf(argv[0], 1024, "supervised");
    snprintf(argv[1], 1024, "-input %s", inputfile);
    snprintf(argv[2], 1024, "-output %s", outputfile);
    snprintf(argv[3], 1024, "-dim %d", dim);
    snprintf(argv[4], 1024, "-lr %f", lr);
    snprintf(argv[5], 1024, "-wordNgrams %d", wordNgrams);
    snprintf(argv[6], 1024, "-minCount %d", minCount);
    snprintf(argv[7], 1024, "-bucket %d", bucket);

    return  FastTextTrain(argc, (char**)argv);
}

/**
 * FastTextSkipgram
 *
 * @access public
 * @param  const char* inputfile
 * @param  const char* outputfile
 * @param  const int dim
 * @param  const double lr
 * @param  const int wordNgrams
 * @param  const int minCount
 * @param  const int bucket
 * @return int
 */
int FastTextSkipgram(const char* inputfile, const char* outputfile, const int dim,
                     const double lr, const int wordNgrams, const int minCount, const int bucket) {
    int argc = 8;
    char argv[8][1024];

    snprintf(argv[0], 1024, "skipgram");
    snprintf(argv[1], 1024, "-input %s", inputfile);
    snprintf(argv[2], 1024, "-output %s", outputfile);
    snprintf(argv[3], 1024, "-dim %d", dim);
    snprintf(argv[4], 1024, "-lr %f", lr);
    snprintf(argv[5], 1024, "-wordNgrams %d", wordNgrams);
    snprintf(argv[6], 1024, "-minCount %d", minCount);
    snprintf(argv[7], 1024, "-bucket %d", bucket);

    return  FastTextTrain(argc, (char**)argv);
}

/**
 * FastTextCbow
 *
 * @access public
 * @param  const char* inputfile
 * @param  const char* outputfile
 * @param  const int dim
 * @param  const double lr
 * @param  const int wordNgrams
 * @param  const int minCount
 * @param  const int bucket
 * @return int
 */
int FastTextCbow(const char* inputfile, const char* outputfile, const int dim,
                 const double lr, const int wordNgrams, const int minCount, const int bucket) {
    int argc = 8;
    char argv[8][1024];

    snprintf(argv[0], 1024, "cbow");
    snprintf(argv[1], 1024, "-input %s", inputfile);
    snprintf(argv[2], 1024, "-output %s", outputfile);
    snprintf(argv[3], 1024, "-dim %d", dim);
    snprintf(argv[4], 1024, "-lr %f", lr);
    snprintf(argv[5], 1024, "-wordNgrams %d", wordNgrams);
    snprintf(argv[6], 1024, "-minCount %d", minCount);
    snprintf(argv[7], 1024, "-bucket %d", bucket);

    return  FastTextTrain(argc, (char**)argv);
}
