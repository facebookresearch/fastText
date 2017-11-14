#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <vector>

#include <string.h>

#include "args.h"
#include "fasttext.h"
#include "libfasttext.h"

using namespace fasttext;

FTValues _newValues(std::string word);
FTValues _errorValues(const char *error);
FTVectors _newVectors(int64_t size);
FTVectors _errorVectors(const char *error);
FTProbs _parseString(const char *buff, bool isSingle);
FTProbs _errorProbs(const char *error);
std::vector<std::pair<int, std::string>> _parseQuery(std::string query);

/**
 * get fastText version
 *
 * @access public
 * @return int
 */
int FastTextVersion()
{
    return FASTTEXT_VERSION;
}

/**
 * create a FastText handle
 *
 * @access public
 * @return FastTextHandle
 */
FastTextHandle FastTextCreate()
{
    FastTextHandle handle = new FastText;
    return handle;
}

/**
 * free a FastText handle
 *
 * @access public
 * @param  FastTextHandle handle
 * @return void
 */
void FastTextFree(FastTextHandle handle)
{
    delete static_cast<FastText*>(handle);
}

/**
 * free a FTValues handle
 *
 * @access public
 * @param  FastTextHandle handle
 * @return void
 */
void FastTextValuesFree(FTValues handle)
{
    if (NULL != handle->vals) {
        delete[] handle->vals;
    }
    delete[] handle->buff;
    delete handle;
}

/**
 * free a FTProbs handle
 *
 * @access public
 * @param  FTVectors handle
 * @return void
 */
void FastTextVectorsFree(FTVectors handle)
{
    delete[] handle->vals;
    delete handle;
}

/**
 * free a FTProbs handle
 *
 * @access public
 * @param  FTPredicts ptr
 * @return void
 */
void FastTextProbsFree(FTProbs handle)
{
    if (NULL != handle->probs) {
        delete[] handle->probs;
    }
    if (NULL != handle->labels) {
        delete[] handle->labels;
    }
    delete[] handle->buff;
    delete handle;
}

/**
 * fasttext.cc::loadModel
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* filename
 * @return int
 */
int FastTextLoadModel(FastTextHandle handle, const char *filename)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    try {
        fasttext->loadModel(filename);
    } catch (const std::invalid_argument& e) {
        return FASTTEXT_FALSE;
    }
    return FASTTEXT_TRUE;
}

/**
 * dictionary.cc::nwords
 *
 * @access public
 * @param  FastTextHandle handle
 * @return int32_t
 */
int32_t FastTextWordRows(FastTextHandle handle)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::shared_ptr<const Dictionary> dict = fasttext->getDictionary();

    return dict->nwords();
}

/**
 * dictionary.cc::nlabels
 *
 * @access public
 * @param  FastTextHandle handle
 * @return int32_t
 */
int32_t FastTextLabelRows(FastTextHandle handle)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::shared_ptr<const Dictionary> dict = fasttext->getDictionary();

    return dict->nlabels();
}

/**
 * fasttext.cc::getWordId
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @return int32_t
 */
int32_t FastTextWordId(FastTextHandle handle, const char* word)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::string text = std::string(word);

    return fasttext->getWordId(text);
}

/**
 * fasttext.cc::getSubwordId
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @return int32_t
 */
int32_t FastTextSubwordId(FastTextHandle handle, const char* word)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::string text = std::string(word);

    return fasttext->getSubwordId(text);
}

/**
 * dictionary.cc::getWord
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  conint32_tst wordId
 * @return FTValues
 */
FTValues FastTextGetWord(FastTextHandle handle, int32_t wordId)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::shared_ptr<const Dictionary> dict = fasttext->getDictionary();
    std::string word = dict->getWord(wordId);

    return _newValues(word);
}

/**
 * dictionary.cc::getLabel
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  conint32_tst wordId
 * @return FTValues
 */
FTValues FastTextGetLabel(FastTextHandle handle, int32_t labelId)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::shared_ptr<const Dictionary> dict = fasttext->getDictionary();
    std::string label = dict->getLabel(labelId);

    return _newValues(label);
}

/**
 * main.cc::printWordVectors
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @return FTPredicts*
 */
FTVectors FastTextWordVectors(FastTextHandle handle, const char* word)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::string text = std::string(word);

    Vector vec(fasttext->getDimension());
    fasttext->getWordVector(vec, text);

    FTVectors retval = _newVectors(vec.m_);
    for (int64_t j = 0; j < retval->size; j++) {
        retval->vals[j] = vec.data_[j];
    }
    return retval;
}

/**
 * main.cc::printWordVectors
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @return FTPredicts*
 */
FTVectors FastTextSubwordVector(FastTextHandle handle, const char* word)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::string text = std::string(word);

    Vector vec(fasttext->getDimension());
    fasttext->getSubwordVector(vec, text);

    FTVectors retval = _newVectors(vec.m_);
    for (int64_t j = 0; j < retval->size; j++) {
        retval->vals[j] = vec.data_[j];
    }
    return retval;
}

/**
 * main.cc::printSentenceVectors
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @return FTVectors
 */
FTVectors FastTextSentenceVectors(FastTextHandle handle, const char* word)
{
    FastText *fasttext = static_cast<FastText*>(handle);
    std::string text = std::string(word);
    std::stringstream ioss;
    std::copy(text.begin(), text.end(), std::ostream_iterator<char>(ioss));

    Vector svec(fasttext->getDimension());
    fasttext->getSentenceVector(ioss, svec);

    FTVectors retval = _newVectors(svec.m_);
    for (int64_t j = 0; j < retval->size; j++) {
        retval->vals[j] = svec.data_[j];
    }
    return retval;
}

/**
 * main.cc::predict
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @param  const int k
 * @return FTProbs
 */
FTProbs FastTextPredict(FastTextHandle handle, const char* word, const int k)
{
    FastText *fasttext = static_cast<FastText*>(handle);

    std::string text = std::string(word);
    std::stringstream ioss;
    std::copy(text.begin(), text.end(), std::ostream_iterator<char>(ioss));
    std::vector<std::pair<fasttext::real, std::string>> predictions;

    try {
        fasttext->predict(ioss, k, predictions);
    } catch (const std::invalid_argument& e) {
        return _errorProbs(e.what());
    }

    std::stringstream ss;
    for (auto n : predictions) {
        ss << n.second << " " << std::exp(n.first) << "\n";
    }

    return _parseString(ss.str().c_str(), true);
}

/**
 * main.cc::nn
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @param  const int k
 * @return FTVectors
 */
FTProbs FastTextNN(FastTextHandle handle, const char* word, const int k)
{
    FTProbs retval;
    FastText *fasttext = static_cast<FastText*>(handle);
    std::shared_ptr<const Dictionary> dict = fasttext->getDictionary();

    std::string queryWord = std::string(word);
    Vector queryVec(fasttext->getDimension());
    Matrix wordVectors(dict->nwords(), fasttext->getDimension());

    std::stringbuf buf;
    std::streambuf *prev = std::cerr.rdbuf(&buf);
    fasttext->precomputeWordVectors(wordVectors);
    std::cerr.rdbuf(prev);

    std::set<std::string> banSet;
    banSet.clear();
    banSet.insert(queryWord);
    fasttext->getWordVector(queryVec, queryWord);

    buf.str("");
    prev = std::cout.rdbuf(&buf);

    try {
        fasttext->findNN(wordVectors, queryVec, k, banSet);
        retval = _parseString(buf.str().c_str(), true);
    } catch (const std::invalid_argument& e) {
        retval = _errorProbs(e.what());
    }
    std::cout.rdbuf(prev);

    return retval;
}

/**
 * main.cc::analogies
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @param  const int k
 * @return FTVectors
 */
FTProbs FastTextAnalogies(FastTextHandle handle, const char* word, const int k)
{
    FTProbs retval;
    FastText *fasttext = static_cast<FastText*>(handle);
    std::shared_ptr<const Dictionary> dict = fasttext->getDictionary();

    Vector buffer(fasttext->getDimension()), query(fasttext->getDimension());
    Matrix wordVectors(dict->nwords(), fasttext->getDimension());

    std::stringbuf buf;
    std::streambuf *prev = std::cerr.rdbuf(&buf);
    fasttext->precomputeWordVectors(wordVectors);
    std::cerr.rdbuf(prev);

    std::set<std::string> banSet;
    banSet.clear();
    query.zero();

    std::vector<std::pair<int, std::string>> queries = _parseQuery(std::string(word));
    for (auto n : queries) {
        banSet.insert(n.second);
        fasttext->getWordVector(buffer, n.second);
        query.addVector(buffer, 1.0 * n.first);
    }

    buf.str("");
    prev = std::cout.rdbuf(&buf);

    try {
        fasttext->findNN(wordVectors, query, k, banSet);
        retval = _parseString(buf.str().c_str(), true);
    } catch (const std::invalid_argument& e) {
        retval = _errorProbs(e.what());
    }
    std::cout.rdbuf(prev);

    return retval;
}

/**
 * main.cc::printNgrams
 *
 * @access public
 * @param  FastTextHandle handle
 * @param  const char* word
 * @return FTProbs
 */
FTProbs FastTextNgramVectors(FastTextHandle handle, const char* word)
{
    FTProbs retval;
    FastText *fasttext = static_cast<FastText*>(handle);

    std::stringbuf buf;
    std::streambuf *prev = std::cout.rdbuf(&buf);

    try {
        fasttext->ngramVectors(std::string(word));
        retval = _parseString(buf.str().c_str(), false);
    } catch (const std::invalid_argument& e) {
        retval = _errorProbs(e.what());
    }
    std::cout.rdbuf(prev);

    return retval;
}

/**
 * create struct _FTValues
 *
 * @access private
 * @param  std::string word
 * @return FTValues
 */
FTValues _newValues(std::string word)
{
    FTValues val = new struct _FTValues;

    val->is_error = FASTTEXT_FALSE;
    val->len = word.length();
    val->buff = new char[val->len + 1];
    strcpy(val->buff, word.c_str());

    val->size = 1;
    val->vals = NULL;

    return val;
}

/**
 * set error message
 *
 * @access private
 * @param  const char* error
 * @return FTValues
 */
FTValues _errorValues(const char *error)
{
    FTValues val = new struct _FTValues;

    val->is_error = FASTTEXT_TRUE;
    val->len = strlen(error);
    val->buff = new char[val->len + 1];
    strcpy(val->buff, error);

    val->vals = NULL;

    return val;
}

/**
 * create struct _FTVectors
 *
 * @access private
 * @param  int64_t size
 * @return FTVectors
 */
FTVectors _newVectors(int64_t size)
{
    FTVectors val = new struct _FTVectors;
    val->is_error = FASTTEXT_FALSE;
    val->len = 0;
    val->buff = NULL;

    val->size = size;
    val->vals = new FTReal[val->size];

    return val;
}

/**
 * set error message
 *
 * @access private
 * @param  const char* error
 * @return FTVectors
 */
FTVectors _errorVectors(const char *error)
{
    FTVectors val = new struct _FTVectors;
    val->is_error = FASTTEXT_TRUE;
    val->len = strlen(error);
    val->buff = new char[val->len + 1];
    strcpy(val->buff, error);

    val->vals = NULL;

    return val;
}

/**
 * count line breaks
 *
 * @access private
 * @param  FTProbs vals
 * @return FTVectors
 */
int _countLine(FTProbs vals)
{
    int count = 0;
    for (size_t idx=0; idx < vals->len; idx++) {
        count = (vals->buff[idx] == '\n')? count + 1 : count;
    } // for (size_t idx=0; idx < len; idx++)

    return (count == 0) ? 1 : count;
}

/**
 * parse a output format
 *
 * @access private
 * @param  const char* word
 * @param  bool isSingle
 * @return FTVectors
 */
FTProbs _parseString(const char *buff, bool isSingle)
{
    FTProbs val = new struct _FTProbs;
    val->is_error = FASTTEXT_FALSE;
    val->len = strlen(buff);
    val->buff = new char[val->len + 1];
    strcpy(val->buff, buff);

    val->size = _countLine(val);
    val->probs = new char*[val->size];
    val->labels = new char*[val->size];

    int curr = 0;
    val->labels[curr] = val->buff;
    bool separator = true;

    for (size_t idx=0; idx < val->len; idx++) {
        switch(val->buff[idx]) {
            case ' ':
                if (separator) {
                    val->buff[idx] = '\0';
                    separator = !isSingle;
                }
                val->probs[curr] = val->buff + idx + 1;
                break;
            case '\n':
                val->buff[idx] = '\0';
                curr++;
                if (curr < val->size) {
                    val->labels[curr] = val->buff + idx + 1;
                    separator = true;
                }
                break;
        }
    } // for (size_t idx=0; idx < len; idx++)

    return val;
}

/**
 * set error message
 *
 * @access private
 * @param  const char* error
 * @return FTProbs
 */
FTProbs _errorProbs(const char *error)
{
    FTProbs val = new struct _FTProbs;
    val->is_error = FASTTEXT_TRUE;
    val->len = strlen(error);
    val->buff = new char[val->len + 1];
    strcpy(val->buff, error);

    val->probs = NULL;
    val->labels = NULL;

    return val;
}

/**
 * parse a query format
 *
 * @access private
 * @param  std::string query
 * @return std::vector<std::pair<int, std::string>>
 */
std::vector<std::pair<int, std::string>> _parseQuery(std::string query)
{
    std::vector<std::pair<int, std::string>> val;
    int sign = 1, left = 0, len=0;
    bool flag = true, step = true;

    val.clear();

    size_t maxlen = query.length();
    for (size_t idx=0; idx < maxlen; idx++) {
        switch (query[idx]) {
            case ' ':
                if (flag) {
                    len = idx - left;
                    val.push_back(
                        std::make_pair(
                            sign,
                            query.substr(left, len)
                        )
                    );
                    flag = false;
                    step = true;
                } // if (flag)
                break;
            case '+':
                sign = 1;
                break;
            case '-':
                sign = -1;
                break;
            default:
                flag = true;
                if (step) {
                    left = idx;
                    step = false;
                }
                break;
        }
    }
    val.push_back(
        std::make_pair(sign, query.substr(left))
    );
    return val;
}
