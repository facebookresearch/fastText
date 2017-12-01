#ifndef FASTTEXT_LIBFASTTEXT_H
#define FASTTEXT_LIBFASTTEXT_H

#include <time.h>
#include <string.h>
#include <stdint.h>

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

typedef float FTReal;

struct _FTValues {
    int is_error;
    size_t len;
    char *buff;
    int size;
    char **vals;
};

struct _FTVectors {
    int is_error;
    size_t len;
    char *buff;
    int64_t size;
    FTReal *vals;
};

struct _FTProbs {
    int is_error;
    size_t len;
    char *buff;
    int size;
    char **labels;
    char **probs;
};

typedef void *FastTextHandle;
typedef struct _FTValues *FTValues;
typedef struct _FTVectors *FTVectors;
typedef struct _FTProbs *FTProbs;

FASTTEXT_API int FastTextVersion();
FASTTEXT_API FastTextHandle FastTextCreate();
FASTTEXT_API void FastTextFree(FastTextHandle handle);
FASTTEXT_API void FastTextValuesFree(FTValues handle);
FASTTEXT_API void FastTextVectorsFree(FTVectors handle);
FASTTEXT_API void FastTextProbsFree(FTProbs handle);

FASTTEXT_API int FastTextLoadModel(FastTextHandle handle, const char *filename);
FASTTEXT_API int32_t FastTextWordRows(FastTextHandle handle);
FASTTEXT_API int32_t FastTextLabelRows(FastTextHandle handle);
FASTTEXT_API int32_t FastTextWordId(FastTextHandle handle, const char* word);
FASTTEXT_API int32_t FastTextSubwordId(FastTextHandle handle, const char* word);
FASTTEXT_API FTValues FastTextGetWord(FastTextHandle handle, int32_t wordId);
FASTTEXT_API FTValues FastTextGetLabel(FastTextHandle handle, int32_t labelId);
FASTTEXT_API FTVectors FastTextWordVectors(FastTextHandle handle, const char* word);
FASTTEXT_API FTVectors FastTextSubwordVector(FastTextHandle handle, const char* word);
FASTTEXT_API FTVectors FastTextSentenceVectors(FastTextHandle handle, const char* sentence);
FASTTEXT_API FTProbs FastTextPredict(FastTextHandle handle, const char* word, const int k);
FASTTEXT_API FTProbs FastTextNN(FastTextHandle handle, const char* word, const int k);
FASTTEXT_API FTProbs FastTextAnalogies(FastTextHandle handle, const char* word, const int k);
FASTTEXT_API FTProbs FastTextNgramVectors(FastTextHandle handle, const char* word);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* FASTTEXT_LIBFASTTEXT_H */
