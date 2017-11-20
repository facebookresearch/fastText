#include <stdio.h>
#include <stdlib.h>

#include "libfasttext.h"

int main(int argc, char **argv)
{
#if 0
    const char* filename = "./result/dj2.bin";
    const char* word = "イベント";
    int k=10;

    FastTextHandle fasttext;
    FTValues values;

    fasttext = FastTextCreate();
    FastTextLoadModel(fasttext, filename);

    values = FastTextPredict(fasttext, word, k);

    for (size_t curr=0; curr < values->size; curr++) {
        printf("%s:%s\n", values->labels[curr], values->probs[curr]);
    }
    FastTextValuesFree(values);

    FastTextFree(fasttext);
#endif

#if 0
    const char* filename = "./result/wvec.bin";
    const char* word = "ピント";
    int k=10;

    FastTextHandle fasttext;
    FTValues list;

    fasttext = FastTextCreate();
    FastTextLoadModel(fasttext, filename);

    list = FastTextNN(fasttext, word, k);

    for (int curr=0; curr < list->size; curr++) {
        printf("[%s]:%s\n", list->labels[curr], list->probs[curr]);
    }

    FastTextValuesFree(list);
    FastTextFree(fasttext);
#endif

#if 1
    const char* filename = "./result/wvec.bin";
    const char* word = "イベント + ピンク - オフホワイト";
    int k=10;

    FastTextHandle fasttext;
    FTValues list;

    fasttext = FastTextCreate();
    FastTextLoadModel(fasttext, filename);

    list = FastTextAnalogies(fasttext, word, k);

    for (int curr=0; curr < list->size; curr++) {
        printf("[%s]:%s\n", list->labels[curr], list->probs[curr]);
    }

    FastTextValuesFree(list);
    FastTextFree(fasttext);
#endif

    return 0;
}
