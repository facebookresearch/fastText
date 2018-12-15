## Supervised Alignment of Word Embeddings

This code aligns word embeddings from two languages with a bilingual lexicon. The details of our approach can be found in [1].

The code is in Python 3 and requires [NumPy](http://www.numpy.org/).

The script `example.sh` shows how to use this code to learn and evaluate a bilingual alignment of word embeddings.

The word embeddings used in [1] can be found on the [fastText project page](https://fasttext.cc) and the supervised bilingual lexicons on the [MUSE project page](https://github.com/facebookresearch/MUSE).

### Download

Wikipedia fastText embeddings aligned with our method can be found [here](https://fasttext.cc/doc/en/aligned_vectors.html).

### References

If you use this code, please cite:

[1] A. Joulin, P. Bojanowski, T. Mikolov, H. Jegou, E. Grave, [*Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion*](https://arxiv.org/abs/1804.07745)

```
@InProceedings{joulin2018loss,
    title={Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion},
    author={Joulin, Armand and Bojanowski, Piotr and Mikolov, Tomas and J\'egou, Herv\'e and Grave, Edouard},
    year={2018},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
}
```
