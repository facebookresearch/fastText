## Alignment of Word Embeddings

This directory provides code for learning alignments between word embeddings in different languages.

The code is in Python 3 and requires [NumPy](http://www.numpy.org/).

The script `example.sh` shows how to use this code to learn and evaluate a bilingual alignment of word embeddings.

The word embeddings used in [1] can be found on the [fastText project page](https://fasttext.cc) and the supervised bilingual lexicons on the [MUSE project page](https://github.com/facebookresearch/MUSE).

### Supervised alignment

The script `align.py` aligns word embeddings from two languages using a bilingual lexicon as supervision.
The details of this approach can be found in [1].

### Unsupervised alignment

The script `unsup_align.py` aligns word embeddings from two languages without requiring any supervision.
Additionally, the script `unsup_multialign.py` aligns multiple languages to a common space with no supervision.
The details of these approaches can be found in [2] and [3] respectively.

In addition to NumPy, the unsupervised methods require the [Python Optimal Transport](https://pot.readthedocs.io/en/stable/) toolbox.

### Download

Wikipedia fastText embeddings aligned with our method can be found [here](https://fasttext.cc/docs/en/aligned-vectors.html).

### References

If you use the supervised alignment method, please cite:

[1] A. Joulin, P. Bojanowski, T. Mikolov, H. Jegou, E. Grave, [*Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion*](https://arxiv.org/abs/1804.07745)

```
@InProceedings{joulin2018loss,
    title={Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion},
    author={Joulin, Armand and Bojanowski, Piotr and Mikolov, Tomas and J\'egou, Herv\'e and Grave, Edouard},
    year={2018},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
}
```

If you use the unsupervised bilingual alignment method, please cite:

[2] E. Grave, A. Joulin, Q. Berthet, [*Unsupervised Alignment of Embeddings with Wasserstein Procrustes*](https://arxiv.org/abs/1805.11222)

```
@article{grave2018unsupervised,
    title={Unsupervised Alignment of Embeddings with Wasserstein Procrustes},
    author={Grave, Edouard and Joulin, Armand and Berthet, Quentin},
    journal={arXiv preprint arXiv:1805.11222},
    year={2018}
}
```

If you use the unsupervised alignment script `unsup_multialign.py`, please cite:

[3] J. Alaux, E. Grave, M. Cuturi, A. Joulin, [*Unsupervised Hyperalignment for Multilingual Word Embeddings*](https://arxiv.org/abs/1811.01124)

```
@article{alaux2018unsupervised,
  title={Unsupervised hyperalignment for multilingual word embeddings},
  author={Alaux, Jean and Grave, Edouard and Cuturi, Marco and Joulin, Armand},
  journal={arXiv preprint arXiv:1811.01124},
  year={2018}
}
```
