---
id: aligned-vectors
title: Aligned word vectors
---

We are publishing aligned word vectors for 44 languages based on the pre-trained vectors computed on [*Wikipedia*](https://www.wikipedia.org) using fastText.
The alignments are performed with the RCSLS method described in [*Joulin et al (2018)*](https://arxiv.org/abs/1804.07745).

### Vectors

The aligned vectors can be downloaded from:

|||||
|-|-|-|-|
| Afrikaans: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.af.align.vec) | Arabic: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ar.align.vec) | Bulgarian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.bg.align.vec) | Bengali: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.bn.align.vec) |
| Bosnian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.bs.align.vec) | Catalan: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ca.align.vec) | Czech: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.cs.align.vec) | Danish: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.da.align.vec) |
| German: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.de.align.vec) | Greek: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.el.align.vec) | English: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec) | Spanish: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.es.align.vec) |
| Estonian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.et.align.vec) | Persian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fa.align.vec) | Finnish: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fi.align.vec) | French: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fr.align.vec) |
| Hebrew: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.he.align.vec) | Hindi: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hi.align.vec) | Croatian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hr.align.vec) | Hungarian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hu.align.vec) |
| Indonesian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.id.align.vec) | Italian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.it.align.vec) | Korean: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ko.align.vec) | Lithuanian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.lt.align.vec) |
| Latvian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.lv.align.vec) | Macedonian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.mk.align.vec) | Malay: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ms.align.vec) | Dutch: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.nl.align.vec) |
| Norwegian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.no.align.vec) | Polish: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.pl.align.vec) | Portuguese: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.pt.align.vec) | Romanian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ro.align.vec) |
| Russian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ru.align.vec) | Slovak: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.sk.align.vec) | Slovenian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.sl.align.vec) | Albanian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.sq.align.vec) |
| Swedish: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.sv.align.vec) | Tamil: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ta.align.vec) | Thai: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.th.align.vec) | Tagalog: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.tl.align.vec) |
| Turkish: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.tr.align.vec) | Ukrainian: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.uk.align.vec) | Vietnamese: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.vi.align.vec) | Chinese: [*text*](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.zh.align.vec) |

### Format

The word vectors come in the default text format of fastText.
The first line gives the number of vectors and their dimension.
The other lines contain a word followed by its vector. Each value is space separated.

### License

The word vectors are distributed under the [*Creative Commons Attribution-Share-Alike License 3.0*](https://creativecommons.org/licenses/by-sa/3.0/).

### References

If you use these word vectors, please cite the following papers:

[1] A. Joulin, P. Bojanowski, T. Mikolov, H. Jegou, E. Grave, [*Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion*](https://arxiv.org/abs/1804.07745)

```markup
@InProceedings{joulin2018loss,
  title={Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion},
  author={Joulin, Armand and Bojanowski, Piotr and Mikolov, Tomas and J\'egou, Herv\'e and Grave, Edouard},
  year={2018},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
}
```

[2] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```markup
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```
