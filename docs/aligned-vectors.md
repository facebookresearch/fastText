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
| Afrikaans: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.af.align.vec) | Arabic: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ar.align.vec) | Bulgarian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.bg.align.vec) | Bengali: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.bn.align.vec) |
| Bosnian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.bs.align.vec) | Catalan: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ca.align.vec) | Czech: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.cs.align.vec) | Danish: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.da.align.vec) |
| German: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.align.vec) | Greek: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.el.align.vec) | English: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.align.vec) | Spanish: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.align.vec) |
| Estonian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.et.align.vec) | Persian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fa.align.vec) | Finnish: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fi.align.vec) | French: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.align.vec) |
| Hebrew: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.he.align.vec) | Hindi: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.hi.align.vec) | Croatian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.hr.align.vec) | Hungarian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.hu.align.vec) |
| Indonesian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.id.align.vec) | Italian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.it.align.vec) | Korean: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ko.align.vec) | Lithuanian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.lt.align.vec) |
| Latvian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.lv.align.vec) | Macedonian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.mk.align.vec) | Malay: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ms.align.vec) | Dutch: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.nl.align.vec) |
| Norwegian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.no.align.vec) | Polish: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.pl.align.vec) | Portuguese: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.pt.align.vec) | Romanian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ro.align.vec) |
| Russian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ru.align.vec) | Slovak: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.sk.align.vec) | Slovenian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.sl.align.vec) | Albanian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.sq.align.vec) |
| Swedish: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.sv.align.vec) | Tamil: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ta.align.vec) | Thai: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.th.align.vec) | Tagalog: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.tl.align.vec) |
| Turkish: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.tr.align.vec) | Ukrainian: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.uk.align.vec) | Vietnamese: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.vi.align.vec) | Chinese: [*text*](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.align.vec) |

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
