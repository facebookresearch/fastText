---
id: crawl-vectors
title: Word vectors for 157 languages
---

We distribute pre-trained word vectors for 157 languages, trained on [*Common Crawl*](http://commoncrawl.org/) and [*Wikipedia*](https://www.wikipedia.org) using fastText.
These models were trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.
We also distribute three new word analogy datasets, for French, Hindi and Polish.

### Download directly with command line or from python

In order to download with command line or from python code, you must have installed the python package as [described here](/docs/en/support.html#building-fasttext-python-module).

<!--DOCUSAURUS_CODE_TABS-->
<!--Command line-->
```bash
$ ./download_model.py en     # English
Downloading https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
 (19.78%) [=========>                                         ]
```
Once the download is finished, use the model as usual:
```bash
$ ./fasttext nn cc.en.300.bin 10
Query word?
```
<!--Python-->
```py
>>> import fasttext.util
>>> fasttext.util.download_model('en', if_exists='ignore')  # English
>>> ft = fasttext.load_model('cc.en.300.bin')
```
<!--END_DOCUSAURUS_CODE_TABS-->

### Adapt the dimension

The pre-trained word vectors we distribute have dimension 300. If you need a smaller size, you can use our dimension reducer.
In order to use that feature, you must have installed the python package as [described here](/docs/en/support.html#building-fasttext-python-module).

For example, in order to get vectors of dimension 100:
<!--DOCUSAURUS_CODE_TABS-->

<!--Command line-->
```bash
$ ./reduce_model.py cc.en.300.bin 100
Loading model
Reducing matrix dimensions
Saving model
cc.en.100.bin saved
```
Then you can use the `cc.en.100.bin` model file as usual.

<!--Python-->
```py
>>> import fasttext
>>> import fasttext.util
>>> ft = fasttext.load_model('cc.en.300.bin')
>>> ft.get_dimension()
300
>>> fasttext.util.reduce_model(ft, 100)
>>> ft.get_dimension()
100
```
Then you can use `ft` model object as usual:
```py
>>> ft.get_word_vector('hello').shape
(100,)
>>> ft.get_nearest_neighbors('hello')
[(0.775576114654541, u'heyyyy'), (0.7686290144920349, u'hellow'), (0.7663413286209106, u'hello-'), (0.7579624056816101, u'heyyyyy'), (0.7495524287223816, u'hullo'), (0.7473770380020142, u'.hello'), (0.7407292127609253, u'Hiiiii'), (0.7402616739273071, u'hellooo'), (0.7399682402610779, u'hello.'), (0.7396857738494873, u'Heyyyyy')]
```
or save it for later use:
```py
>>> ft.save_model('cc.en.100.bin')
```
<!--END_DOCUSAURUS_CODE_TABS-->


### Format

The word vectors are available in both binary and text formats.

Using the binary models, vectors for out-of-vocabulary words can be obtained with
```
$ ./fasttext print-word-vectors wiki.it.300.bin < oov_words.txt
```
where the file oov_words.txt contains out-of-vocabulary words.

In the text format, each line contain a word followed by its vector.
Each value is space separated, and words are sorted by frequency in descending order.
These text models can easily be loaded in Python using the following code:
```python
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
```

### Tokenization

We used the [*Stanford word segmenter*](https://nlp.stanford.edu/software/segmenter.html) for Chinese, [*Mecab*](http://taku910.github.io/mecab/) for Japanese and [*UETsegmenter*](https://github.com/phongnt570/UETsegmenter) for Vietnamese.
For languages using the Latin, Cyrillic, Hebrew or Greek scripts, we used the tokenizer from the [*Europarl*](http://www.statmt.org/europarl/) preprocessing tools.
For the remaining languages, we used the ICU tokenizer.

More information about the training of these models can be found in the article [*Learning Word Vectors for 157 Languages*](https://arxiv.org/abs/1802.06893).

### License

The word vectors are distributed under the [*Creative Commons Attribution-Share-Alike License 3.0*](https://creativecommons.org/licenses/by-sa/3.0/).

### References

If you use these word vectors, please cite the following paper:

E. Grave\*, P. Bojanowski\*, P. Gupta, A. Joulin, T. Mikolov, [*Learning Word Vectors for 157 Languages*](https://arxiv.org/abs/1802.06893)

```markup
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```

### Evaluation datasets

The analogy evaluation datasets described in the paper are available here: [French](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-fr.txt), [Hindi](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-hi.txt), [Polish](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-pl.txt).

### Models

The models can be downloaded from:

||||
|-|-|-|
|  Afrikaans:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.af.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.af.300.vec.gz) |  Albanian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sq.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sq.300.vec.gz) |  Alemannic:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.als.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.als.300.vec.gz) |
|  Amharic:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.am.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.am.300.vec.gz) |  Arabic:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.vec.gz) |  Aragonese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.an.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.an.300.vec.gz) |
|  Armenian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hy.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hy.300.vec.gz) |  Assamese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.as.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.as.300.vec.gz) |  Asturian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ast.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ast.300.vec.gz) |
|  Azerbaijani:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.az.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.az.300.vec.gz) |  Bashkir:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ba.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ba.300.vec.gz) |  Basque:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.vec.gz) |
|  Bavarian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bar.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bar.300.vec.gz) |  Belarusian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.be.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.be.300.vec.gz) |  Bengali:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz) |
|  Bihari:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bh.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bh.300.vec.gz) |  Bishnupriya Manipuri:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bpy.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bpy.300.vec.gz) |  Bosnian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bs.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bs.300.vec.gz) |
|  Breton:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.br.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.br.300.vec.gz) |  Bulgarian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.vec.gz) |  Burmese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.my.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.my.300.vec.gz) |
|  Catalan:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ca.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ca.300.vec.gz) |  Cebuano:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ceb.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ceb.300.vec.gz) |  Central Bicolano:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bcl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bcl.300.vec.gz) |
|  Chechen:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ce.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ce.300.vec.gz) |  Chinese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz) |  Chuvash:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cv.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cv.300.vec.gz) |
|  Corsican:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.co.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.co.300.vec.gz) |  Croatian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hr.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hr.300.vec.gz) |  Czech:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz) |
|  Danish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.da.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.da.300.vec.gz) |  Divehi:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.dv.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.dv.300.vec.gz) |  Dutch:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz) |
|  Eastern Punjabi:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pa.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pa.300.vec.gz) |  Egyptian Arabic:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.arz.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.arz.300.vec.gz) |  Emilian-Romagnol:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eml.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eml.300.vec.gz) |
|  English: [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz) |  Erzya:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.myv.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.myv.300.vec.gz) |  Esperanto:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eo.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eo.300.vec.gz) |
|  Estonian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.et.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.et.300.vec.gz) |  Fiji Hindi:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hif.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hif.300.vec.gz) |  Finnish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.vec.gz) |
|  French:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz) |  Galician:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gl.300.vec.gz) |  Georgian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ka.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ka.300.vec.gz) |
|  German:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz) |  Goan Konkani:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gom.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gom.300.vec.gz) |  Greek:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.vec.gz) |
|  Gujarati:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gu.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gu.300.vec.gz) |  Haitian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ht.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ht.300.vec.gz) |  Hebrew:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.vec.gz) |
|  Hill Mari:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mrj.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mrj.300.vec.gz) |  Hindi:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz) |  Hungarian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hu.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hu.300.vec.gz) |
|  Icelandic:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.is.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.is.300.vec.gz) |  Ido:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.io.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.io.300.vec.gz) |  Ilokano:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ilo.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ilo.300.vec.gz) |
|  Indonesian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz) |  Interlingua:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ia.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ia.300.vec.gz) |  Irish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ga.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ga.300.vec.gz) |
|  Italian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz) |  Japanese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz) |  Javanese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.jv.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.jv.300.vec.gz) |
|  Kannada:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.kn.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.kn.300.vec.gz) |  Kapampangan:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pam.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pam.300.vec.gz) |  Kazakh:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.kk.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.kk.300.vec.gz) |
|  Khmer:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.km.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.km.300.vec.gz) |  Kirghiz:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ky.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ky.300.vec.gz) |  Korean:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz) |
|  Kurdish (Kurmanji):  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ku.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ku.300.vec.gz) |  Kurdish (Sorani):  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ckb.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ckb.300.vec.gz) |  Latin:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.la.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.la.300.vec.gz) |
|  Latvian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lv.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lv.300.vec.gz) |  Limburgish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.li.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.li.300.vec.gz) |  Lithuanian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lt.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lt.300.vec.gz) |
|  Lombard:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lmo.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lmo.300.vec.gz) |  Low Saxon:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nds.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nds.300.vec.gz) |  Luxembourgish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lb.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lb.300.vec.gz) |
|  Macedonian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mk.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mk.300.vec.gz) |  Maithili:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mai.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mai.300.vec.gz) |  Malagasy:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mg.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mg.300.vec.gz) |
|  Malay:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ms.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ms.300.vec.gz) |  Malayalam:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ml.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ml.300.vec.gz) |  Maltese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mt.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mt.300.vec.gz) |
|  Manx:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gv.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gv.300.vec.gz) |  Marathi:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mr.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mr.300.vec.gz) |  Mazandarani:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mzn.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mzn.300.vec.gz) |
|  Meadow Mari:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mhr.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mhr.300.vec.gz) |  Minangkabau:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.min.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.min.300.vec.gz) |  Mingrelian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.xmf.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.xmf.300.vec.gz) |
|  Mirandese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mwl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mwl.300.vec.gz) |  Mongolian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mn.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mn.300.vec.gz) |  Nahuatl:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nah.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nah.300.vec.gz) |
|  Neapolitan:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nap.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nap.300.vec.gz) |  Nepali:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ne.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ne.300.vec.gz) |  Newar:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.new.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.new.300.vec.gz) |
|  North Frisian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.frr.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.frr.300.vec.gz) |  Northern Sotho:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nso.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nso.300.vec.gz) |  Norwegian (Bokmål):  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.no.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.no.300.vec.gz) |
|  Norwegian (Nynorsk):  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nn.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nn.300.vec.gz) |  Occitan:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.oc.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.oc.300.vec.gz) |  Oriya:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.or.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.or.300.vec.gz) |
|  Ossetian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.os.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.os.300.vec.gz) |  Palatinate German:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pfl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pfl.300.vec.gz) |  Pashto:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ps.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ps.300.vec.gz) |
|  Persian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz) |  Piedmontese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pms.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pms.300.vec.gz) |  Polish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz) |
|  Portuguese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz) |  Quechua:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.qu.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.qu.300.vec.gz) |  Romanian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.vec.gz) |
|  Romansh:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.rm.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.rm.300.vec.gz) |  Russian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz) |  Sakha:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sah.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sah.300.vec.gz) |
|  Sanskrit:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sa.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sa.300.vec.gz) |  Sardinian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sc.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sc.300.vec.gz) |  Scots:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sco.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sco.300.vec.gz) |
|  Scottish Gaelic:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gd.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gd.300.vec.gz) |  Serbian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sr.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sr.300.vec.gz) |  Serbo-Croatian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sh.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sh.300.vec.gz) |
|  Sicilian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.scn.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.scn.300.vec.gz) |  Sindhi:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sd.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sd.300.vec.gz) |  Sinhalese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.vec.gz) |
|  Slovak:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sk.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sk.300.vec.gz) |  Slovenian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.vec.gz) |  Somali:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.so.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.so.300.vec.gz) |
|  Southern Azerbaijani:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.azb.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.azb.300.vec.gz) |  Spanish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz) |  Sundanese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.su.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.su.300.vec.gz) |
|  Swahili:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sw.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sw.300.vec.gz) |  Swedish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sv.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sv.300.vec.gz) |  Tagalog:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tl.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tl.300.vec.gz) |
|  Tajik:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tg.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tg.300.vec.gz) |  Tamil:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ta.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ta.300.vec.gz) |  Tatar:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tt.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tt.300.vec.gz) |
|  Telugu:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.te.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.te.300.vec.gz) |  Thai:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.th.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.th.300.vec.gz) |  Tibetan:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bo.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bo.300.vec.gz) |
|  Turkish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz) |  Turkmen:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tk.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tk.300.vec.gz) |  Ukrainian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.vec.gz) |
|  Upper Sorbian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hsb.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hsb.300.vec.gz) |  Urdu:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ur.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ur.300.vec.gz) |  Uyghur:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ug.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ug.300.vec.gz) |
|  Uzbek:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uz.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uz.300.vec.gz) |  Venetian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vec.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vec.300.vec.gz) |  Vietnamese:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz) |
|  Volapük:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vo.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vo.300.vec.gz) |  Walloon:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.wa.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.wa.300.vec.gz) |  Waray:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.war.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.war.300.vec.gz) |
|  Welsh:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cy.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cy.300.vec.gz) |  West Flemish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vls.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vls.300.vec.gz) |  West Frisian:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fy.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fy.300.vec.gz) |
|  Western Punjabi:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pnb.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pnb.300.vec.gz) |  Yiddish:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yi.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yi.300.vec.gz) |  Yoruba:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yo.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yo.300.vec.gz) |
|  Zazaki:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.diq.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.diq.300.vec.gz) |  Zeelandic:  [bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zea.300.bin.gz), [text](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zea.300.vec.gz) |
