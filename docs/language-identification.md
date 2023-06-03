---
id: language-identification
title: Language identification
---

### Description

We distribute two models for language identification, which can recognize 176 languages (see the list of ISO codes below). These models were trained on data from [Wikipedia](https://www.wikipedia.org/), [Tatoeba](https://tatoeba.org/eng/) and  [SETimes](http://nlp.ffzg.hr/resources/corpora/setimes/), used under [CC-BY-SA](http://creativecommons.org/licenses/by-sa/3.0/).

We distribute two versions of the models:

* [lid.176.bin](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin), which is faster and slightly more accurate, but has a file size of 126MB ;
* [lid.176.ftz](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz), which is the compressed version of the model, with a file size of 917kB.

These models were trained on UTF-8 data, and therefore expect UTF-8 as input.

#### Updated model (NLLB project)
A newer LID (**L**anguage **ID**entification) model was [released as part of the NLLB project](https://github.com/facebookresearch/fairseq/tree/nllb#lid-model) under [CC-BY-NC 4.0](LICENSE.model.md) license. 

* [lid218e.bin](https://tinyurl.com/nllblid218e) uses different language codes from the original models—the ISO 639-3 code (e.g. "eng", "fra", "rus") plus an additional code describing the script (e.g., "eng_Latn", "ukr_Cyrl")—and has a file size of 1.2GB.

You can read more about the data the model was trained on [here](https://github.com/facebookresearch/fairseq/blob/nllb/README.md#datasets).

#### 🤗 HuggingFace Integration
This model is [available](https://huggingface.co/facebook/fasttext-language-identification) on the Hugging Face Hub. 

```python
>>> import fasttext
>>> from huggingface_hub import hf_hub_download

>>> model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
>>> model = fasttext.load_model(model_path)
>>> model.predict("Hello, world!")

(('__label__eng_Latn',), array([0.81148803]))

>>> model.predict("Hello, world!", k=5)

(('__label__eng_Latn', '__label__vie_Latn', '__label__nld_Latn', '__label__pol_Latn', '__label__deu_Latn'), 
 array([0.61224753, 0.21323682, 0.09696738, 0.01359863, 0.01319415]))
```


### License

The models are distributed under the [*Creative Commons Attribution-Share-Alike License 3.0*](https://creativecommons.org/licenses/by-sa/3.0/).

### List of supported languages
```
af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh
```

### References

If you use these models, please cite the following papers:

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)
```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```
[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [*FastText.zip: Compressing text classification models* ](https://arxiv.org/abs/1612.03651)
```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```
