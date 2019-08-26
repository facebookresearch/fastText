---
id: cheatsheet
title: Cheatsheet
---

## Word representation learning

In order to learn word vectors do:

```bash
$ ./fasttext skipgram -input data.txt -output model
```

## Obtaining word vectors

Print word vectors for a text file `queries.txt` containing words.

```bash
$ ./fasttext print-word-vectors model.bin < queries.txt
```

## Text classification

In order to train a text classifier do:

```bash
$ ./fasttext supervised -input train.txt -output model
```

Once the model was trained, you can evaluate it by computing the precision and recall at k (P@k and R@k) on a test set using:

```bash
$ ./fasttext test model.bin test.txt 1
```

In order to obtain the k most likely labels for a piece of text, use:

```bash
$ ./fasttext predict model.bin test.txt k
```

In order to obtain the k most likely labels and their associated probabilities for a piece of text, use:

```bash
$ ./fasttext predict-prob model.bin test.txt k
```

If you want to compute vector representations of sentences or paragraphs, please use:

```bash
$ ./fasttext print-sentence-vectors model.bin < text.txt
```

## Quantization

In order to create a `.ftz` file with a smaller memory footprint do:

```bash
$ ./fasttext quantize -output model
```

All other commands such as test also work with this model

```bash
$ ./fasttext test model.ftz test.txt
```

## Autotune

Activate hyperparameter optimization with `-autotune-validation` argument:

```bash
$ ./fasttext supervised -input train.txt -output model -autotune-validation valid.txt
```

Set timeout (in seconds):
```bash
$ ./fasttext supervised -input train.txt -output model -autotune-validation valid.txt -autotune-duration 600
```

Constrain the final model size:
```bash
$ ./fasttext supervised -input train.txt -output model -autotune-validation valid.txt -autotune-modelsize 2M
```





