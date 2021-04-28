## Preprocessing Common Crawl

This code downloads, preprocesses and splits per language the data from [Common Crawl](http://commoncrawl.org/).

This script uses the scripts and language identifier of [1].

This code inherits its requirements form [fastText](https://github.com/facebookresearch/fastText).

Set the variable WET_PATHS_URL to the crawl you want to process.
Please also set the variables NUM_LANGID and NUM_DEDUP in `download_crawl.sh` according to the capacity of your machine.
Langid processes are mostly limited by CPU usage, while dedup processes are likely to be limited by RAM usage (each use 2GB of RAM).

### Reference

If you use this code, please cite:

[1] E. Grave*, P. Bojanowski*, P. Gupta, A. Joulin, T. Mikolov, [*Learning Word Vectors for 157 Languages*](https://arxiv.org/abs/1802.06893)

```
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```
