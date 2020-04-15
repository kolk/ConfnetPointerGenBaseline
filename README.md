# Full length Answer Generation from Spoken Questions
# Based on OpenNMT-py: Open-Source Neural Machine Translation


This is a [PyTorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains. Some companies have proven the code to be production ready.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

## Requirements


### Step 1a: Add padding:
Pad the answer text file.
```
python add_padding.py --input <answer-filename> --output <padded-answer-filename> --padding <num of pad>
```

### Step 1: Preprocess the data

```bash
onmt_preprocess -train_ques data/ques-train.txt -train_ans data/ans-train.txt -train_tgt data/tgt-train.txt -valid_ques data/ques-val.txt -valid_ans data/ans-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo --dynamic_dict --share_vocab
```

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `ques-train.txt` : text file containing path to a question confusion network in each line
* `ans-train.txt` : text file containing the padded factoid answer in each line
* `tgt-train.txt` : text file containing the target full length answer
* `ques-val.txt` : text file containing the path to a validation question confusion network in each line 
* `ans-val.txt` : validation text file containing the path to the padded factoid answer in each line
* `tgt-val.txt` : validation text file containing the target full length answer

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
onmt_train -data data/demo -save_model demo-model -word_vec_size 300 -model_type lattice -encoder_type brnn -layers 2 -rnn_size 512 \
-data data/demo -batch_size 32 -valid_batch_size 32 -valid_steps 2500 -dropout 0.5 -start_decay_steps 10000 -coverage_attn -copy_attn \
--share_embeddings
```

### Step 3: Translate

```bash
onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -ques data/ques-test.txt -ans data/ans-test.txt -output pred.txt -replace_unk -verbose
```

## Pretrained embeddings (e.g. GloVe)

Please see the FAQ: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC) to reproduce OpenNMT-Lua using Pytorch.

Major contributors are:
[Sasha Rush](https://github.com/srush) (Cambridge, MA)
[Vincent Nguyen](https://github.com/vince62s) (Ubiqus)
[Ben Peters](http://github.com/bpopeters) (Lisbon)
[Sebastian Gehrmann](https://github.com/sebastianGehrmann) (Harvard NLP)
[Yuntian Deng](https://github.com/da03) (Harvard NLP)
[Guillaume Klein](https://github.com/guillaumekln) (Systran)
[Paul Tardy](https://github.com/pltrdy) (Ubiqus / Lium)
[François Hernandez](https://github.com/francoishernandez) (Ubiqus)
[Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai)
[Dylan Flaute](http://github.com/flauted (University of Dayton)
and more !

OpenNMT-py belongs to the OpenNMT project along with OpenNMT-Lua and OpenNMT-tf.

## Citation

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
