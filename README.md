# Full length Answer Generation from Spoken Questions
# Based on OpenNMT-py: Open-Source Neural Machine Translation

Code base for paper.The dataset is contained in data directory. train.ques, train.ans, train.tgt contains data triplet (question, factoid answer, target full length answer) in each line respectively.

The codebase is built over [OpenNMT](https://github.com/OpenNMT/OpenNMT)

## Requirements
All dependencies can be installed via:

```bash
pip install -r requirements.txt
```
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
onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -ques data/ques-test.txt -ans data/ans-test.txt -output pred.txt -replace_unk -verbose -beam 5
```

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

## Citation
