The confusion network can directories can be found at https://drive.google.com/drive/folders/1nFtsOrSdE5v6Bjsw-B90MBYWvVtFBsYr?usp=sharing



> python preprocess.py -train_confnet data/ques-train.txt -train_src data/ans-train.txt -train_tgt data/tgt-train.txt -valid_confnet data/ques-val.txt -valid_src data/ans-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo --dynamic_dict --share_vocab


> python train.py -data data/demo -save_model demo-model -word_vec_size 300 -model_type lattice -encoder_type brnn -layers 2 -rnn_size 512 \
-data data/demo -batch_size 32 -valid_batch_size 32 -valid_steps 2500 -dropout 0.5 -start_decay_steps 10000 -coverage_attn -copy_attn \
--share_embeddings
