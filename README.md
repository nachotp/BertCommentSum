# Bert Abstractive Comment Summarization

**This code is based on PreSumm from the EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) ([repo](https://github.com/nlpyang/PreSumm/))**


**Python version**: This code is in Python 3.6

**Package Requirements**: torch==1.1.0 ransformers tensorboardX multiprocess pyrouge


For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.


Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## BERT Model used
[BETO](https://github.com/dccuchile/beto) from [Spanish Pre-Trained BERT Model and Evaluation Data](https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf) 

### Preprocessing

#### Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


#### Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

#### Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**


### Abstractive Setting

#### BertAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../logs/abs_bert_cnndm
```



