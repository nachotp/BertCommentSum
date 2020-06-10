#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --account=def-emilios
#SBATCH --job-name=BertCommentSum 
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12G
#SBATCH --output=%j-%x.out
source ~/bertcommentsum/bin/activate
lemiknow telegram --token 1081026745:AAG11u-kKTPlEqnmYvmdXsnMLVDlah-8kTI --chat-id 178174272 --include_details True \
    src/train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file ../logs/abs_bert_cnndm