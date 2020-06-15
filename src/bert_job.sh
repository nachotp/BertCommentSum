#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --account=def-emilios
#SBATCH --job-name=BertCommentSum 
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12G
#SBATCH --output=%j-%x.out
source ~/bertcommentsum/bin/activate
export CLASSPATH=~/projects/def-emilios/nachotp/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar 
export CLASSPATH=$CLASSPATH:~/projects/def-emilios/nachotp/BertCommentSum/stanford-corenlp-4.0.0-models-spanish.jar

lemiknow telegram --token 1081026745:AAG11u-kKTPlEqnmYvmdXsnMLVDlah-8kTI --chat-id 178174272 --include_details True \
    python train.py  -mode train -dec_dropout 0.2  -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0  -log_file logs/demo1