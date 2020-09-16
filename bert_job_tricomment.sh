#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=rrg-emilios

#SBATCH --job-name=BCS_attention_tricomment_test
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=32G
#SBATCH --output=%j-%x.log
source ~/bertcommentsum/bin/activate
export CLASSPATH=~/projects/rrg-emilios/nachotp/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar 
export CLASSPATH=$CLASSPATH:~/projects/rrg-emilios/nachotp/BertCommentSum/stanford-corenlp-4.0.0-models-spanish.jar

python src/train.py  -mode train -dec_dropout 0.2 -n_comments 3 -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 100 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1  -log_file logs/attn_3comment.log

# python src/train.py  -mode train -dec_dropout 0.2 -n_comments 3 -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 100 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file logs/attn_3comment.log
