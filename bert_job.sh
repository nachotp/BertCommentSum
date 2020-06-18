#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-emilios
#SBATCH --mail-user=thenachotp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=BertCommentSum_Test_job
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --output=%j-%x.out
source ~/bertcommentsum/bin/activate
export CLASSPATH=~/projects/def-emilios/nachotp/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar 
export CLASSPATH=$CLASSPATH:~/projects/def-emilios/nachotp/BertCommentSum/stanford-corenlp-4.0.0-models-spanish.jar

python src/train.py  -mode train -dec_dropout 0.2 -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0  -log_file logs/micro_demo1.log