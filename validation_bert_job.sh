#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --account=rrg-emilios
#SBATCH --mail-user=thenachotp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=BertCommentSum_Validation_job
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --output=%j-%x.log
source ~/bertcommentsum/bin/activate
export CLASSPATH=~/projects/rrg-emilios/nachotp/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar 
export CLASSPATH=$CLASSPATH:~/projects/rrg-emilios/nachotp/BertCommentSum/stanford-corenlp-4.0.0-models-spanish.jar

#python src/train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500  -log_file logs/sqrtattn_notitles_val.log -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 512 -alpha 0.95 -min_length 10 -result_path logs/sqrtatt_notitle.log 

python src/train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -log_file logs/nofinetune_notitle.log -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 512 -alpha 0.95 -min_length 10 -result_path results/nofinetune_notitle.log 