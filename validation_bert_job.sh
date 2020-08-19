#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --account=def-emilios
#SBATCH --mail-user=thenachotp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=BertCommentSum_Validation_job
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --output=%j-%x.log
source ~/bertcommentsum/bin/activate
export CLASSPATH=~/projects/def-emilios/nachotp/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar 
export CLASSPATH=$CLASSPATH:~/projects/def-emilios/nachotp/BertCommentSum/stanford-corenlp-4.0.0-models-spanish.jar

python src/train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500  -log_file logs/attention_layer_sqrt.log -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 512 -alpha 0.95 -min_length 10 -result_path logs/macro_demo5hr.log 