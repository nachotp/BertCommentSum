#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=def-emilios
#SBATCH --mail-user=thenachotp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Bert_To_lines_job
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --output=%j-%x.out
module load java
source ~/bertcommentsum/bin/activate
export CLASSPATH=~/projects/def-emilios/nachotp/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar 
export CLASSPATH=$CLASSPATH:~/projects/def-emilios/nachotp/BertCommentSum/stanford-corenlp-4.0.0-models-spanish.jar

python src/preprocess.py -mode format_to_lines