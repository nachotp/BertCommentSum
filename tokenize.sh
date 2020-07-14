#!/bin/bash
#SBATCH --time=05:30:00
#SBATCH --account=def-emilios
#SBATCH --mail-user=thenachotp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Bert_tokenize_job
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=%j-%x.out
module load java
source ~/bertcommentsum/bin/activate
export CLASSPATH=~/projects/def-emilios/nachotp/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar 
export CLASSPATH=$CLASSPATH:~/projects/def-emilios/nachotp/BertCommentSum/stanford-corenlp-4.0.0-models-spanish.jar

python src/preprocess.py -mode tokenize