#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:00:30:00
#$ -j y
#$ -o outputs/inference/mixtral-8x7b/
#$ -cwd

set -e

# swich virtual env

python ~/Mixtral/tools/inference/inference-mixtral.py \
  --model-path Eric2333/Mixtral-GCP-upload-test2 \
  --tokenizer-path ~/Mixtral/tokenizer_model_directory \
  --prompt "Tokyo is the capital of "
