#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:00:30:00
#$ -j y
#$ -o outputs/inference/mixtral-8x7b/
#$ -cwd

set -e

# swich virtual env
conda activate mixtralenv
source .env/bin/activate

python tools/inference/inference-mixtral.py \
  --model-path /bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-v0.1 \
  --tokenizer-path /bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-v0.1 \
  --prompt "Tokyo is the capital of "
