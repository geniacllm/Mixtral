#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd
# module load

set -e

# Activate the correct conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mixtralenv

ucllm_nedo_dev="${HOME}/moe-recipes"
megatron_deepspeed_dir="${ucllm_nedo_dev}/Megatron-DeepSpeed"
export PYTHONPATH="${ucllm_nedo_dev}/src:${PYTHONPATH}"
export PYTHONPATH="${ucllm_nedo_dev}:${PYTHONPATH}"

ITERATION=1000
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_DIR=${ucllm_nedo_dev}/okazaki-cc-lr_2e-5-minlr_2e-6_warmup_1000_sliding_window_1024/${FORMATTED_ITERATION}

python ${ucllm_nedo_dev}/tools/checkpoint-convert/zero_to_fp32.py \
  --checkpoint-dir $CHECK_POINT_DIR \
  --output-file $CHECK_POINT_DIR/fp32_model.bin \
  --debug
