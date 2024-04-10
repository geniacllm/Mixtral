#!/bin/bash
#SBATCH --job-name=deepspeed_job
#SBATCH --output=deepspeed_job_output.txt
#SBATCH --error=deepspeed_job_error.txt

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

ucllm_nedo_dev="${HOME}/Mixtral"
saved_model_directory="Mixtral-8x7b-GENIAC-eric-gcp-single-node-v0.2"
megatron_deepspeed_dir="${ucllm_nedo_dev}/Megatron-DeepSpeed"
export PYTHONPATH="${ucllm_nedo_dev}/src:${PYTHONPATH}"
export PYTHONPATH="${ucllm_nedo_dev}:${PYTHONPATH}"

ITERATION=50
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_DIR=${ucllm_nedo_dev}/${saved_model_directory}/${FORMATTED_ITERATION}

python ${ucllm_nedo_dev}/tools/checkpoint-convert/zero_to_fp32.py \
  --checkpoint-dir $CHECK_POINT_DIR \
  --output-file $CHECK_POINT_DIR/fp32_model.bin \
  --debug
