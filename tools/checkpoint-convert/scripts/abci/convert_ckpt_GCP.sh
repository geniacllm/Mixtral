#!/bin/bash

#SBATCH --job-name=convert_job
#SBATCH --output=convert_job_output.txt
#SBATCH --error=convert_job_error.txt


#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd
# module load
set -e

ucllm_nedo_dev="${HOME}/moe-recipes"
megatron_deepspeed_dir="${ucllm_nedo_dev}/Megatron-DeepSpeed"
saved_model_directory="Mixtral-8x7b-GENIAC-eric-gcp-single-node-v0.2"

# export PYTHONPATH="${ucllm_nedo_dev}/src:${PYTHONPATH}"
# export PYTHONPATH="${ucllm_nedo_dev}:${PYTHONPATH}"

# export MASTER_ADDR=$(hostname -i)
# export MASTER_PORT=$((10000 + ($SLURM_JOB_ID % 50000)))
# echo "MASTER_ADDR=${MASTER_ADDR}"

start=50
end=50
increment=50

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)
  CHECK_POINT_PATH=${ucllm_nedo_dev}/${saved_model_directory}/${FORMATTED_ITERATION}/fp32_model.bin
  OUTPUT_PATH=${ucllm_nedo_dev}/huggingface/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=Mixtral_pretrain

  python ${ucllm_nedo_dev}/tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 4096
done
