#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd
# module load
set -e

ucllm_nedo_dev="${HOME}/moe-recipes"
megatron_deepspeed_dir="${ucllm_nedo_dev}/Megatron-DeepSpeed"
export PYTHONPATH="${ucllm_nedo_dev}/src:${PYTHONPATH}"
export PYTHONPATH="${ucllm_nedo_dev}:${PYTHONPATH}"

# Google Colab用
export MASTER_ADDR="localhost"
export MASTER_PORT="12345" # 任意の未使用ポートを指定

echo "MASTER_ADDR=${MASTER_ADDR}"

start=250
end=1000
increment=250

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)
  CHECK_POINT_PATH=${ucllm_nedo_dev}/okazaki-cc-lr_2e-5-minlr_2e-6_warmup_1000_sliding_window_1024/${FORMATTED_ITERATION}/fp32_model.bin
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
