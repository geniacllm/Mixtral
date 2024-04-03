#!/bin/bash
#SBATCH --job-name=pretrain_job
#SBATCH --output=pretrain_job_output.txt
#SBATCH --error=pretrain_job_error.txt

#$ -l rt_AF=16
#$ -l h_rt=12:0:00:00
#$ -j y
#$ -o outputs/mixtral-7bx8/okazaki-cc/
#$ -cwd

# 説明：シングルノードマルチGPUのCUDA環境下で実行
set -e
echo ""

# Activate the correct conda environment
# source /home/i23_eric/miniconda3/etc/profile.d/conda.sh
# conda init
# conda activate mixtralenv

# Stores the directory paths as variables.
export recipe_dir="/mnt/nfs-mnj-home-43/i23_eric/code-server/userdata/Mixtral"

megatron_deepspeed_dir="${recipe_dir}/Megatron-DeepSpeed"

export PYTHONPATH="${recipe_dir}/src:${PYTHONPATH}"
export PYTHONPATH="${recipe_dir}:${PYTHONPATH}"

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29501

echo "MASTER_ADDR=${MASTER_ADDR}"

# SLURMを使用しない
NUM_GPU_PER_NODE=1
NUM_NODES=2
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# mkdir -p ~/app/hostfile
HOSTFILE_NAME=/tmp/hostfile
# touch $HOSTFILE_NAME
echo "$(hostname) slots=${NUM_GPU_PER_NODE}" >"$HOSTFILE_NAME"

## scontrolを使ってノードリストを取得し、NUM_GPU_PER_NODEで指定されたGPUスロット数とともにファイルに書き出す
# scontrol show hostname $SLURM_JOB_NODELIST | while read -r line; do
#   echo "${line} slots=${NUM_GPU_PER_NODE}"
# done >"$HOSTFILE_NAME"

# training config
# Mixtral-8x7B https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
SEQ_LENGTH=1024
SLIDING_WINDOW_SIZE=1024
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=256
TRAIN_STEPS=25000

# optimizer config
LR=2e-5
MIN_LR=2e-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL="${recipe_dir}/tokenizer_model_directory/tokenizer.model"
CHECKPOINT_DIR=Mixtral_pretrain
CHECKPOINT_SAVE_DIR="${recipe_dir}/scripts/abci/mixtral/v0.1/"
mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH="${megatron_deepspeed_dir}/dataset/arxiv_text_document"
if [ ! -f "${DATA_PATH}.bin" ] || [ ! -f "${DATA_PATH}.idx" ]; then
    echo "Either ${DATA_PATH}.bin or ${DATA_PATH}.idx doesn't exist yet, so download arxiv.jsonl and preprocess the data."
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl \
        --directory-prefix ${megatron_deepspeed_dir}/dataset/
    mv ${megatron_deepspeed_dir}/dataset/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl ${megatron_deepspeed_dir}/dataset/arxiv.jsonl
    python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --input ${megatron_deepspeed_dir}/dataset/arxiv.jsonl \
        --output-prefix ${megatron_deepspeed_dir}/dataset/arxiv \
        --dataset-impl mmap \
        --workers 32 \
        --append-eod
else
    echo "Both ${data_path}.bin and ${data_path}.idx already exist."
fi
echo ""


# job name
JOB_NAME="Mixtral-8x7b-GENIAC-eric-v0.1"

# --bf16 --mixed-precision を削除
# run
mpirun -np $NUM_GPUS \
  --allow-run-as-root \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python ${recipe_dir}/examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SLIDING_WINDOW_SIZE} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --bf16 \
  --param-dtype bf16 \
  --mixed-precision \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-6 \
  --save-interval 250 \
  --eval-interval 100 \
  --eval-iters 10 \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --use-zero \
  --zero-config "${recipe_dir}/scripts/abci/mixtral/mixtral-config.json" \
  --zero-stage 3 \
  --no-meta-device \
  --use-mpi \
  --wandb-project "Mixtral-8x7b" \
  --wandb-name "${JOB_NAME}"
