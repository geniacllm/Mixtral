#!/bin/bash
#SBATCH --job-name=pretrain_job
#SBATCH --output=pretrain_job_output.txt
#SBATCH --error=pretrain_job_error.txt

#$ -l rt_AF=16
#$ -l h_rt=12:0:00:00
#$ -j y
#$ -o outputs/mixtral-7bx8/okazaki-cc/
#$ -cwd

set -e
echo ""

# Activate the correct conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mixtralenv

# Stores the directory paths as variables.
ucllm_nedo_dev="${HOME}/moe-recipes"
megatron_deepspeed_dir="${ucllm_nedo_dev}/Megatron-DeepSpeed"

export PYTHONPATH="${ucllm_nedo_dev}/src:${PYTHONPATH}"
export PYTHONPATH="${ucllm_nedo_dev}:${PYTHONPATH}"

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$((10000 + ($SLURM_JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$GPU_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$GPU_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized GPU_TYPE: $GPU_TYPE"
fi

#GCP用1node1GPU
NUM_NODES=1
NUM_GPU_PER_NODE=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

# SlurmのジョブIDを使用してファイル名を設定
HOSTFILE_NAME=./hostfile/hostfile_${SLURM_JOB_ID}

# scontrolを使ってノードリストを取得し、NUM_GPU_PER_NODEで指定されたGPUスロット数とともにファイルに書き出す
scontrol show hostname $SLURM_JOB_NODELIST | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

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
TOKENIZER_MODEL="${ucllm_nedo_dev}/tokenizer_model_directory/tokenizer.model"
CHECKPOINT_DIR=Mixtral_pretrain
CHECKPOINT_SAVE_DIR="${ucllm_nedo_dev}/okazaki-cc-lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_sliding_window_${SLIDING_WINDOW_SIZE}"

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
JOB_NAME="Mixtral-8x7b-NVE-okazaki-lab-cc-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --allow-run-as-root \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python ${ucllm_nedo_dev}/examples/finetuning.py \
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
  --bf16 \
  --mixed-precision \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --use-zero \
  --zero-config "${HOME}/moe-recipes/scripts/abci/mixtral/mixtral-config.json" \
  --zero-stage 3 \
  --no-meta-device \
  --use-mpi \
  --wandb-project "Mixtral-8x7b" \
  --wandb-name "${JOB_NAME}"
