#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=12:0:00:00
#$ -j y
#$ -o outputs/mixtral-7bx8/okazaki-cc/
#$ -cwd

set -e
echo ""


# Stores the directory paths as variables.
ucllm_nedo_dev="${HOME}/moe-recipes"
megatron_deepspeed_dir="${ucllm_nedo_dev}/Megatron-DeepSpeed"

export PYTHONPATH="${ucllm_nedo_dev}/src:${PYTHONPATH}"
export PYTHONPATH="${ucllm_nedo_dev}:${PYTHONPATH}"

# distributed settings
#export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
#export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

# Google Colab用
export MASTER_ADDR="localhost"
export MASTER_PORT="12345" # 任意の未使用ポートを指定

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS

#Google Colab用
NUM_NODES=1
NUM_GPU_PER_NODE=1
JOB_ID=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}

# 仮のホストリストを作成するためのサンプルテキスト Google Colab用
echo "localhost" > ./sample_hostlist.txt
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"./sample_hostlist.txt" >"$HOSTFILE_NAME"

# training config
# Mixtral-8x7B https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
SEQ_LENGTH=1024
SLIDING_WINDOW_SIZE=1024
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=2
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
CHECKPOINT_SAVE_DIR="/bb/llm/gaf51275/llama/checkpoints/Mixtral-8x7b/okazaki-cc-lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_sliding_window_${SLIDING_WINDOW_SIZE}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH="${megatron_deepspeed_dir}/dataset/arxiv_text_document"

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
  --tokenizer-type Llama2Tokenizer \
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
  --zero-config "scripts/abci/mixtral/mixtral-config.json" \
  --zero-stage 3 \
  --no-meta-device \
  --use-mpi \
  --wandb-entity "prj-jalm" \
  --wandb-project "Mixtral-8x7b" \
  --wandb-name "${JOB_NAME}"
