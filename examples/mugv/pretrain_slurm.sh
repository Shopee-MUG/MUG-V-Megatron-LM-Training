#!/bin/bash -e
#
# SLURM training script
#
# Usage:
#   Single-node (8 GPUs):
#     sbatch --nodes=1 --gpus-per-node=8 examples/mugv/pretrain_slurm.sh
#
#   Multi-node (512 GPUs):
#     sbatch --nodes=64 --gpus-per-node=8 examples/mugv/pretrain_slurm.sh
#
# Or run directly without SLURM (single-node):
#     bash examples/mugv/pretrain_slurm.sh

if [ -n "${SLURM_JOB_ID}" ]; then
    echo "Detected SLURM environment"

    # SLURM provides these automatically
    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
    export MASTER_PORT=${MASTER_PORT:-29500}
    export NNODES=${SLURM_NNODES}
    export NODE_RANK=${SLURM_NODEID}
    export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
    export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

    echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
    echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
    echo "SLURM_NODELIST: ${SLURM_NODELIST}"
else
    echo "Standalone mode (no SLURM detected)"

    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29500}
    export NNODES=${NNODES:-1}
    export NODE_RANK=${NODE_RANK:-0}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
fi

echo "Distributed Configuration:"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "WORLD_SIZE: ${WORLD_SIZE}"

# Export environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# export NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
export TORCH_LOGS="recompiles"
export TRITON_CACHE_DIR="${HOME}/tmp/triton-cache/"
export NCCL_IB_SL=1
# export NCCL_DEBUG=WARN
# export NCCL_P2P_DISABLE=1

export WORKSPACE='/data'
export LOAD_NAME=MUG-V-10B-TP4-legacy
export MODEL_TYPE=${MODEL_TYPE:-"mugdit_10b"}

SOURCE=$(pwd)
AIP_TAG="${SLURM_JOB_ID:-local}-${SLURM_JOB_NAME:-run}"
MODEL_NAME="${AIP_TAG}-mcore-mugdit"

OUTPUT_WORKSPACE="/data"
OUTPUT_BASE="${OUTPUT_WORKSPACE}/mugdit_debug_output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR="${OUTPUT}/checkpoints"
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

# Allow user to override CHECKPOINT_DIR via environment variable
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORKSPACE}/${LOAD_NAME}/checkpoints}"
# PRETRAINED_TASK_ID=993185-mcore-test-tp-n-aip-mcore-mugdit
# CHECKPOINT_DIR="${OUTPUT_BASE}/${PRETRAINED_TASK_ID}/checkpoints"
# LOAD_ITER=2600
# echo $LOAD_ITER > ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt

export DATA_TRAIN="${DATA_TRAIN:-"/data/train.csv"}"

# Map MODEL_TYPE to hidden size / heads / layers
case "${MODEL_TYPE}" in
  mugdit_1b)
    HIDDEN_SIZE=1152
    NUM_HEADS=16
    NUM_LAYERS=${NUM_LAYERS:-56}
    ;;
  mugdit_4b)
    HIDDEN_SIZE=2304
    NUM_HEADS=32
    NUM_LAYERS=${NUM_LAYERS:-56}
    ;;
  mugdit_10b)
    HIDDEN_SIZE=3456
    NUM_HEADS=48
    NUM_LAYERS=${NUM_LAYERS:-56}
    ;;
  mugdit_18b)
    HIDDEN_SIZE=4608
    NUM_HEADS=64
    NUM_LAYERS=${NUM_LAYERS:-56}
    ;;
  mugdit_debug)
    HIDDEN_SIZE=288
    NUM_HEADS=4
    NUM_LAYERS=${NUM_LAYERS:-56}
    ;;
  *)
    echo "[ERROR] Unknown MODEL_TYPE: ${MODEL_TYPE}" >&2
    exit 1
    ;;
esac

KV_CHANNELS=$(( HIDDEN_SIZE / NUM_HEADS ))
NUM_QUERY_GROUPS=${NUM_HEADS}

# Training configurations
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}

BZ=$((WORLD_SIZE / TP_SIZE / PP_SIZE * 1))
SEQ_LEN=580000
TRAIN_ITERS=${TRAIN_ITERS:-100000}
EVAL_INTERVAL=100000
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
NUM_WORKERS=10

EXTRA_ARGS=()

# Options
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    # --num-layers-per-virtual-pipeline-stage ${VP_SIZE}
    # --no-overlap-p2p-communication
    --sequence-parallel
    --context-parallel-size ${CP_SIZE}
)

MODEL_ARGS=(
    --model-type ${MODEL_TYPE}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --kv-channels ${KV_CHANNELS}
    --num-attention-heads ${NUM_HEADS}
    --num-query-groups ${NUM_QUERY_GROUPS}
    --seq-length ${SEQ_LEN}
    --add-qkv-bias
    --normalization RMSNorm
    --qk-layernorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --max-position-embeddings $((4 * SEQ_LEN))
    --rotary-percent 1.0
    --rotary-base 10000
    --rotary-interleaved
    --no-rope-fusion
    --transformer-impl transformer_engine
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --tokenizer-type NullTokenizer
    --vocab-size 0
    --untie-embeddings-and-output-weights
)

TRAINING_ARGS=(
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
    --override-opt_param-scheduler
    --train-iters ${TRAIN_ITERS}
    --micro-batch-size 1
    --global-batch-size ${BZ}
    --lr 1e-5
    --min-lr 1e-5
    --lr-warmup-iters 100
    --lr-decay-iters 200
    --lr-decay-style cosine
    --weight-decay 0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-10
    --clip-grad 1.0
    --bf16
    --init-method-std 0.014
    --seed 6309
)

DATA_ARGS=(
    --dataloader-type external
    --data-path ${DATA_TRAIN}
    --num-workers ${NUM_WORKERS}
    --split 100,0,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-iters 1
    --eval-interval ${EVAL_INTERVAL}
    --save-interval ${SAVE_INTERVAL}
    --save ${FINETUNE_DIR}
    --load ${FINETUNE_DIR}
    --dataloader-save ${FINETUNE_DIR}/dataloader
    --no-load-rng
    --no-load-optim
    # --allow-missing-norm-checkpoint
    # --check-weight-hash-across-dp-replicas-interval 200
    --dist-ckpt-strictness log_all
    # --freeze-context-embedder
    --pretrained-checkpoint ${CHECKPOINT_DIR}
    # --finetune
    --tensorboard-dir ${TENSORBOARD_DIR}
    --log-params-norm
    --log-num-zeros-in-grad
    --log-throughput
    --log-progress
    ${LOGGING_ARGS[@]}
)

PERFORMANCE_ARGS=(
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --recompute-method uniform
    --recompute-granularity full
    --recompute-num-layers 1
    --use-flash-attn
    --manual-gc
    --async-save
    --distributed-timeout-minutes 60
    --exit-duration-in-mins 230000
)

DISTRIBUTED_ARGS=(
    --nproc_per_node ${GPUS_PER_NODE}
    --nnodes ${NNODES}
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

# Launch training with torchrun
torchrun \
    ${DISTRIBUTED_ARGS[@]} \
    examples/mugv/train_mugdit.py \
    ${MODEL_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PERFORMANCE_ARGS[@]} \
    ${EXTRA_ARGS[@]}
