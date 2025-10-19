#!/bin/bash -e
# Pretrain a MUGDiT model

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

SOURCE=`pwd`

AIP_TAG="${AIP_RUN_ID:-0}-${AIP_RUN_NAME:-0}"
MODEL_NAME="${AIP_TAG}-mcore-mugdit"

OUTPUT_WORKSPACE="/data"
OUTPUT_BASE="${OUTPUT_WORKSPACE}/mugdit_debug_output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

# Tag for this run and each rank
DIST_TAG="${WORLD_SIZE}-${RANK}-${CUDA_VISIBLE_DEVICES}-${NODE_IP}-$(hostname)"

FINETUNE_DIR="${OUTPUT}/checkpoints"
LOGS_DIR="${OUTPUT}/logs-${AIP_TAG}"
TENSORBOARD_DIR="${OUTPUT}/tensorboard-${AIP_TAG}"
STDOUT_FILE="${LOGS_DIR}/stdout/${DIST_TAG}.stdout"
STDERR_FILE="${LOGS_DIR}/stderr/${DIST_TAG}.stderr"
ENV_FILE="${LOGS_DIR}/env/${DIST_TAG}.env"
CODE_FILE="${LOGS_DIR}/code/${DIST_TAG}.tar"
mkdir -p "$FINETUNE_DIR" "${LOGS_DIR}/stdout" "${LOGS_DIR}/stderr" "${LOGS_DIR}/env" "${LOGS_DIR}/code"
env > ${ENV_FILE}

if [[ "${RANK:-0}" -ne 0 ]]; then
    echo "Skipping packing source code as RANK=${RANK:-not set}."
else
    tar -cf "$CODE_FILE" -C "$SOURCE" .
fi

CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-${OUTPUT_BASE}/ckpt_save_demo}"
# Allow user to override CHECKPOINT_DIR via environment variable
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORKSPACE}/${LOAD_NAME}/checkpoints}"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "CKPT_SAVE_DIR: $CKPT_SAVE_DIR"

DATA_TRAIN="${DATA_TRAIN:-"/data/train.csv"}"

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
TP_SIZE=4
PP_SIZE=1
VP_SIZE=1
CP_SIZE=1
BZ=$((WORLD_SIZE / TP_SIZE / PP_SIZE * 1))
SEQ_LEN=580000
TRAIN_ITERS=100000
EVAL_INTERVAL=100000
SAVE_INTERVAL=100

NUM_WORKERS=10

EXTRA_ARGS=()

# Options
CONVERT_ARGS=(
    --ckpt-convert-format torch_dist
    --ckpt-convert-save ${CKPT_SAVE_DIR}
)

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
    --distributed-timeout-minutes 60
    --exit-duration-in-mins 230000
)

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes ${WORLD_SIZE}
    --node_rank ${RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

# Launch training with torchrun
torchrun \
    ${DISTRIBUTED_ARGS[@]} \
    examples/mugv/train_mugdit.py \
    ${CONVERT_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PERFORMANCE_ARGS[@]} \
    ${EXTRA_ARGS[@]} \
    > >(tee -a "${STDOUT_FILE}") 2> >(tee -a "${STDERR_FILE}" >&2)

# Add exit status to make aip retry happy
EXIT_CODE=$?
echo "Exiting with status $EXIT_CODE" >> "${STDOUT_FILE}.finished"
exit $EXIT_CODE
