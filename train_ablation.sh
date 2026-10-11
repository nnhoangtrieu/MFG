#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo "ERROR: Command failed at line $LINENO"; exit 1' ERR

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Server arguments
PORT="${PORT:-9001}"
GROUP_PORT="${GROUP_PORT:-51211}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
SEED="${SEED:-910}"
NAME="${NAME:-run$(date +%m%d_%H%M%S)}"

# Training arguments
BASE_MODEL="${BASE_MODEL:-/GeLLMO-P6-Llama}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-24}"
LR="${LR:-1e-5}"
LR_MIN="${LR_MIN:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
BETA="${BETA:-0.1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-96}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}"
MULTI="${MULTI:-5}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-100}"
EXISTING_DATA="${EXISTING_DATA:-}"
MAX_STEPS=$((GENERATION_BATCH_SIZE * MULTI / (GRAD_ACCUM * PER_DEVICE_BATCH)))
SHARD_SIZE=$((((GENERATION_BATCH_SIZE / NUM_GENERATIONS)) * MULTI))
ACCEPTABLE_SIMILARITY_THRESHOLD="${ACCEPTABLE_SIMILARITY_THRESHOLD:-0.4}"
GOOD_SIMILARITY_THRESHOLD="${GOOD_SIMILARITY_THRESHOLD:-0.6}"

echo "$((SHARD_SIZE)) samples will be used for each shard"
echo "$((MAX_STEPS)) training steps will be performed for each shard"

mkdir -p "checkpoint/${NAME}"
cat > "checkpoint/${NAME}/train_args.json" <<EOF
{
    "TRAIN_STEPS": "${TRAIN_STEPS}",
    "NUM_GENERATIONS": "${NUM_GENERATIONS}",
    "GENERATION_BATCH_SIZE": "${GENERATION_BATCH_SIZE}",
    "GRAD_ACCUM": "${GRAD_ACCUM}",
    "PER_DEVICE_BATCH": "${PER_DEVICE_BATCH}",
    "LR": "${LR}",
    "LR_SCHEDULE": "${LR_SCHEDULE}",
    "BETA": "${BETA}",
    "TEMPERATURE": "${TEMPERATURE}",
    "TOP_P": "${TOP_P}",
    "MAX_COMPLETION_LENGTH": "${MAX_COMPLETION_LENGTH}",
    "MAX_RESAMPLE_TIMES": "${MAX_RESAMPLE_TIMES}",
    "MULTI": "${MULTI}",
    "SAMPLES_PER_TASK": "${SAMPLES_PER_TASK}",
    "EXISTING_DATA": "${EXISTING_DATA}",
    "ACCEPTABLE_SIMILARITY_THRESHOLD": "${ACCEPTABLE_SIMILARITY_THRESHOLD}",
    "GOOD_SIMILARITY_THRESHOLD": "${GOOD_SIMILARITY_THRESHOLD}"
}
EOF


DATA_ARGS=(
    --output "data/${NAME}"
    --seed "${SEED}"
)
if [[ -n "${EXISTING_DATA}" ]]; then
    DATA_ARGS+=(--existing_data "${EXISTING_DATA}")
else
    DATA_ARGS+=(
        --exclude MPQ BDMQ BHMQ BMPQ HMPQ
        --samples_per_task "${SAMPLES_PER_TASK}"
    )
fi

python core/ablation/ablation_data.py "${DATA_ARGS[@]}"


CURRENT_ADAPTER="${CURRENT_ADAPTER:-}"


for ((SHARD_IDX=0; SHARD_IDX<TRAIN_STEPS; SHARD_IDX++)); do 

    echo "Sampling ablation shard ${SHARD_IDX}..."
    python core/ablation/ablation_sample.py \
        --train_data "data/${NAME}/train.jsonl" \
        --shard_size "${SHARD_SIZE}" \
        --shard_idx "${SHARD_IDX}" \
        --seed "${SEED}" \
        --output_dir "checkpoint/${NAME}/data"

    SHARD_LR="$(python utils/get_shard_lr.py --base_lr "${LR}" --min_lr "${LR_MIN}" --shard_index "${SHARD_IDX}" --num_shards "${TRAIN_STEPS}" --schedule "${LR_SCHEDULE}")"
    
    export SHARD_IDX
    export ACCEPTABLE_SIMILARITY_THRESHOLD
    export GOOD_SIMILARITY_THRESHOLD
    export NUM_GENERATIONS
    
    echo "Start training for shard ${SHARD_IDX}..."
    TRAIN_ARGS=(rlhf 
        --rlhf_type grpo \
        --loss_type dapo \
        --epsilon_high 0.28 \
        --dynamic_sample true \
        --max_resample_times "${MAX_RESAMPLE_TIMES}" \
        --overlong_filter true \
        --model "${BASE_MODEL}" \
        --max_steps "${MAX_STEPS}" \
        --use_hf true \
        --model_type llama \
        --template llama3 \
        --dataset "checkpoint/${NAME}/data/shard_$(printf "%04d" "$SHARD_IDX").jsonl" \
        --external_plugins core/ablation/ablation_plugin.py \
        --reward_funcs molecule \
        --output_dir "checkpoint/${NAME}/shard_$(printf "%04d" "$SHARD_IDX")" \
        --tuner_type lora \
        --lora_rank 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --target_modules all-linear \
        --torch_dtype bfloat16 \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate "${SHARD_LR}" \
        --lr_scheduler_type constant \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 1 \
        --save_total_limit 1 \
        --dataset_shuffle false \
        --num_generations "${NUM_GENERATIONS}" \
        --generation_batch_size "${GENERATION_BATCH_SIZE}" \
        --max_completion_length "${MAX_COMPLETION_LENGTH}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --beta "${BETA}" \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host "${ROLLOUT_NODE}" \
        --vllm_server_port "${PORT}" \
        --vllm_server_group_port "${GROUP_PORT}" \
        --vllm_server_timeout 300 \
        --loss_scale last_round \
        --log_completions true \
        --log_entropy true \
        --report_to tensorboard \
    )
        if [[ -n "${CURRENT_ADAPTER}" ]]; then
            TRAIN_ARGS+=(--adapters "${CURRENT_ADAPTER}")
        fi

        CUDA_VISIBLE_DEVICES=0 swift "${TRAIN_ARGS[@]}"

        python utils/update_metrics.py \
            "checkpoint/${NAME}" \
            --plot-every 1 \

        CURRENT_ADAPTER="$(python utils/find_latest_adapter.py "checkpoint/${NAME}/shard_$(printf "%04d" "$SHARD_IDX")")"
        python utils/prune_checkpoints.py "checkpoint/${NAME}" --keep 3 --protect "${CURRENT_ADAPTER}"
done 

python utils/update_metrics.py "checkpoint/${NAME}" --force-plot
