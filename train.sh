#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo "ERROR: Command failed at line $LINENO"; exit 1' ERR

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Server arguments
PORT="${PORT:-9001}"
GROUP_PORT="${GROUP_PORT:-51211}"
ROLLOUT_NODE="${ROLLOUT_NODE:-a000}"
SEED="${SEED:-910}"
NAME="${NAME:-run$(date +%m%d_%H%M%S)}"

# Precompute arguments
MIN_CONTEXT_TURNS="${MIN_CONTEXT_TURNS:-0}"
MAX_CONTEXT_TURNS="${MAX_CONTEXT_TURNS:-1}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-48}"

# LoRA arguments
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"

# Training arguments
BASE_MODEL="${BASE_MODEL:-/GeLLMO-P6-Llama}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-10}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-20}"
LR="${LR:-1e-5}"
LR_MIN="${LR_MIN:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
BETA="${BETA:-0.1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-96}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}"
GENERATION_MODE="${GENERATION_MODE:-sampling}"
MULTI="${MULTI:-10}"
MAX_STEPS=$((GENERATION_BATCH_SIZE * MULTI / (GRAD_ACCUM * PER_DEVICE_BATCH)))
SHARD_SIZE=$((((GENERATION_BATCH_SIZE / NUM_GENERATIONS)) * MULTI))
ACCEPTABLE_SIMILARITY_THRESHOLD="${ACCEPTABLE_SIMILARITY_THRESHOLD:-0.4}"
GOOD_SIMILARITY_THRESHOLD="${GOOD_SIMILARITY_THRESHOLD:-0.6}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-100}"


echo "$((SHARD_SIZE)) samples will be used for each shard"
echo "$((MAX_STEPS)) training steps will be performed for each shard"


python core/data.py \
    --exclude MPQ BDMQ BHMQ BMPQ HMPQ \
    --output "data/${NAME}" \
    --seed "${SEED}" \
    --samples_per_task "${SAMPLES_PER_TASK}"


CURRENT_ADAPTER="${CURRENT_ADAPTER:-}"

mkdir -p "checkpoint/${NAME}"
cat > "checkpoint/${NAME}/train_args.json" <<EOF
{
    "NUM_GENERATIONS": "${NUM_GENERATIONS}",
    "GENERATION_BATCH_SIZE": "${GENERATION_BATCH_SIZE}",
    "GRAD_ACCUM": "${GRAD_ACCUM}",
    "PER_DEVICE_BATCH": "${PER_DEVICE_BATCH}",
    "LORA_RANK": "${LORA_RANK}",
    "LORA_ALPHA": "${LORA_ALPHA}",
    "LORA_DROPOUT": "${LORA_DROPOUT}",
    "TARGET_MODULES": "${TARGET_MODULES}",
    "MIN_CONTEXT_TURNS": "${MIN_CONTEXT_TURNS}",
    "MAX_CONTEXT_TURNS": "${MAX_CONTEXT_TURNS}",
    "TRAIN_STEPS": "${TRAIN_STEPS}",
    "MAX_COMPLETION_LENGTH": "${MAX_COMPLETION_LENGTH}",
    "ACCEPTABLE_SIMILARITY_THRESHOLD": "${ACCEPTABLE_SIMILARITY_THRESHOLD}",
    "GOOD_SIMILARITY_THRESHOLD": "${GOOD_SIMILARITY_THRESHOLD}",
    "LR": "${LR}",
    "LR_SCHEDULE": "${LR_SCHEDULE}",
    "BETA": "${BETA}",
    "TEMPERATURE": "${TEMPERATURE}",
    "TOP_P": "${TOP_P}",
    "MAX_RESAMPLE_TIMES": "${MAX_RESAMPLE_TIMES}",
    "GENERATION_MODE": "${GENERATION_MODE}",
    "MULTI": "${MULTI}",
    "SAMPLES_PER_TASK": "${SAMPLES_PER_TASK}",
    "EXCLUDE": "MPQ BDMQ BHMQ BMPQ HMPQ"
}
EOF

for ((SHARD_IDX=0; SHARD_IDX<TRAIN_STEPS; SHARD_IDX++)); do 

    echo "Precomputing shard ${SHARD_IDX}..."
    python core/precompute_batch.py \
        --name "${NAME}" \
        --train_data "data/${NAME}/train.jsonl" \
        --shard_size "${SHARD_SIZE}" \
        --shard_idx "${SHARD_IDX}" \
        --path_base_model "${BASE_MODEL}" \
        --path_adapter "${CURRENT_ADAPTER}" \
        --min_context_turns "${MIN_CONTEXT_TURNS}" \
        --max_context_turns "${MAX_CONTEXT_TURNS}" \
        --seed "$((${SEED} + ${SHARD_IDX}))" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --output_dir "checkpoint/${NAME}/data" \
        --generation_mode "${GENERATION_MODE}" \
        --batch_size "${PRECOMPUTE_BATCH_SIZE}" \
        --acceptable_similarity_threshold "${ACCEPTABLE_SIMILARITY_THRESHOLD}" \
        --good_similarity_threshold "${GOOD_SIMILARITY_THRESHOLD}"

    SHARD_LR="$(python utils/get_shard_lr.py --base_lr "${LR}" --min_lr "${LR_MIN}" --shard_index "${SHARD_IDX}" --num_shards "${TRAIN_STEPS}" --schedule "${LR_SCHEDULE}")"
    
    export ACCEPTABLE_SIMILARITY_THRESHOLD
    export GOOD_SIMILARITY_THRESHOLD
    export NUM_GENERATIONS

    echo "Start training for shard ${SHARD_IDX}..."
    TRAIN_ARGS=(rlhf 
        --use_hf true \
        --model_type llama \
        --template llama3 \
        --tuner_type lora \
        --rlhf_type grpo \
        --loss_type dapo \
        --epsilon_high 0.28 \
        --dynamic_sample true \
        --overlong_filter true \
        --max_resample_times "${MAX_RESAMPLE_TIMES}" \
        --model "${BASE_MODEL}" \
        --max_steps "${MAX_STEPS}" \
        --external_plugins core/plugin.py \
        --reward_funcs molecule \
        --lr_scheduler_type constant \
        --learning_rate "${SHARD_LR}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --target_modules "${TARGET_MODULES}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --num_generations "${NUM_GENERATIONS}" \
        --generation_batch_size "${GENERATION_BATCH_SIZE}" \
        --max_completion_length "${MAX_COMPLETION_LENGTH}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --beta "${BETA}" \
        --dataset "checkpoint/${NAME}/data/shard_$(printf "%04d" "$SHARD_IDX").jsonl" \
        --output_dir "checkpoint/${NAME}/shard_$(printf "%04d" "$SHARD_IDX")" \
        --torch_dtype bfloat16 \
        --save_strategy steps \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --save_steps 1 \
        --save_total_limit 1 \
        --dataset_shuffle false \
        --loss_scale last_round \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host "${ROLLOUT_NODE}.ten.osc.edu" \
        --vllm_server_port "${PORT}" \
        --vllm_server_group_port "${GROUP_PORT}" \
        --vllm_server_timeout 300 \
        --log_completions true \
        --log_entropy true \
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
