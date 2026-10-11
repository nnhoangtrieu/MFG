#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo "ERROR: Command failed at line $LINENO"; exit 1' ERR

: "${CHECKPOINT:?Set CHECKPOINT to an ablation checkpoint (for example: CHECKPOINT=checkpoint/my_run/shard_0012/v0-.../checkpoint-1)}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="$(realpath "${CHECKPOINT}")"
cd "${SCRIPT_DIR}"

export LD_LIBRARY_PATH="${CONDA_PREFIX:?CONDA_PREFIX is not set}/lib:${LD_LIBRARY_PATH:-}"

CURRENT_ADAPTER="$(python utils/find_latest_adapter.py "${CHECKPOINT}")"
CURRENT_ADAPTER="$(realpath "${CURRENT_ADAPTER}")"

# Locate checkpoint/<NAME>/train_args.json from either a checkpoint, shard,
# or complete run directory supplied through CHECKPOINT.
RUN_DIR="${CURRENT_ADAPTER}"
while [[ "${RUN_DIR}" != "/" && ! -f "${RUN_DIR}/train_args.json" ]]; do
    RUN_DIR="$(dirname "${RUN_DIR}")"
done

if [[ ! -f "${RUN_DIR}/train_args.json" ]]; then
    echo "Could not find train_args.json above CHECKPOINT=${CHECKPOINT}" >&2
    exit 1
fi

CHECKPOINT_ROOT="$(realpath checkpoint)"
if [[ "$(dirname "${RUN_DIR}")" != "${CHECKPOINT_ROOT}" ]]; then
    echo "Expected the resumed run at checkpoint/<NAME>, found: ${RUN_DIR}" >&2
    exit 1
fi

NAME="$(basename "${RUN_DIR}")"
if [[ "${CURRENT_ADAPTER}" =~ /shard_([0-9]+)/ ]]; then
    LAST_SHARD=$((10#${BASH_REMATCH[1]}))
else
    echo "Could not infer a shard number from: ${CURRENT_ADAPTER}" >&2
    exit 1
fi
export NAME

declare -A SAVED_ARGS=()
while IFS=$'\t' read -r key value; do
    SAVED_ARGS["${key}"]="${value}"
done < <(python - "${RUN_DIR}/train_args.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    args = json.load(f)
for key, value in args.items():
    print(f"{key}\t{value}")
PY
)

saved_or_default() {
    local key="$1"
    local fallback="$2"
    printf '%s' "${SAVED_ARGS[${key}]:-${fallback}}"
}

# Server arguments can be overridden when reconnecting to the rollout server.
PORT="${PORT:-9001}"
GROUP_PORT="${GROUP_PORT:-51211}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
SEED="${SEED:-910}"

# Restore the original run configuration, while allowing explicit overrides.
BASE_MODEL="${BASE_MODEL:-/GeLLMO-P6-Llama}"
TRAIN_STEPS="${TRAIN_STEPS:-$(saved_or_default TRAIN_STEPS 1000)}"
NUM_GENERATIONS="${NUM_GENERATIONS:-$(saved_or_default NUM_GENERATIONS 4)}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$(saved_or_default GENERATION_BATCH_SIZE 24)}"
GRAD_ACCUM="${GRAD_ACCUM:-$(saved_or_default GRAD_ACCUM 1)}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-$(saved_or_default PER_DEVICE_BATCH 24)}"
LR="${LR:-$(saved_or_default LR 1e-5)}"
LR_MIN="${LR_MIN:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-$(saved_or_default LR_SCHEDULE cosine)}"
BETA="${BETA:-$(saved_or_default BETA 0.1)}"
TEMPERATURE="${TEMPERATURE:-$(saved_or_default TEMPERATURE 1.0)}"
TOP_P="${TOP_P:-$(saved_or_default TOP_P 0.9)}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-$(saved_or_default MAX_RESAMPLE_TIMES 3)}"
MULTI="${MULTI:-$(saved_or_default MULTI 5)}"
ACCEPTABLE_SIMILARITY_THRESHOLD="${ACCEPTABLE_SIMILARITY_THRESHOLD:-$(saved_or_default ACCEPTABLE_SIMILARITY_THRESHOLD 0.4)}"
GOOD_SIMILARITY_THRESHOLD="${GOOD_SIMILARITY_THRESHOLD:-$(saved_or_default GOOD_SIMILARITY_THRESHOLD 0.6)}"

MAX_STEPS=$((GENERATION_BATCH_SIZE * MULTI / (GRAD_ACCUM * PER_DEVICE_BATCH)))
SHARD_SIZE=$(((GENERATION_BATCH_SIZE / NUM_GENERATIONS) * MULTI))

RESUME_CHECKPOINT=""
CHECKPOINT_STEP=""
CHECKPOINT_MAX_STEPS=""
if [[ -f "${CURRENT_ADAPTER}/trainer_state.json" ]]; then
    read -r CHECKPOINT_STEP CHECKPOINT_MAX_STEPS < <(python - "${CURRENT_ADAPTER}/trainer_state.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    state = json.load(f)
print(state["global_step"], state["max_steps"])
PY
    )
fi

if [[ -n "${CHECKPOINT_STEP}" ]] && (( CHECKPOINT_STEP < CHECKPOINT_MAX_STEPS )); then
    START_SHARD=${LAST_SHARD}
    RESUME_CHECKPOINT="${CURRENT_ADAPTER}"
    echo "Checkpoint stopped at step ${CHECKPOINT_STEP} of ${CHECKPOINT_MAX_STEPS} in shard ${LAST_SHARD}."
else
    START_SHARD=$((LAST_SHARD + 1))
fi

if (( START_SHARD >= TRAIN_STEPS )); then
    echo "Nothing to resume: shard ${LAST_SHARD} is already the last of ${TRAIN_STEPS} shards."
    exit 0
fi

if [[ ! -f "data/${NAME}/train.jsonl" ]]; then
    echo "Missing existing data for run ${NAME}: data/${NAME}/train.jsonl" >&2
    exit 1
fi

echo "Using adapter: ${CURRENT_ADAPTER}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    echo "Resuming ablation run ${NAME} within shard ${START_SHARD}"
else
    echo "Resuming ablation run ${NAME} after shard ${LAST_SHARD}"
    echo "Continuing at shard ${START_SHARD} of ${TRAIN_STEPS}"
fi
echo "${SHARD_SIZE} samples will be used for each shard"
echo "${MAX_STEPS} training steps will be performed for each shard"

for ((SHARD_IDX=START_SHARD; SHARD_IDX<TRAIN_STEPS; SHARD_IDX++)); do
    SHARD_DATA="checkpoint/${NAME}/data/shard_$(printf "%04d" "${SHARD_IDX}").jsonl"
    if [[ -n "${RESUME_CHECKPOINT}" && -f "${SHARD_DATA}" ]]; then
        echo "Reusing existing data for interrupted shard ${SHARD_IDX}."
    else
        echo "Sampling ablation shard ${SHARD_IDX}..."
        python core/ablation/ablation_sample.py \
            --train_data "data/${NAME}/train.jsonl" \
            --shard_size "${SHARD_SIZE}" \
            --shard_idx "${SHARD_IDX}" \
            --seed "${SEED}" \
            --output_dir "checkpoint/${NAME}/data"
    fi

    SHARD_LR="$(python utils/get_shard_lr.py \
        --base_lr "${LR}" \
        --min_lr "${LR_MIN}" \
        --shard_index "${SHARD_IDX}" \
        --num_shards "${TRAIN_STEPS}" \
        --schedule "${LR_SCHEDULE}")"

    export SHARD_IDX
    export ACCEPTABLE_SIMILARITY_THRESHOLD
    export GOOD_SIMILARITY_THRESHOLD
    export NUM_GENERATIONS

    echo "Start training for shard ${SHARD_IDX}..."
    TRAIN_ARGS=(rlhf
        --rlhf_type grpo
        --loss_type dapo
        --epsilon_high 0.28
        --dynamic_sample true
        --max_resample_times "${MAX_RESAMPLE_TIMES}"
        --overlong_filter true
        --model "${BASE_MODEL}"
        --max_steps "${MAX_STEPS}"
        --use_hf true
        --model_type llama
        --template llama3
        --dataset "${SHARD_DATA}"
        --external_plugins core/ablation/ablation_plugin.py
        --reward_funcs molecule
        --output_dir "checkpoint/${NAME}/shard_$(printf "%04d" "${SHARD_IDX}")"
        --tuner_type lora
        --lora_rank 16
        --lora_alpha 32
        --lora_dropout 0.05
        --target_modules all-linear
        --torch_dtype bfloat16
        --per_device_train_batch_size "${PER_DEVICE_BATCH}"
        --gradient_accumulation_steps "${GRAD_ACCUM}"
        --learning_rate "${SHARD_LR}"
        --lr_scheduler_type constant
        --num_train_epochs 1
        --logging_steps 1
        --save_strategy steps
        --save_steps 1
        --save_total_limit 1
        --dataset_shuffle false
        --num_generations "${NUM_GENERATIONS}"
        --generation_batch_size "${GENERATION_BATCH_SIZE}"
        --max_completion_length 96
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --beta "${BETA}"
        --use_vllm true
        --vllm_mode server
        --vllm_server_host "${ROLLOUT_NODE}"
        --vllm_server_port "${PORT}"
        --vllm_server_group_port "${GROUP_PORT}"
        --vllm_server_timeout 300
        --loss_scale last_round
        --log_completions true
        --log_entropy true
        --report_to tensorboard
    )

    if [[ -n "${RESUME_CHECKPOINT}" ]]; then
        TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
    else
        TRAIN_ARGS+=(--adapters "${CURRENT_ADAPTER}")
    fi

    CUDA_VISIBLE_DEVICES=0 swift "${TRAIN_ARGS[@]}"

    python utils/update_metrics.py "checkpoint/${NAME}" --plot-every 1

    CURRENT_ADAPTER="$(python utils/find_latest_adapter.py \
        "checkpoint/${NAME}/shard_$(printf "%04d" "${SHARD_IDX}")")"
    RESUME_CHECKPOINT=""
    python utils/prune_checkpoints.py \
        "checkpoint/${NAME}" --keep 3 --protect "${CURRENT_ADAPTER}"
done

python utils/update_metrics.py "checkpoint/${NAME}" --force-plot
