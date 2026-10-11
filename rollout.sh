BASE_MODEL="${BASE_MODEL:-/GeLLMO-P6-Llama}"
PORT="${PORT:-9001}"

CUDA_VISIBLE_DEVICES=0 swift rollout \
    --model "${BASE_MODEL}" \
    --model_type llama \
    --template llama3 \
    --use_hf true \
    --vllm_enable_lora true \
    --vllm_max_lora_rank 16 \
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.7 \
    --port "${PORT}"