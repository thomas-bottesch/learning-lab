#!/bin/bash

set -e

echo "=== vLLM Container Entrypoint ==="

MODEL="${VLLM_MODEL:?VLLM_MODEL environment variable must be set}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.95}"

echo "Model:                  $MODEL"
echo "Max model length:       $MAX_MODEL_LEN"
echo "GPU memory utilization: $GPU_MEM"
echo ""
echo "Model will be downloaded from HuggingFace on first run."
echo "API will be available at: http://localhost:11434"
echo ""

# if MODEL == palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 then have some custom settings for that model
#  --override-generation-config '{"temperature": 0.8, "top_k": -1, "min_p": 0.05, "frequency_penalty": 0.3, "top_p": 0.95}'
#  --served-model-name palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
#  --enable-auto-tool-choice
#  --tool-call-parser qwen3_coder
#  --quantization gptq_marlin
#  --tensor-parallel-size 1
#  --max-model-len 131072
#  --max-num-seqs 1
#  --max-num-batched-tokens 4096
#  --kv-cache-dtype fp8
#  --enable-prefix-caching
#  --gpu-memory-utilization 0.98
#  --dtype auto
#  --trust-remote-code

HOST=0.0.0.0
PORT=11434

if [ "$MODEL" = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" ]; then
    GPU_MEM=0.98
    MAX_MODEL_LEN=180072
    echo "Using custom settings for model: $MODEL"
    exec python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --override-generation-config '{"temperature": 0.8, "top_k": -1, "min_p": 0.05, "frequency_penalty": 0.3, "top_p": 0.95}' \
        --served-model-name palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --quantization gptq_marlin \
        --tensor-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 1 \
        --max-num-batched-tokens 4096 \
        --kv-cache-dtype fp8 \
        --enable-prefix-caching \
        --gpu-memory-utilization "$GPU_MEM" \
        --dtype auto \
        --trust-remote-code
else
    exec python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 1 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --kv-cache-dtype fp8 \
        --tensor-parallel-size 1 \
        --no-enable-log-requests
fi
