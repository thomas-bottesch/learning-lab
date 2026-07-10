#!/bin/bash

set -e

echo "=== vLLM + Open WebUI Container Entrypoint ==="

# print vllm version
echo "vLLM version: $(python3 -c 'import vllm; print(getattr(vllm, "__version__", "unknown"))')"

MODEL="${VLLM_MODEL:?VLLM_MODEL environment variable must be set}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.95}"

# Model-specific overrides (applied before VLLM_ARGS is built)
case "$MODEL" in
    palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4)
        GPU_MEM=0.98
        MAX_MODEL_LEN=180072
        echo "Using custom settings for model: $MODEL"
        VLLM_EXTRA_ARGS=(
            --override-generation-config '{"temperature": 0.8, "top_k": -1, "min_p": 0.05, "frequency_penalty": 0.3, "top_p": 0.95}'
            --served-model-name palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
            --enable-auto-tool-choice
            --tool-call-parser qwen3_coder
            --quantization gptq_marlin
            --max-num-batched-tokens 4096
            --dtype auto
            --trust-remote-code
        )
        ;;
    nvidia/Qwen3.6-35B-A3B-NVFP4)
        GPU_MEM=0.98
        MAX_MODEL_LEN=140000
        echo "Using custom settings for model: $MODEL (NVFP4 quantized with modelopt)"
        VLLM_EXTRA_ARGS=(
            --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4
            --enable-auto-tool-choice
            --tool-call-parser qwen3_coder
            --quantization modelopt
            --dtype auto
            --trust-remote-code
            --moe-backend marlin
            --attention-backend flashinfer
            --max-num-batched-tokens 8192
            --max-num-seqs 4
            --async-scheduling
            --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
        )
        ;;
    *)
        VLLM_EXTRA_ARGS=()
        ;;
esac

echo "Model:                  $MODEL"
echo "Max model length:       $MAX_MODEL_LEN"
echo "GPU memory utilization: $GPU_MEM"
echo ""
echo "Model will be downloaded from HuggingFace on first run."
echo "vLLM API will be available at: http://localhost:11434"
echo "Open WebUI will be available at: http://localhost:8080"
echo ""

# =============================================================================
# 1. Start vLLM server in background
# =============================================================================
echo "[1/2] Starting vLLM server..."

# Build vLLM command arguments
VLLM_ARGS=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --host 0.0.0.0
    --port 11434
    --dtype bfloat16
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs 3
    --enable-prefix-caching
    --enable-chunked-prefill
    --kv-cache-dtype fp8
    --tensor-parallel-size 1
    --no-enable-log-requests
    "${VLLM_EXTRA_ARGS[@]}"
)

# Start vLLM in background
"${VLLM_ARGS[@]}" &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to start..."
max_attempts=240
attempt=1
while ! curl -sf http://localhost:11434/v1/models >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
        echo "ERROR: vLLM server failed to start after $max_attempts attempts"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
    attempt=$((attempt + 1))
    if [ $((attempt % 10)) -eq 0 ]; then
        echo "  Still waiting... (attempt $attempt/$max_attempts)"
    fi
done

echo "vLLM server is ready!"
echo ""

# =============================================================================
# 2. Configure and start Open WebUI
# =============================================================================
echo "[2/2] Starting Open WebUI..."
echo "      UI will be available at: http://localhost:8080"
echo "      Backend: http://localhost:11434 (vLLM)"
echo ""

# Set Open WebUI environment variables
# Point Open WebUI to the local vLLM instance

# Enable "open mode" - no authentication required, works out of the box
export WEBUI_AUTH="False"
export ENABLE_REALTIME_WS="${ENABLE_REALTIME_WS:-false}"

# vLLM API configuration
export WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY:-$(openssl rand -hex 32)}"
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-http://localhost:11434/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-no-key-required}"

# Pre-configure the default model for Open WebUI
export DEFAULT_SYSTEM_PROMPT=""
export WEBUI_NAME="${WEBUI_NAME:-Open WebUI}"

# Disable the Open WebUI landing/welcome page, changelog, and community sharing
export ENABLE_LANDING_PAGE="False"
export SHOW_CHANGELOG="False"
export ENABLE_COMMUNITY_SHARING="False"

# Bypass model access control so all users can see all models
export BYPASS_MODEL_ACCESS_CONTROL="True"

# Set default model(s) to show in Open WebUI
export DEFAULT_MODELS="${DEFAULT_MODELS:-$VLLM_MODEL}"

# Start Open WebUI in foreground (this will keep the container running)
exec open-webui serve \
    --host 0.0.0.0 \
    --port 8080
