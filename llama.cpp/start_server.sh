#!/bin/bash

# Get the actual model path from HF cache
MODEL_PATH=$(python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download( \
    repo_id='bartowski/google_gemma-3-4b-it-GGUF', \
    filename='google_gemma-3-4b-it-Q4_K_M.gguf'))")

# Start llama-server in a screen session
LLAMA_SERVER="/home/vscode/repos/llama.cpp/build/bin/llama-server"
screen -dmS gemma_server bash -c \
    "$LLAMA_SERVER -m $MODEL_PATH \
    --host ${LLAMA_SERVER_HOST:-0.0.0.0} \
    --port ${LLAMA_SERVER_PORT:-8080}"

echo "Server started in screen session 'gemma_server'"
echo "Model path: $MODEL_PATH"
echo "Connect at: http://${LLAMA_SERVER_HOST:-0.0.0.0}:${LLAMA_SERVER_PORT:-8080}"