#!/usr/bin/env bash
set -euo pipefail

# Download NInfer models into models/ directory
# Requires: huggingface-cli installed (pip install huggingface_hub)
#
# Usage:
#   ./scripts/download-model.sh          # Download Qwen3.8-27B NVFP4 (default)
#   ./scripts/download-model.sh qwen3.6  # Download Qwen3.6-35B-A3B
#   ./scripts/download-model.sh qwen3.8  # Download Qwen3.8-27B NVFP4

MODEL_DIR="models"
MODEL="${1:-qwen3.8}"

case "$MODEL" in
  qwen3.6)
    MODEL_REPO="neroued/Qwen3.6-35B-A3B-NInfer"
    MODEL_FILE="qwen3_6_35b_a3b.ninfer"
    ;;
  qwen3.8)
    MODEL_REPO="neroued/Qwen3.8-27B-nvfp4-NInfer"
    MODEL_FILE="qwen3_8_27b_nvfp4.ninfer"
    ;;
  *)
    echo "Error: Unknown model '$MODEL'"
    echo "Available models: qwen3.6, qwen3.8"
    exit 1
    ;;
esac

echo "=== NInfer Model Download Script ==="
echo ""
echo "Model repository: $MODEL_REPO"
echo "Model file: $MODEL_FILE"
echo "Download directory: $MODEL_DIR"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Download the model using huggingface-cli
echo "Downloading model to $MODEL_DIR/$MODEL_FILE ..."
hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"

echo ""
echo "Model downloaded successfully to $MODEL_DIR/$MODEL_FILE"
ls -lh "$MODEL_DIR/$MODEL_FILE"
