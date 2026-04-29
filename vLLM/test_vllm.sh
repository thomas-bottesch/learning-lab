#!/bin/bash

# Test script to check if vLLM server is running
# Usage: ./test_vllm.sh

set -e

echo "Testing vLLM server connection..."

# Check if the container is running
if docker ps --filter "name=vllm-server" --format "{{.Names}}" | grep -q "vllm-server"; then
    echo "✓ vLLM container is running"
else
    echo "✗ vLLM container is not running"
    echo "Start it with: docker compose up -d"
    exit 1
fi

# Test the health endpoint
echo "Testing API endpoint on port 11434..."
if curl -sf http://localhost:11434/health > /dev/null; then
    echo "✓ vLLM API is responding"
else
    echo "✗ vLLM API is not responding"
    echo "Check if the server is starting up with: docker compose logs vllm"
    exit 1
fi

# List available models
echo "Available models:"
curl -s http://localhost:11434/v1/models | python3 -m json.tool 2>/dev/null || curl -s http://localhost:11434/v1/models

echo ""
echo "vLLM server is ready!"
echo "API available at: http://localhost:11434 (OpenAI-compatible)"
MODEL=$(curl -s http://localhost:11434/v1/models | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo '')
if [ -z "$MODEL" ]; then
    echo "✗ Could not determine model name"
    exit 1
fi

echo "Testing model '$MODEL' with a Hello message..."
RESPONSE=$(curl -s http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}")
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
