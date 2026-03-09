#!/bin/bash

# Test script to check if Ollama server is running
# Usage: ./test_ollama.sh

set -e

echo "Testing Ollama server connection..."

# Check if the container is running
if docker ps --filter "name=ollama-server" --format "{{.Names}}" | grep -q "ollama-server"; then
    echo "✓ Ollama container is running"
else
    echo "✗ Ollama container is not running"
    echo "Start it with: docker-compose up -d"
    exit 1
fi

# Test the API endpoint
echo "Testing API endpoint on port 11434..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✓ Ollama API is responding"
    
    # Get the list of models
    echo "Available models:"
    curl -s http://localhost:11434/api/tags | jq -r '.models[]?.name' 2>/dev/null || echo "  (install jq for better formatting)"
else
    echo "✗ Ollama API is not responding"
    echo "Check if the server is starting up with: docker-compose logs ollama"
    exit 1
fi

# Test a simple completion if a model is available
echo ""
echo "Testing model availability..."
MODELS_RESPONSE=$(curl -s http://localhost:11434/api/tags)
if echo "$MODELS_RESPONSE" | grep -q '"models":\[\]'; then
    echo "⚠ No models found. You need to pull a model first."
    echo "To pull a model, run:"
    echo "  docker exec ollama-server ollama pull qwen3.5:35b-a3b"
    echo "Or modify the OLLAMA_MODEL environment variable in docker-compose.yml"
else
    echo "✓ Models are available"
    
    # Try a simple health check with the first model
    FIRST_MODEL=$(echo "$MODELS_RESPONSE" | jq -r '.models[0]?.name' 2>/dev/null || echo "")
    if [ -n "$FIRST_MODEL" ]; then
        echo "Testing model '$FIRST_MODEL' with a simple health check..."
        if curl -s http://localhost:11434/api/generate -d '{"model": "'"$FIRST_MODEL"'", "prompt": "Hello", "stream": false}' > /dev/null 2>&1; then
            echo "✓ Model '$FIRST_MODEL' is responding"
        else
            echo "⚠ Model '$FIRST_MODEL' health check failed (might still be loading)"
        fi
    fi
fi

echo ""
echo "Ollama server is ready!"
echo "You can interact with it at: http://localhost:11434"
echo "API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md"