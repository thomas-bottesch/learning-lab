#!/bin/bash

set -e

echo "=== Ollama Container Entrypoint ==="
echo "Starting Ollama server with model management..."

# Ensure Ollama models directory exists and has correct permissions
echo "Setting up Ollama models directory..."
mkdir -p /home/vscode/.ollama/models
chown -R vscode:vscode /home/vscode/.ollama

# Check if OLLAMA_MODEL environment variable is set
if [ -n "$OLLAMA_MODEL" ]; then
    echo "Model specified: $OLLAMA_MODEL"
    
    # Start Ollama server in background
    echo "Starting Ollama server in background..."
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for server to be ready with retry logic
    echo "Waiting for Ollama server to start..."
    max_attempts=30
    attempt=1
    while ! curl -sf http://localhost:11434/ >/dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            echo "ERROR: Ollama server failed to start after $max_attempts attempts"
            kill $OLLAMA_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo "Ollama server is ready!"
    
    # Pull the specified model
    echo "Pulling model: $OLLAMA_MODEL"
    if ollama pull "$OLLAMA_MODEL"; then
        echo "Model pull completed successfully"
    else
        echo "Model pull failed or model already exists"
    fi
    
    # List available models
    echo "Available models:"
    ollama list
    
    echo ""
    echo "Ollama server is running!"
    echo "API available at: http://localhost:11434"
    echo ""
    
    # Keep the server running
    wait $OLLAMA_PID
else
    echo "No OLLAMA_MODEL specified, starting Ollama server only"
    echo "To pull a model, set OLLAMA_MODEL environment variable"
    echo "Example: OLLAMA_MODEL=qwen3.5:35b-a3b"
    echo ""
    
    # Start Ollama server in foreground
    exec ollama serve
fi