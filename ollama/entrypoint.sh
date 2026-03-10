#!/bin/bash

set -e

echo "=== Ollama Container Entrypoint ==="
echo "Starting Ollama server with model management..."

# Set Ollama environment variables BEFORE starting the server
# OLLAMA_KEEP_ALIVE=-1 keeps model loaded indefinitely
export OLLAMA_KEEP_ALIVE=-1

echo "Ollama configuration:"
echo "  OLLAMA_KEEP_ALIVE=$OLLAMA_KEEP_ALIVE"
echo ""

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
    
    # Load the model into GPU memory using API call
    echo ""
    echo "Loading model into GPU memory..."
    if curl -s http://localhost:11434/api/generate -d '{"model": "'"$OLLAMA_MODEL"'", "prompt": "Hello", "stream": false}' > /dev/null 2>&1; then
        echo "Model loaded into GPU memory successfully!"
    else
        echo "Model loading completed"
    fi
    
    echo ""
    echo "Ollama server is running!"
    echo "API available at: http://localhost:11434"
    echo "Model '$OLLAMA_MODEL' is loaded and ready for inference!"
    echo ""
    
    # Start a background keep-alive process to periodically ping the model
    echo "Starting keep-alive background process..."
    (
        while true; do
            sleep 60  # Ping every 60 seconds
            echo "[Keep-alive] Pinging model to keep it loaded..."
            curl -s -X POST http://localhost:11434/api/generate \
                -H "Content-Type: application/json" \
                -d '{"model": "'"$OLLAMA_MODEL"'", "prompt": ".", "stream": false}' \
                > /dev/null 2>&1 || true
        done
    ) &
    KEEP_ALIVE_PID=$!
    
    echo "Keep-alive process started with PID: $KEEP_ALIVE_PID"
    echo ""
    
    # Keep the server running
    wait $OLLAMA_PID
    
    # Clean up keep-alive process when server stops
    kill $KEEP_ALIVE_PID 2>/dev/null || true
else
    echo "No OLLAMA_MODEL specified, starting Ollama server only"
    echo "To pull a model, set OLLAMA_MODEL environment variable"
    echo "Example: OLLAMA_MODEL=qwen3.5:35b-a3b"
    echo ""
    
    # Start Ollama server in foreground
    exec ollama serve
fi