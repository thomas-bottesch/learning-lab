# Ollama Docker Setup

This setup provides a Docker-based Ollama server with GPU support and persistent model storage.

## Features

- **GPU Acceleration**: Uses NVIDIA CUDA for GPU acceleration (if available)
- **Persistent Storage**: Models are stored in `./ollama_models/` directory
- **Easy Model Switching**: Change models by modifying `start-ollama.sh`
- **Port Forwarding**: Port 11434 is exposed to the host
- **VS Code DevContainer**: Can be used as a VS Code DevContainer

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **NVIDIA Docker** (optional, for GPU acceleration)
3. **VS Code** with **Dev Containers** extension (optional, for DevContainer usage)

## Quick Start

### Option 1: Using the start script (recommended)

```bash
# Make scripts executable
chmod +x start-ollama.sh start.sh stop.sh

# Start with default model (qwen3.5:35b-a3b)
./start.sh

# Or use the full control script
./start-ollama.sh
```

### Option 2: Using Docker Compose directly

```bash
# Build and start the container
docker-compose up --build

# Stop the container
docker-compose down
```

## Changing the Model

To use a different model, edit the `start-ollama.sh` file and change the `MODEL` variable:

```bash
# Change this line in start-ollama.sh
MODEL="llama3.2:3b"  # or any other model from https://ollama.com/library
```

Available models can be found at: https://ollama.com/library

## Accessing the Ollama Server

Once running, the Ollama server is available at:
- **HTTP API**: `http://localhost:11434`
- **Web UI**: `http://localhost:11434` (Ollama's web interface)

## Using as a VS Code DevContainer

1. Open the folder in VS Code
2. Press `F1` and select "Dev Containers: Reopen in Container"
3. The container will build and start with Ollama pre-installed

## File Structure

```
.
├── docker-compose.yml          # Docker Compose configuration
├── start-ollama.sh            # Main start script with model configuration
├── start.sh                   # Quick start script
├── stop.sh                    # Stop script
├── .devcontainer/             # VS Code DevContainer configuration
│   ├── Dockerfile            # Docker image with Ollama
│   └── devcontainer.json     # DevContainer settings
└── ollama_models/            # Persistent model storage (created automatically)
```

## Environment Variables

- `OLLAMA_HOST=0.0.0.0` - Allows external connections
- `OLLAMA_ORIGINS=*` - Allows all origins (CORS)
- `OLLAMA_MODEL` - Set by start script to control which model to use

## Persistent Storage

Models are stored in `./ollama_models/` directory on your host machine. This means:
- Models persist between container restarts
- Models are not lost when the container is deleted
- You can share models between different containers

## Troubleshooting

### GPU Not Detected
If you see "Warning: NVIDIA Docker runtime not detected", install NVIDIA Docker:
```bash
# Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Port Already in Use
If port 11434 is already in use, modify the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "11435:11434"  # Change 11435 to any available port
```

### Model Download Issues
If model download fails, check your internet connection and try:
```bash
# Pull the model manually inside the container
docker exec -it ollama-server ollama pull <model-name>
```

## Useful Commands

```bash
# Check container logs
docker-compose logs

# Enter the running container
docker exec -it ollama-server bash

# List downloaded models
docker exec -it ollama-server ollama list

# Pull additional models
docker exec -it ollama-server ollama pull <model-name>

# Remove the container and images (keeps models)
docker-compose down --rmi all
```