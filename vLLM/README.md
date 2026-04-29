# vLLM Docker Setup

This setup provides a Docker-based vLLM server with GPU support and persistent model storage.

## Features

- **GPU Acceleration**: Full NVIDIA GPU support via NVIDIA Container Toolkit
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API on port 11434
- **Persistent Storage**: Models are cached in `./vllm_models/` (downloaded once from HuggingFace)
- **Easy Model Switching**: Change the `VLLM_MODEL` variable in `docker-compose.yml`
- **Single-User Optimized**: Tuned for maximum speed on one machine, one user

## Quick Start

```bash
docker compose build
docker compose up
```

The server downloads the configured model on first run, then serves the API at `http://localhost:11434`.

## Switching Models

Edit `VLLM_MODEL` in `docker-compose.yml`:

```yaml
environment:
  - VLLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
```

Use any HuggingFace model ID. For gated models, also uncomment and set `HF_TOKEN`.

## Accessing the vLLM Server

Once running, the API is available at `http://localhost:11434` (OpenAI-compatible):

```bash
# List loaded models
curl http://localhost:11434/v1/models

# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-Coder-32B-Instruct-AWQ","messages":[{"role":"user","content":"Hello"}]}'
```

## Testing

```bash
./test_vllm.sh
```

## Directory Structure

```
vLLM/
├── docker-compose.yml         # Main config — edit VLLM_MODEL here
├── entrypoint.sh              # Container startup script
├── test_vllm.sh               # Health check script
├── .devcontainer/
│   └── Dockerfile             # Docker image based on vllm/vllm-openai
└── vllm_models/               # Persistent HuggingFace model cache
```

## Performance Settings (RTX 5090, single user)

| Flag | Value | Reason |
|---|---|---|
| `--dtype bfloat16` | BF16 | Blackwell native precision |
| `--gpu-memory-utilization` | 0.95 | 95% of 32 GB VRAM |
| `--max-num-seqs` | 1 | All compute to one request |
| `--enable-prefix-caching` | on | Caches repeated system prompts (big win for coding) |
| `--kv-cache-dtype fp8` | FP8 | Halves KV cache memory, longer context |
| `--enable-chunked-prefill` | on | Lower first-token latency |

Adjust `MAX_MODEL_LEN` and `GPU_MEMORY_UTILIZATION` in `docker-compose.yml` as needed.

## Useful Commands

```bash
# Check container logs
docker compose logs vllm

# Enter the running container
docker exec -it vllm-server bash

# Stop and remove the container
docker compose down

# Remove container and image (keeps cached models)
docker compose down --rmi all
```
