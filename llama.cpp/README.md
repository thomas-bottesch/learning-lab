# LlamaCpp LLM Integration Project

A comprehensive project demonstrating integration with llama.cpp server for running local LLM models. This repository includes a testing framework and a story analysis application built with LangChain.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
  - [Starting the Server](#starting-the-server)
  - [Testing the Integration](#testing-the-integration)
  - [Story Analyzer](#story-analyzer)

## Overview

This project demonstrates how to:
- Run a local LLM using llama.cpp server with the Gemma 3 4B model
- Connect to the server via OpenAI-compatible API
- Build applications using LangChain with local LLMs
- Implement LLM-based validation workflows

The repository includes two main components:
1. **Direct Test Script** - Validates the llama-server integration and measures response times
2. **Story Analyzer** - A LangChain-based application that extracts metadata (characters, genre) from story files with built-in LLM validation

## Project Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── start_server.sh                     # Server startup script
├── llamacpp_direct_test.py            # Integration test script
└── projects/
    └── story_analyzer/
        ├── analyzer.py                 # Story analysis application
        ├── README.md                   # Story analyzer documentation
        ├── chatgpt_generated_story_1   # Sample story file
        ├── chatgpt_generated_story_2   # Sample story file
        └── chatgpt_generated_story_3   # Sample story file
```

## Prerequisites

- **llama.cpp**: Built and compiled with the `llama-server` binary
  - Expected location: `/home/vscode/repos/llama.cpp/build/bin/llama-server`
  - If your installation is elsewhere, update [start_server.sh](start_server.sh)
  
- **Python 3.8+**: For running the scripts

- **screen**: For running the server in the background
  ```bash
  sudo apt-get install screen
  ```

- **Hugging Face CLI**: For downloading models (installed via requirements.txt)

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain` and `langchain-openai` - LLM application framework
- `openai` - For OpenAI-compatible API access
- `huggingface_hub` - For downloading models
- Additional utilities for PDF/web scraping and vector search

### 2. Download the Model

The project uses the Gemma 3 4B model (quantized Q4_K_M version):

```bash
# The start_server.sh script will automatically download the model on first run
# Or manually download with:
huggingface-cli download bartowski/google_gemma-3-4b-it-GGUF google_gemma-3-4b-it-Q4_K_M.gguf
```

### 3. Configure Server Path

Edit [start_server.sh](start_server.sh) if your llama-server binary is not at the default location:

```bash
LLAMA_SERVER="/path/to/your/llama.cpp/build/bin/llama-server"
```

## Usage

### Starting the Server

Start the llama-server in a detached screen session:

```bash
bash start_server.sh
```

This will:
- Download the model from Hugging Face (if not already cached)
- Start llama-server on `0.0.0.0:8080` by default
- Run in a screen session named `gemma_server`

**Server Management:**
```bash
# View server logs
screen -r gemma_server

# Detach from screen (Ctrl+A, then D)

# Stop the server
screen -X -S gemma_server quit
```

### Testing the Integration

Test the server connection and measure response time:

```bash
python llamacpp_direct_test.py
```

**Sample Output:**
```
Invoking LLM with message: 'Please sort the following numbers. 3,2,1,4,5'

Direct invoke output:
1, 2, 3, 4, 5

Direct invoke took 2.34 seconds.
```

### Story Analyzer

The story analyzer extracts characters and genre from story files using LangChain with a two-stage LLM validation approach.

**Basic Usage:**

```bash
cd projects/story_analyzer
python analyzer.py chatgpt_generated_story_1
```

**Output Format:**
```json
{
  "characters": ["Orion", "copper-brown skinned girl"],
  "genre": ["science-fiction", "fantasy"]
}
```

**How It Works:**

1. **Analysis Stage**: The LLM reads the story and extracts:
   - List of character names
   - Genre classification(s)

2. **Validation Stage**: A second LLM instance validates:
   - JSON structure completeness
   - Genre appropriateness
   - Character list accuracy

3. **Retry Logic**: Automatically retries up to 3 times on failure

