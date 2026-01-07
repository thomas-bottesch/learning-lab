## Getting Started

Once inside the container, follow these steps to set up Ollama:

1. **Start the Ollama server in the background:**
    ```bash
    export OLLAMA_LOG_LEVEL=error
    screen -dmS ollama_server ollama serve
    ```

2. **Pull the `orca-mini:3b` model (run this only once):**
    ```bash
    ollama pull orca-mini:3b
    ```

3. **Run ollama_test.py to verify the setup:**
    ```bash
    python ollama_test.py
    ```