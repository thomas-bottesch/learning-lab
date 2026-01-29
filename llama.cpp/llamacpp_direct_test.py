# This script tests if the LlamaCpp server integration works correctly
# by invoking the model via OpenAI-compatible API and measuring response time.

import os
import time
from openai import OpenAI

# Connect to llama-server (OpenAI-compatible API)
llama_host = os.getenv("LLAMA_SERVER_HOST", "localhost")
llama_port = os.getenv("LLAMA_SERVER_PORT", "8080")
base_url = f"http://{llama_host}:{llama_port}/v1"

client = OpenAI(
    base_url=base_url,
    api_key="not-needed",  # llama-server doesn't require API key
)

# Measure direct invoke speed
start_direct = time.time()
invoke_message = "Please sort the following numbers. 3,2,1,4,5"
print(f"Invoking LLM with message: '{invoke_message}'")

response = client.chat.completions.create(
    model="llama-model",  # Model name doesn't matter for llama-server
    messages=[{"role": "user", "content": invoke_message}],
    temperature=0.7,
    max_tokens=512,
)

end_direct = time.time()
print("\n\nDirect invoke output:")
print(response.choices[0].message.content)
print(f"\nDirect invoke took {end_direct - start_direct:.2f} seconds.\n")
