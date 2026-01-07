# This script tests if the Ollama LLM integration works correctly
# by invoking the model and measuring response time.

from langchain_ollama import OllamaLLM
import time

llm = OllamaLLM(model="orca-mini:3b", temperature=0)

# Measure direct invoke speed
start_direct = time.time()
invoke_message = "Please sort the following numbers. You should write 3,2,1,4,5"
print(f"Invoking LLM with message: '{invoke_message}'")
response = llm.invoke(invoke_message)
end_direct = time.time()
print("\n\nDirect invoke output:")
print(response)
print(f"\nDirect invoke took {end_direct - start_direct:.2f} seconds.\n")
