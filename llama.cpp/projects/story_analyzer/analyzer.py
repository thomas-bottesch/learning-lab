#!/usr/bin/env python3
"""
Story analyzer using Ollama and LangChain
"""

import os
import sys
import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser


# --- LLM Validation Step ---
def validate_output(original_prompt, content, response):
    # Connect to llama-server
    llama_host = os.getenv("LLAMA_SERVER_HOST", "localhost")
    llama_port = os.getenv("LLAMA_SERVER_PORT", "8080")
    base_url = f"http://{llama_host}:{llama_port}/v1"

    llm_validator = ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",
        model="llama-model",
        temperature=0.0,
        max_tokens=100,
    )
    validation_prompt = f"""
    Task: You are the supervisor of your company and your task is to check the work of your workers.
    Your worker had to process the following requenst encapsulated between <START_ORIGINAL_PROMPT>
    and finished by <END_ORIGINAL_PROMPT> on the given content that starts with
    <BEGIN_CONTENT> and ends with <END_CONTENT>.The response that the worker gave is between <START_RESPONSE>
    and <END_RESPONSE>.
    <START_ORIGINAL_PROMPT>{original_prompt}<END_ORIGINAL_PROMPT>
    <START_CONTENT>{content}<END_CONTENT>
    <START_RESPONSE>{response}<END_RESPONSE>

    Does the json output contain all fields that the examples had?
    Does the genre make sense?
    Are all characters from the story in the values list from characters?

    If everything is correct then just answer with: yes. Do not answer every single question just one yes!
    Else describe what is missing.
    
    Here some examples for valid and invalid answers:

    Your answer:yes  (correct format)
    Your answer:yes  (correct format)
    Your answer:yes the genre matches, all fields present (incorrect answer. wrong format. too much info!)
    
    If something is incorrect then:

    Your answer:no, its invalid because of reason x,y,z (correct format)

    Now its your turn to answer about the workers response from above:

    """
    validation_response = llm_validator.invoke(validation_prompt)

    # Extract content from AIMessage
    validation_text = (
        validation_response.content
        if hasattr(validation_response, "content")
        else str(validation_response)
    )

    l_validation_response = validation_text.lower()
    validation_success = l_validation_response.startswith(
        "yes"
    ) or l_validation_response.startswith("your answer:yes")

    if not validation_success:
        raise RuntimeError(f"{l_validation_response}")

    return validation_success


def cleanup_response(response: str) -> str:
    """
    Remove leading and trailing triple backticks and 'json' if present.
    """
    response = response.strip()
    if response.startswith("```json"):
        response = response[len("```json") :].lstrip()
    elif response.startswith("```"):
        response = response[len("```") :].lstrip()
    if response.endswith("```"):
        response = response[: -len("```")].rstrip()
    return response


def extract_story_metadata(story_file_path: str) -> list[str]:
    """
    Extract keywords from a webpage using Ollama LLM.

    Args:
        url: The URL of the webpage to analyze
        num_keywords: Number of keywords to extract (default: 10)

    Returns:
        A list of keyword strings
    """
    # Load the webpage content with retry logic

    print(f"Loading content from: {story_file_path}", file=sys.stderr)
    with open(story_file_path, "r") as f:
        content = f.read()

    # Connect to llama-server
    llama_host = os.getenv("LLAMA_SERVER_HOST", "localhost")
    llama_port = os.getenv("LLAMA_SERVER_PORT", "8080")
    base_url = f"http://{llama_host}:{llama_port}/v1"

    # Initialize LLM with slightly higher temperature for creativity
    llm = ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",
        model="llama-model",
        temperature=0.7,
        max_tokens=150,
    )
    parser = JsonOutputParser()

    # Create a prompt for keyword extraction with multiple examples
    prompt = f"""

Look at these example:

Example 1:
<START_STORY>
Captain Ryn, engineer Sol, and the AI named Echo drifted through space aboard a damaged research ship.
When a strange blue star began pulling them off course, Echo predicted it was artificial.
Sol rewired the engines while Ryn negotiated with the star’s hidden intelligence through light signals.
The star released them, revealing itself as an ancient guardian testing travelers.
As they escaped, the crew realized the universe was far more alive than they had ever imagined.
<END_STORY>
Json:
{{ 
  "characters": ["Captain Ryn", "Sol", "Echo"], 
  "genre": ["science-fiction"] 
}}

Example 2:
<START_STORY>
Elin lived alone at the edge of a forest where trees whispered names.
One night, the forest spoke hers clearly for the first time.
Following the sound, she found a glowing seed pulsing with warmth.
When she planted it, the forest fell silent in peace.
Elin understood she had become its new guardian.
<END_STORY>
Json:
{{ 
  "characters": ["Elin"], 
  "genre": ["fantasy", "drama"] 
}}

Now it's your turn.

Task: Retrieve the characters and the genre(s) from a story which starts with <START_STORY>
and ends with <END_STORY> and provide them in json format as the examples above:

So the json should be a dict and only contain two keys "characters" and "genre". Both of which are lists!

Content:
{content}

Json:
"""
    # LLM invocation with retry logic
    print("Analyzing story...", file=sys.stderr)
    response = None
    max_retries = 3
    json_obj = {}
    for attempt in range(max_retries):
        try:
            response = None
            response = llm.invoke(prompt)
            print("Response", response)

            # Extract content from AIMessage
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            response_text = cleanup_response(response_text)

            # Validate the raw response with second LLM before any further processing
            json_obj = parser.parse(response_text)
            if not isinstance(json_obj, dict):
                raise ValueError("Response is not a json dict")
            json_obj_dump = json.dumps(json_obj)

            if not validate_output(prompt, content, json_obj_dump):
                raise RuntimeError("LLM response failed validation")

            break

        except Exception as e:
            json_obj = {}
            print(
                f"Error in chain. raw response: {response} | Exception: {e}",
                file=sys.stderr,
            )
    if response is None:
        raise RuntimeError("LLM failed to generate a response after retries")

    # Use OutputParser for strict JSON parsing
    try:
        return json_obj
    except Exception as e:
        print(f"OutputParser failed, using fallback", file=sys.stderr)
        # ...existing code for fallback parsing...


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <file_path>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        json_obj = extract_story_metadata(file_path)
        # Output as clean JSON to stdout
        print(json.dumps(json_obj, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
