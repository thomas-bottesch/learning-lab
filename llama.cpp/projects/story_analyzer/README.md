# Story Analyzer

An intelligent story analysis tool that extracts structured metadata from narrative text using LangChain and local LLM inference via llama.cpp.

## Overview

The Story Analyzer uses a two-stage LLM approach to extract and validate story metadata:
1. **Extraction**: Analyzes story content to identify characters and genre(s)
2. **Validation**: A second LLM validates the output for completeness and accuracy

## Features

- **Character Extraction**: Identifies all character names mentioned in the story
- **Genre Classification**: Determines appropriate genre(s) (can be multiple)
- **Few-Shot Learning**: Uses embedded examples to guide the LLM
- **Self-Validation**: Built-in LLM validator ensures output quality
- **Robust Error Handling**: Automatic retries with cleanup on failure
- **JSON Output**: Clean, structured output ready for downstream processing

## Requirements

Ensure the llama-server is running before using this tool. See the [main README](../../README.md) for server setup instructions.

```bash
# From the repository root
bash start_server.sh
```

## Usage

### Basic Usage

```bash
python analyzer.py <story_file>
```

### Example

```bash
python analyzer.py chatgpt_generated_story_1
```

**Output:**
```json
{
  "characters": [
    "Orion",
    "copper-brown skinned girl"
  ],
  "genre": [
    "science-fiction",
    "fantasy"
  ]
}
```

### Processing Multiple Stories

```bash
# Process all sample stories
for story in chatgpt_generated_story_*; do
    echo "Analyzing $story..."
    python analyzer.py "$story"
    echo "---"
done
```

## How It Works

### 1. Story Loading

The analyzer reads the story file content:

```python
with open(story_file_path, "r") as f:
    content = f.read()
```

### 2. Extraction Phase

A LangChain LLM chain analyzes the story using few-shot prompting:

```python
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=150,
)
```

The prompt includes:
- Two complete examples showing expected output format
- Clear task description
- The story content wrapped in delimiter tags
- JSON schema specification

### 3. Response Cleanup

Removes markdown code fences and validates JSON structure:

```python
def cleanup_response(response: str) -> str:
    # Removes ```json and ``` markers
    # Returns clean JSON string
```

### 4. Validation Phase

A second LLM instance validates the extraction:

```python
llm_validator = ChatOpenAI(
    temperature=0.0,  # Deterministic validation
    max_tokens=100,
)
```

The validator checks:
- ✓ All required fields present (`characters`, `genre`)
- ✓ Genre makes sense for the story
- ✓ All characters mentioned in the story are included
- ✓ Proper list format for both fields

### 5. Retry Logic

Up to 3 attempts are made if extraction or validation fails:

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        # Extract and validate
        break
    except Exception as e:
        # Log and retry
```

## Expected Story Format

Stories should be plain text files with clear narrative structure:

```
<Your Story Title>

<Story content with clear character names and plot...>
```

The analyzer works best when:
- Character names are clearly stated (not just pronouns)
- The story has a recognizable genre
- Content is coherent narrative text

## Output Schema

```json
{
  "characters": ["string", "..."],
  "genre": ["string", "..."]
}
```

**Fields:**
- `characters` (array): List of character names found in the story
- `genre` (array): One or more genre classifications

**Common Genres:**
- science-fiction
- fantasy
- drama
- mystery
- thriller
- romance
- horror
- adventure