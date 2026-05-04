# MCP Server Conventions

This document defines how MCP servers are structured in this repository. Follow
it when adding a new server so that every server is consistent, independently
usable, and well-described to any AI agent that loads it.

---

## Directory layout

```
mcp_servers/
├── SPEC.md               ← this file
├── __init__.py
└── <service>/            ← one directory per server (e.g. forgejo/, mlflow/)
    ├── __init__.py
    ├── client.py         ← domain logic + CLI  (no MCP dependency)
    └── __main__.py       ← MCP server          (thin adapter over client.py)
```

Each server lives in its own subdirectory. No flat files in `mcp_servers/`.

---

## client.py

**One responsibility: talk to the external service.**

Rules:
- No dependency on the `mcp` package. Must be usable without the MCP server running.
- Exposes all domain functions as plain importable Python functions.
- Is also a runnable CLI via `python -m mcp_servers.<service>.client`.
- Has a `main()` function with argument parsing and a `if __name__ == "__main__": main()` guard.
- The module docstring documents all CLI commands with examples.
- Internal CLI helpers are prefixed with `_` (e.g. `_cmd_get_logs`) to distinguish
  them from the public library API.

**Running without the MCP server:**
```
python -m mcp_servers.forgejo.client ml-platform/ml-components list workflows
python -m mcp_servers.forgejo.client ml-platform/ml-components get logs 2
```

---

## __main__.py

**One responsibility: expose client.py functions as MCP tools.**

Rules:

### 1. Define Pydantic models for every structured return type

Never return bare `dict`. Claude (and any other MCP client) reads the output
schema to understand what fields to expect. A bare `dict` produces no schema.

```python
class RunInfo(BaseModel):
    run_index: int   = Field(description="Sequential run number shown in the UI")
    status:    str   = Field(description="One of: success, failure, cancelled, running")
    title:     str   = Field(description="Commit message that triggered the run")
    url:       str   = Field(description="Full URL to the run in the web UI")
```

### 2. Annotate every parameter with a description

Use `Annotated` + `Field(description=...)` so the input schema is self-documenting.

```python
def get_logs(
    owner_repo: Annotated[str, Field(description="Repository in 'owner/repo' format, e.g. 'team-tron/pipelines'")],
    run_index:  Annotated[int, Field(description="Run number from get_runs (the run_index field)")],
    job_index:  Annotated[int, Field(description="Zero-based job index, almost always 0")] = 0,
) -> str:
```

### 3. Write docstrings that explain *when* to use each tool

Don't just describe what the function does — tell the agent how it fits with
the other tools, when to prefer it over an alternative, and what to do next.

```python
def get_jobs(...) -> JobSummary:
    """
    Return the job and step summary for a specific run without fetching full logs.

    Use this to quickly check which step failed before deciding whether to call
    get_logs. Steps are listed in execution order with their individual statuses.
    """
```

### 4. Keep __main__.py thin

No business logic. No HTTP calls. Import everything from `client.py` and wrap
it in a tool. If you find yourself doing non-trivial work in `__main__.py`,
move it into `client.py` first.

### 5. Entry point

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**Running the MCP server:**
```
python -m mcp_servers.forgejo
```

---

## Configuration

### .mcp.json (repo root)

Declares the servers for any MCP-compatible client. Not Claude-specific.

```json
{
  "mcpServers": {
    "<server-name>": {
      "command": "python3",
      "args": ["-m", "mcp_servers.<service>"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

### .claude/settings.json

Approves servers from `.mcp.json` for Claude Code. Keep this file minimal —
only Claude-specific settings belong here.

```json
{
  "enabledMcpjsonServers": ["forgejo-actions"]
}
```

---

## Reference implementation

`mcp_servers/forgejo/` is the canonical example. Read it before writing a new
server. Key things to look at:

- `client.py` — module docstring with CLI usage, public API, `_`-prefixed CLI
  helpers, `main()`, `if __name__ == "__main__"` guard
- `__main__.py` — Pydantic models at the top, `Annotated`+`Field` on every
  parameter, docstrings that guide tool selection, no logic beyond what's
  needed to call `client.py` and return a typed model

---

## Checklist for a new server

- [ ] `mcp_servers/<service>/__init__.py` exists (empty)
- [ ] `client.py` has no `mcp` import
- [ ] `client.py` module docstring lists all CLI commands with examples
- [ ] `client.py` is runnable: `python -m mcp_servers.<service>.client --help` (or shows usage)
- [ ] `__main__.py` uses Pydantic models for all structured return types
- [ ] Every tool parameter has a `Field(description=...)` via `Annotated`
- [ ] Every tool docstring explains *when* to use it, not just *what* it does
- [ ] Server added to `.mcp.json`
- [ ] Server name added to `enabledMcpjsonServers` in `.claude/settings.json`
