"""
LangGraph integration helpers for Langfuse.

This module provides functions to create LangChain callback handlers that
automatically capture LangGraph execution hierarchy in Langfuse:

    research-activity
       └── research-graph
            ├── search [tool]
            └── summarize [chain]
                 └── generation [LLM]

Covers rules #5 and #13:
  - Let LangGraph create the children via callbacks
  - Instrument actual LLM calls, not just graphs

Usage:

    from app.infrastructure.observability.callbacks import langfuse_callbacks

    # Inside an Activity:
    result = await graph.ainvoke(
        {"question": request.question},
        config={
            "callbacks": langfuse_callbacks(),
            "run_name": "research-graph",
        },
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import list


def langfuse_callbacks() -> list:
    """
    Return a list of LangChain callback handlers for Langfuse tracing.

    This creates handlers on demand. Each call creates fresh handlers so
    that concurrent graph executions don't share state incorrectly.

    IMPORTANT: If Langfuse is not configured, this returns an empty list
    so the graph runs without tracing (non-authoritative observability).
    """
    from app.infrastructure.observability.config import is_tracing_enabled

    if not is_tracing_enabled():
        return []

    try:
        # Import lazily — graphs must not import Langfuse at module load time
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except ImportError:
        # langfuse not installed or import failed — return empty list
        return []
