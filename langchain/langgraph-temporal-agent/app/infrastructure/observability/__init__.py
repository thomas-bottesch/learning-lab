"""
Observability infrastructure for the LangGraph + Temporal agent.

This module centralizes ALL Langfuse integration so that Activities, Workflows,
and Graphs know nothing about Langfuse internals.

Architecture:

    Activity
       ↓
    observability API  ← Application depends on abstraction
       ↓
    Langfuse SDK       ← Concrete implementation

This gives you:
  - Clean separation of concerns (observability = infrastructure)
  - Easy testing (mock the observability API)
  - No Langfuse coupling in domain or orchestration code

USAGE GUIDE BY PHASE:

Phase 1 — Foundation:
    from app.infrastructure.observability import get_langfuse, workflow_context

Phase 2 — Activity tracing:
    from app.infrastructure.observability import observe_activity

Phase 3 — LangGraph tracing:
    from app.infrastructure.observability.callbacks import langfuse_callbacks

Phase 4 — LLM quality:
    from app.infrastructure.observability.llm_tracing import (
        traced_llm_call,
        TokenUsage,
        estimate_cost,
    )

Phase 5 — Evaluation:
    from app.infrastructure.observability.evaluations import (
        log_approval_score,
        log_quality_score,
    )

Phase 6 — Production hardening:
    from app.infrastructure.observability.redaction import sanitize
    from app.infrastructure.observability.config import should_sample

IMPORTANT: This module is designed to be OPTIONAL. If Langfuse is unavailable,
the worker should continue processing business work without failure.
"""

from app.infrastructure.observability.client import get_langfuse
from app.infrastructure.observability.context import workflow_context
from app.infrastructure.observability.tracing import (
    activity_observer,
    observe_activity,
    workflow_trace,
)

__all__ = [
    "activity_observer",
    "get_langfuse",
    "observe_activity",
    "workflow_context",
    "workflow_trace",
]
