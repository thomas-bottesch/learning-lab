"""
Context managers for Langfuse workflow sessions.

This module provides ``workflow_context`` — a context manager that sets up
Langfuse session-level propagation for one logical workflow.

Covers rules #1 and #2:
  - One trace = one logical workflow
  - Centralized infrastructure module (not scattered imports)

Usage:

    from app.infrastructure.observability import workflow_context
    from app.infrastructure.observability.config import get_application_metadata

    # Inside an Activity or any process that knows workflow_id/run_id:
    with workflow_context(
        workflow_id="research-abc123",
        run_id="run-xyz789",
    ):
        # All Langfuse calls inside this block share the session.
        pass
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from app.infrastructure.observability.client import get_langfuse
from app.infrastructure.observability.config import (
    ENVIRONMENT,
    LANGFUSE_TRACING_ENVIRONMENT,
    get_application_metadata,
    is_tracing_enabled,
)


@contextmanager
def workflow_context(
    *,
    workflow_id: str,
    run_id: str | None = None,
    environment: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """
    Context manager that propagates workflow-level attributes to Langfuse.

    This sets the Langfuse session_id to the workflow_id so that:
      - You can search for exact workflows in the UI
      - All traces within a workflow are grouped under one session
      - Metadata contains Temporal correlation info

    Parameters
    ----------
    workflow_id : str
        The Temporal workflow ID. Used as session_id.
    run_id : str, optional
        The Temporal run ID. Added to metadata.
    environment : str, optional
        Override environment name. Defaults to APP_ENV.
    extra_metadata : dict, optional
        Additional metadata to propagate.
    """
    env = environment or ENVIRONMENT
    langfuse = get_langfuse()

    # Build base metadata from application config
    app_meta = get_application_metadata()

    # Add Temporal-specific keys
    temporal_meta: dict[str, Any] = {
        "temporal.workflow_id": workflow_id,
        "temporal.namespace": os.environ.get("TEMPORAL_NAMESPACE", "default"),
        "temporal.task_queue": os.environ.get("TEMPORAL_TASK_QUEUE", "research-agent"),
        "environment": env,
    }

    if run_id:
        temporal_meta["temporal.run_id"] = run_id

    # Merge metadata
    metadata: dict[str, Any] = {**app_meta, **temporal_meta}
    if extra_metadata:
        metadata.update(extra_metadata)

    # Enter Langfuse trace/session scope
    # When IS_TRACING_ENABLED is False, this uses no-op client that silently passes through
    with _maybe_trace(langfuse, workflow_id, metadata) as trace:
        yield trace


@contextmanager
def _maybe_trace(
    langfuse: Any,
    trace_id: str,
    metadata: dict[str, Any],
) -> Iterator[Any]:
    """
    Create a Langfuse trace if tracing is enabled, otherwise noop.

    This ensures rule #23: tracing never blocks business logic.
    """
    if is_tracing_enabled():
        try:
            t = langfuse.trace(
                name="research-workflow",
                id=trace_id,
                session_id=trace_id,
                metadata=metadata,
                tags=["temporal", "langgraph"],
            )
            yield t
        except Exception:
            # If trace creation fails, continue without tracing (rule #23)
            yield None
    else:
        yield None
