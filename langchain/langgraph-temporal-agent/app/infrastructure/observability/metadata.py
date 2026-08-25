"""
Metadata helpers for Temporal correlation.

This module provides functions to build standard metadata dicts from
Temporal context, covering rules #6, #8, and #17:

  - workflow_id
  - run_id
  - activity_id
  - activity_type
  - attempt
  - execution_id (composite key)
  - task_queue
  - namespace

Usage:

    from app.infrastructure.observability.metadata import (
        get_temporal_metadata,
        build_execution_id,
    )

    # Inside an Activity:
    from temporalio import activity

    @activity.defn
    async def my_activity(request: MyRequest) -> MyResult:
        meta = get_temporal_metadata()
        # meta = {
        #     "temporal.workflow_id": "...",
        #     "temporal.run_id": "...",
        #     "temporal.activity_id": "...",
        #     "temporal.attempt": 1,
        #     ...
        # }
"""

from __future__ import annotations

import os
from typing import Any

from temporalio import activity


def build_execution_id(
    workflow_id: str,
    activity_id: str,
    attempt: int,
) -> str:
    """
    Build a deterministic execution ID for correlation.

    Covers rule #17: execution_id = f"{workflow_id}:{activity_id}:{attempt}"

    This gives you a unique identifier per physical attempt while still
    being able to group retries under the same logical execution.
    """
    return f"{workflow_id}:{activity_id}:{attempt}"


def get_temporal_metadata() -> dict[str, Any]:
    """
    Extract all available Temporal metadata into a flat dict.

    Returns a dict with keys prefixed by "temporal." for consistency:

        {
            "temporal.workflow_id": ...,
            "temporal.run_id": ...,
            "temporal.workflow_type": ...,
            "temporal.activity_id": ...,
            "temporal.activity_type": ...,
            "temporal.task_queue": ...,
            "temporal.namespace": ...,
            "temporal.attempt": ...,
            "temporal.retry.status": ...,  # if retried
            "temporal.retry.last_attempt": ...,
            "temporal.retry.total_attempts": ...,
        }
    """
    info = activity.info()

    result: dict[str, Any] = {
        "temporal.workflow_id": info.workflow_id,
        "temporal.run_id": info.run_id,
        "temporal.workflow_type": info.workflow_type,
        "temporal.activity_id": info.activity_id,
        "temporal.activity_type": info.activity_type,
        "temporal.task_queue": info.task_queue,
        "temporal.namespace": info.namespace,
        "temporal.attempt": info.attempt,
    }

    # Include retry information if available (rule #8)
    retry = info.retry
    if retry:
        result["temporal.retry.status"] = (
            retry.status.name if hasattr(retry.status, "name") else str(retry.status)
        )
        result["temporal.retry.last_attempt"] = retry.last_attempt
        result["temporal.retry.total_attempts"] = retry.total_attempts

    return result


def build_workflow_context(workflow_id: str | None = None) -> dict[str, Any]:
    """
    Build a minimal workflow-level context dict for session tracking.

    Covers rule #1: one trace = one logical workflow.

    The session_id should equal the workflow_id so Langfuse lets you
    search for exact workflow traces.
    """
    meta = get_temporal_metadata()

    # Override workflow_id if explicitly provided
    if workflow_id:
        meta["temporal.workflow_id"] = workflow_id
        meta["session_id"] = workflow_id

    return meta


def get_or_build_workflow_id() -> str:
    """
    Get the current workflow_id from Temporal context.

    Used as the session_id for Langfuse traces.
    """
    return activity.info().workflow_id
