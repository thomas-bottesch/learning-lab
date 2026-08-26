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


def _extract_retry_info(info: activity.Info) -> dict[str, Any]:
    """Extract retry policy metadata from ``info.retry_policy``."""
    retry_policy = info.retry_policy
    if retry_policy is None:
        return {}

    result: dict[str, Any] = {}

    for attr, key in [
        ("maximum_attempts", "temporal.retry.maximum_attempts"),
        ("maximum_per_second", "temporal.retry.maximum_per_second"),
        ("backoff_coefficient", "temporal.retry.backoff_coefficient"),
        ("maximum_interval", "temporal.retry.maximum_interval"),
    ]:
        val = getattr(retry_policy, attr, None)
        if val is not None:
            result[key] = val

    initial_interval = getattr(retry_policy, "initial_interval", None)
    if initial_interval is not None:
        result["temporal.retry.initial_interval"] = str(initial_interval)

    return result


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
            "temporal.retry.*": ...,       # if retry policy is configured
        }
    """
    info = activity.info()

    # ``Info.run_id`` was renamed to ``Info.workflow_run_id`` in SDK v1.31+.
    run_id = getattr(info, "workflow_run_id", None)

    return {
        "temporal.workflow_id": info.workflow_id,
        "temporal.run_id": run_id,
        "temporal.workflow_type": info.workflow_type,
        "temporal.activity_id": info.activity_id,
        "temporal.activity_type": info.activity_type,
        "temporal.task_queue": info.task_queue,
        "temporal.namespace": info.namespace,
        "temporal.attempt": info.attempt,
        **_extract_retry_info(info),
    }


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
    wid = activity.info().workflow_id
    if wid is None:
        raise RuntimeError(
            "Cannot determine workflow_id — not running inside a workflow"
        )
    return wid
