"""
Temporal Activities: execute_action, notify_user

These activities handle external side-effects (action execution, notifications).
They do NOT import any graph or infrastructure code.

ARCHITECTURAL NOTE — Purity Invariant:

This module MUST NOT import LangChain, LangGraph, or any external I/O code.
It only depends on domain.models and temporalio.
"""

from __future__ import annotations

from temporalio import activity

from app.domain.models import ExecutionResult, NotificationResult

# ---------------------------------------------------------------------------
# Activity: execute_action
# ---------------------------------------------------------------------------


@activity.defn
async def execute_action(workflow_id: str, proposed_action: str) -> ExecutionResult:
    """
    Execute the approved action (mocked).

    **Idempotency note:**
    External side-effecting Activities can be retried by Temporal.  For
    example:

        Activity executes external action
                ↓
        external action succeeds
                ↓
        worker crashes before Temporal receives completion
                ↓
        Temporal retries Activity
                ↓
        same idempotency key
                ↓
        external system returns existing result

    We derive a stable idempotency key from the Workflow ID so that
    repeated invocations of this Activity for the same workflow produce
    the same result without duplicating the side effect.

    **Temporal does NOT make arbitrary external APIs exactly-once.**
    Idempotency keys are the developer's responsibility.
    """
    idempotency_key = f"{workflow_id}:execute-action"

    activity.logger.info(
        "execute_action started",
        workflow_id=workflow_id,
        idempotency_key=idempotency_key,
        action=proposed_action,
    )

    return ExecutionResult(
        workflow_id=workflow_id,
        action=proposed_action,
        success=True,
        detail=f"Action executed with idempotency key '{idempotency_key}'.",
    )


# ---------------------------------------------------------------------------
# Activity: notify_user
# ---------------------------------------------------------------------------


@activity.defn
async def notify_user(workflow_id: str, message: str) -> NotificationResult:
    """
    Send a notification to the user (mocked).

    This is an Activity because it performs external I/O (email, webhook,
    push notification, etc.).
    """
    activity.logger.info("notify_user started", workflow_id=workflow_id)

    return NotificationResult(
        workflow_id=workflow_id,
        message=message,
        delivered=True,
    )
