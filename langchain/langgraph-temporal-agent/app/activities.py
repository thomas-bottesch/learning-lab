"""
Temporal Activities for the research agent.

Each Activity is a thin, serialisable wrapper around a LangGraph invocation
(or, for the non-agent activities, a simple external-side-effect placeholder).

All external I/O — LLM calls, HTTP requests, database writes — happens
**inside Activities**, never in Workflow code.  This preserves Temporal's
determinism guarantees.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta

from temporalio import activity
from temporalio.common import RetryPolicy

from app.graphs import (
    build_research_graph,
    build_verification_graph,
    build_proposal_graph,
)
from app.models import (
    ExecutionResult,
    NotificationResult,
    Proposal,
    ResearchRequest,
    ResearchResult,
    VerifiedResearch,
)

# ---------------------------------------------------------------------------
# Retry policies per activity (tunable per nature of the work)
# ---------------------------------------------------------------------------

_RESEARCH_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=5,
)

_VERIFY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=5,
)

_PROPOSAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=3,
)

# execute_action and notify_user use conservative defaults — no custom
# RetryPolicy means Temporal uses its own (1 s initial, 2× backoff,
# 100 max attempts, no max interval).  We keep them light-weight below.


# ---------------------------------------------------------------------------
# Activity: research
# ---------------------------------------------------------------------------


@activity.defn
async def research(request: ResearchRequest) -> ResearchResult:
    """
    Run the Research LangGraph to investigate the given question.

    This Activity:
      1. Builds (or retrieves) the compiled LangGraph.
      2. Invokes the graph with the research question.
      3. Returns a serialisable ResearchResult.

    The graph internally calls mock_search / mock_llm_summarize.
    In production these would be real search-LLM pipelines.
    """
    activity.logger.info("research activity started", question=request.question)

    graph = build_research_graph()

    # Invoke the graph — LangGraph handles its own internal state.
    result = await graph.ainvoke({"question": request.question})

    activity.logger.info(
        "research activity completed",
        findings_count=len(result.get("findings", [])),
        sources_count=len(result.get("sources", [])),
    )

    return ResearchResult(
        question=request.question,
        findings=result.get("findings", []),
        sources=result.get("sources", []),
    )


# ---------------------------------------------------------------------------
# Activity: verify_sources
# ---------------------------------------------------------------------------


@activity.defn
async def verify_sources(
    research_result: ResearchResult,
) -> VerifiedResearch:
    """
    Verify the credibility of sources returned by the research phase.

    Invokes the Verification LangGraph.
    """
    activity.logger.info(
        "verify_sources activity started",
        sources_count=len(research_result.sources),
    )

    graph = build_verification_graph()

    result = await graph.ainvoke(
        {
            "question": research_result.question,
            "findings": research_result.findings,
            "sources": research_result.sources,
        }
    )

    return VerifiedResearch(
        question=research_result.question,
        verified_sources=result.get("verified_sources", []),
        rejected_sources=result.get("rejected_sources", []),
    )


# ---------------------------------------------------------------------------
# Activity: generate_proposal
# ---------------------------------------------------------------------------


@activity.defn
async def generate_proposal(
    verified: VerifiedResearch,
) -> Proposal:
    """
    Generate a research proposal from verified sources.

    Invokes the Proposal LangGraph.
    """
    activity.logger.info(
        "generate_proposal activity started",
        verified_count=len(verified.verified_sources),
    )

    graph = build_proposal_graph()

    result = await graph.ainvoke(
        {
            "question": verified.question,
            "verified_sources": verified.verified_sources,
            "rejected_sources": verified.rejected_sources,
        }
    )

    return Proposal(
        question=verified.question,
        title=result.get("title", ""),
        summary=result.get("summary", ""),
        proposed_action=result.get("proposed_action", ""),
    )


# ---------------------------------------------------------------------------
# Activity: execute_action
# ---------------------------------------------------------------------------


@activity.defn
async def execute_action(
    workflow_id: str,
    proposed_action: str,
) -> ExecutionResult:
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
    # Derive idempotency key from workflow identity
    idempotency_key = f"{workflow_id}:execute-action"

    activity.logger.info(
        "execute_action started",
        workflow_id=workflow_id,
        idempotency_key=idempotency_key,
        action=proposed_action,
    )

    # In production, the real action would check the idempotency key
    # against the external system's deduplication store before executing.

    # For the demo, we simply return a mock result.
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
async def notify_user(
    workflow_id: str,
    message: str,
) -> NotificationResult:
    """
    Send a notification to the user (mocked).

    This is an Activity because it performs external I/O (email, webhook,
    push notification, etc.).
    """
    activity.logger.info("notify_user started", workflow_id=workflow_id)

    # In production: send email, webhook, Slack message, etc.
    return NotificationResult(
        workflow_id=workflow_id,
        message=message,
        delivered=True,
    )
