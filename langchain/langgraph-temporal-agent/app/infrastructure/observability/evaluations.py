"""
Evaluation and scoring helpers for Langfuse.

This module provides functions to log evaluation scores to Langfuse traces,
covering rules #14 and #16:

  - Add scores later (quality metrics, human approval)
  - Use the approval signal as an observability goldmine

Score types supported:

  research_quality        — Quality of the research findings (1-5)
  source_quality          — Average quality of sources (1-5)
  citation_correctness    — Accuracy of citations (boolean or 1-5)
  proposal_quality        — Quality of the generated proposal (1-5)
  human_approval          — Human decision (true/false)
  action_success          — Whether the executed action succeeded (boolean)

Usage (Phase 5):

    from app.infrastructure.observability.evaluations import (
        log_approval_score,
        log_quality_score,
    )

    # Log human approval after user decision
    await log_approval_score(
        workflow_id="research-abc123",
        score=True,
        metadata={"reason": "meets requirements"},
    )

    # Log quality metrics
    await log_quality_score(
        workflow_id="research-abc123",
        name="research_quality",
        value=4.0,
        metadata={"reviewer": "alice"},
    )
"""

from __future__ import annotations

import os
from typing import Any

from app.infrastructure.observability.client import get_langfuse
from app.infrastructure.observability.config import is_tracing_enabled


async def log_quality_score(
    *,
    workflow_id: str,
    name: str,
    value: float,
    metadata: dict[str, Any] | None = None,
    reason: str | None = None,
) -> bool:
    """
    Log a quality metric score to a Langfuse trace.

    Parameters
    ----------
    workflow_id : str
        The Temporal workflow ID (used as trace session_id).
    name : str
        The score name (e.g., "research_quality", "proposal_quality").
    value : float
        The numeric score value.
    metadata : dict, optional
        Additional metadata for this score.
    reason : str, optional
        Human-readable reason for the score.

    Returns
    -------
    bool
        True if the score was logged, False if skipped (tracing disabled).
    """
    if not is_tracing_enabled():
        return False

    try:
        langfuse = get_langfuse()
        score_metadata: dict[str, Any] = {"workflow_id": workflow_id}
        if metadata:
            score_metadata.update(metadata)
        if reason:
            score_metadata["reason"] = reason

        # Langfuse's score API attaches to a trace by session_id or trace_id
        # Using direct SDK call to attach score to existing trace
        langfuse.score(
            name=name,
            value=value,
            session_id=workflow_id,
            metadata=score_metadata,
        )
        return True
    except Exception:
        # Never let observability failures affect business logic
        return False


async def log_approval_score(
    *,
    workflow_id: str,
    approved: bool,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Log a human approval score to a Langfuse trace.

    This is particularly valuable because it correlates LLM execution
    with eventual human outcome:

        model version → approval rate
        prompt version → approval rate
        graph version → approval rate

    Parameters
    ----------
    workflow_id : str
        The Temporal workflow ID.
    approved : bool
        Whether the proposal was approved.
    metadata : dict, optional
        Additional metadata (reason, reviewer, etc.).

    Returns
    -------
    bool
        True if logged, False if skipped.
    """
    return await log_quality_score(
        workflow_id=workflow_id,
        name="human_approval",
        value=1.0 if approved else 0.0,
        metadata={**(metadata or {}), "approved": approved},
    )


async def log_research_quality(
    workflow_id: str,
    quality: float,
    **kwargs: Any,
) -> bool:
    """Log research quality score (typically 1-5 scale)."""
    return await log_quality_score(
        workflow_id=workflow_id,
        name="research_quality",
        value=quality,
        **kwargs,
    )


async def log_source_quality(
    workflow_id: str,
    quality: float,
    **kwargs: Any,
) -> bool:
    """Log source quality score (typically 1-5 scale)."""
    return await log_quality_score(
        workflow_id=workflow_id,
        name="source_quality",
        value=quality,
        **kwargs,
    )


async def log_citation_correctness(
    workflow_id: str,
    correct: bool | float,
    **kwargs: Any,
) -> bool:
    """Log citation correctness (boolean or 1-5 scale)."""
    value = 1.0 if correct else 0.0 if isinstance(correct, bool) else correct
    return await log_quality_score(
        workflow_id=workflow_id,
        name="citation_correctness",
        value=value,
        **kwargs,
    )


async def log_proposal_quality(
    workflow_id: str,
    quality: float,
    **kwargs: Any,
) -> bool:
    """Log proposal quality score (typically 1-5 scale)."""
    return await log_quality_score(
        workflow_id=workflow_id,
        name="proposal_quality",
        value=quality,
        **kwargs,
    )


async def log_action_success(
    workflow_id: str,
    success: bool,
    **kwargs: Any,
) -> bool:
    """Log whether the executed action succeeded."""
    return await log_quality_score(
        workflow_id=workflow_id,
        name="action_success",
        value=1.0 if success else 0.0,
        **kwargs,
    )
