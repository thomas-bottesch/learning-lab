"""
Shared serializable data models for the LangGraph + Temporal agent.

All data passed through Temporal Activities must be serializable.
These Pydantic models ensure type safety and explicit contracts between
Activities.

IMPORTANT: Do NOT pass large documents through Temporal workflow state.
Temporal history has size limits. For production systems, store large
artifacts in external object storage (S3, GCS, etc.) and reference them
by ID:

    Temporal Workflow
            │
            └── document_id (e.g., "s3://bucket/research/abc123")
                    │
                    ▼
            Object Storage (S3 / GCS / Azure Blob)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Source model — rich source representation
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A single source retrieved during research."""

    id: str = Field(description="Unique identifier for this source.")
    url: str = Field(description="URL of the source.")
    title: str = Field(description="Title of the source.")
    snippet: str = Field(
        default="",
        description="Excerpt or summary of the source content.",
    )
    publisher: str = Field(
        default="",
        description="Publisher or author of the source.",
    )
    retrieved_at: datetime | None = Field(
        default=None,
        description="Timestamp when the source was retrieved.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific metadata.",
    )


# ---------------------------------------------------------------------------
# Research Request / Result
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    """Input to the research workflow."""

    question: str = Field(
        description="The research question to investigate.",
        examples=["Should our company migrate from PostgreSQL to CockroachDB?"],
    )
    # Optional: reference to pre-existing documents stored externally
    external_doc_ids: list[str] = Field(
        default_factory=list,
        description="IDs referencing large documents stored in external object storage.",
    )


class ResearchResult(BaseModel):
    """Output of the research phase (from the Research LangGraph)."""

    question: str
    findings: list[str] = Field(
        description="Summarised findings from the research.",
    )
    sources: list[Source] = Field(
        description="List of source objects with metadata.",
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class VerifiedResearch(BaseModel):
    """Output of the verification phase."""

    question: str
    verified_sources: list[Source] = Field(
        description="Sources that passed verification.",
    )
    rejected_sources: list[Source] = Field(
        description="Sources that were rejected during verification.",
    )


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


class Proposal(BaseModel):
    """Output of the proposal generation phase."""

    question: str
    title: str = Field(description="A concise title for the proposal.")
    summary: str = Field(
        description="A summary of the verified research and rationale.",
    )
    proposed_action: str = Field(
        description="The recommended action to take.",
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    """Result of executing the approved action."""

    workflow_id: str
    action: str
    success: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------


class NotificationResult(BaseModel):
    """Result of sending a notification."""

    workflow_id: str
    message: str
    delivered: bool


# ---------------------------------------------------------------------------
# Internal graph state (not passed through Temporal, used inside LangGraph)
# ---------------------------------------------------------------------------


@dataclass
class ResearchGraphInput:
    """Input passed to the research LangGraph."""

    question: str


@dataclass
class ResearchGraphOutput:
    """Output from the research LangGraph."""

    findings: list[str]
    sources: list[Source]


@dataclass
class VerificationGraphInput:
    """Input passed to the verification LangGraph."""

    question: str
    findings: list[str]
    sources: list[Source]


@dataclass
class VerificationGraphOutput:
    """Output from the verification LangGraph."""

    verified_sources: list[Source]
    rejected_sources: list[Source]


@dataclass
class ProposalGraphInput:
    """Input passed to the proposal LangGraph."""

    question: str
    verified_sources: list[Source]
    rejected_sources: list[Source]


@dataclass
class ProposalGraphOutput:
    """Output from the proposal LangGraph."""

    title: str
    summary: str
    proposed_action: str
