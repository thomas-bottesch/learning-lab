"""
LangGraph graphs for the research agent.

Three separate graphs are defined:
  1. Research Graph    — search + summarise
  2. Verification Graph — verify sources
  3. Proposal Graph     — generate a proposal

Each graph uses **mocked** async search/LLM behaviour so the project runs
without external API keys.  The structure makes it trivial to swap in real
APIs later (e.g. Tavily, SerpAPI, OpenAI).

The graphs are invoked via ``ainvoke()`` from Temporal Activities (which are
themselves async).  For testing we also use ``ainvoke()`` / ``asyncio.run()``.
"""

from __future__ import annotations

import typing
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Typed state for each graph (LangGraph expects TypedDict or dataclass)
# ---------------------------------------------------------------------------


class ResearchState(typing.TypedDict, total=False):
    """State used by the research graph."""

    question: str
    findings: list[str]
    sources: list[str]


class VerificationState(typing.TypedDict, total=False):
    """State used by the verification graph."""

    question: str
    findings: list[str]
    sources: list[str]
    verified_sources: list[str]
    rejected_sources: list[str]


class ProposalState(typing.TypedDict, total=False):
    """State used by the proposal graph."""

    question: str
    verified_sources: list[str]
    rejected_sources: list[str]
    title: str
    summary: str
    proposed_action: str


# ---------------------------------------------------------------------------
# Mock external calls (async — replace with real APIs later)
# ---------------------------------------------------------------------------


async def mock_search(question: str) -> list[dict[str, str]]:
    """
    Mock search function.

    TODO: Replace with a real async search API (Tavily, SerpAPI, etc.).
    """
    return [
        {
            "title": "CockroachDB vs PostgreSQL: A Comparison",
            "url": "https://example.com/cockroach-postgres-comparison",
            "snippet": "CockroachDB offers horizontal scalability and strong consistency, while PostgreSQL excels in feature richness and ecosystem maturity.",
        },
        {
            "title": "Migrating from PostgreSQL to Distributed SQL",
            "url": "https://example.com/migrating-postgres-distributed",
            "snippet": "Organisations migrating to CockroachDB often do so for global distribution and automatic sharding. However, migration complexity and tooling gaps should be considered.",
        },
        {
            "title": "PostgreSQL 17 Released with Performance Improvements",
            "url": "https://example.com/postgres-17",
            "snippet": "PostgreSQL 17 brings significant performance improvements, making it a strong candidate for many workloads previously requiring distributed databases.",
        },
    ]


async def mock_llm_summarize(question: str, sources: list[dict]) -> list[str]:
    """
    Mock LLM summarisation.

    TODO: Replace with a real async LLM call (OpenAI, Anthropic, etc.).
    """
    return [
        f"Based on analysis of {len(sources)} sources regarding '{question}':",
        "CockroachDB provides automatic sharding and multi-region consistency "
        "but sacrifices some of the rich ecosystem and tooling that PostgreSQL offers.",
        "PostgreSQL 17 delivers significant performance improvements and may "
        "suffice for most workloads without the complexity of a distributed database.",
        "Migration from PostgreSQL to CockroachDB requires careful planning, "
        "as driver compatibility, migration tools, and operational expertise "
        "are non-trivial factors.",
    ]


async def mock_llm_verify(sources: list[dict]) -> tuple[list[str], list[str]]:
    """
    Mock source verification.

    Returns (verified, rejected) source titles.

    TODO: Replace with a real async LLM verification call.
    """
    verified: list[str] = []
    rejected: list[str] = []
    for src in sources:
        # Simulate a simple heuristic: reject sources with "migrating" in URL
        if "migrating" in src.get("url", ""):
            rejected.append(src["title"])
        else:
            verified.append(src["title"])
    # If nothing was rejected, reject the least informative one
    if not rejected and sources:
        rejected.append(sources[-1]["title"])
    return verified, rejected


async def mock_llm_generate_proposal(
    question: str,
    verified_sources: list[str],
    rejected_sources: list[str],
) -> dict[str, str]:
    """
    Mock proposal generation.

    TODO: Replace with a real async LLM call.
    """
    return {
        "title": f"Assessment: Migrating from PostgreSQL to CockroachDB",
        "summary": (
            f"After researching '{question}' and verifying {len(verified_sources)} "
            f"sources (rejecting {len(rejected_sources)}), the evidence suggests that "
            "a full migration to CockroachDB is not immediately warranted. "
            "PostgreSQL 17 offers strong performance, and a phased approach "
            " — such as read replicas or eventual migration — is recommended."
        ),
        "proposed_action": (
            "Set up a CockroachDB test cluster in staging, run benchmark "
            "workloads matching production patterns, and evaluate results "
            "before committing to a migration."
        ),
    }


# ---------------------------------------------------------------------------
# Node functions (async — invoke mock I/O)
# ---------------------------------------------------------------------------


async def search(state: ResearchState) -> dict:
    """Perform a (mock) web search and store results."""
    question = state.get("question", "")
    sources = await mock_search(question)
    return {"findings": [], "sources": [s["title"] for s in sources]}


async def summarize(state: ResearchState) -> dict:
    """Summarise findings using a (mock) LLM."""
    question = state.get("question", "")
    sources = await mock_search(question)
    findings = await mock_llm_summarize(question, sources)
    return {"findings": findings, "sources": [s["title"] for s in sources]}


async def verify_sources_node(state: VerificationState) -> dict:
    """Verify the credibility of sources using a (mock) LLM."""
    question = state.get("question", "")
    sources = await mock_search(question)
    verified, rejected = await mock_llm_verify(sources)
    return {
        "verified_sources": verified,
        "rejected_sources": rejected,
    }


async def generate_proposal_node(state: ProposalState) -> dict:
    """Generate a proposal using a (mock) LLM."""
    return await mock_llm_generate_proposal(
        state.get("question", ""),
        state.get("verified_sources", []),
        state.get("rejected_sources", []),
    )


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_research_graph():
    """
    Research Graph:

        START → search → summarize → END

    This graph accepts a research question and produces findings + sources.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("search", search)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def build_verification_graph():
    """
    Verification Graph:

        START → verify_sources → END
    """
    graph = StateGraph(VerificationState)

    graph.add_node("verify_sources", verify_sources_node)

    graph.add_edge(START, "verify_sources")
    graph.add_edge("verify_sources", END)

    return graph.compile()


def build_proposal_graph():
    """
    Proposal Graph:

        START → generate_proposal → END
    """
    graph = StateGraph(ProposalState)

    graph.add_node("generate_proposal", generate_proposal_node)

    graph.add_edge(START, "generate_proposal")
    graph.add_edge("generate_proposal", END)

    return graph.compile()
