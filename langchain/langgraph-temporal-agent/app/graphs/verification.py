"""
LangGraph graph for the verification phase.

This graph verifies the credibility of sources returned by the research phase.
"""

import typing

from langgraph.graph import END, START, StateGraph

from app.domain.models import Source
from app.infrastructure.llm import llm_verify as mock_llm_verify

# ---------------------------------------------------------------------------
# Typed state for the verification graph
# ---------------------------------------------------------------------------


class VerificationState(typing.TypedDict, total=False):
    """State used by the verification graph."""

    question: str
    findings: list[str]
    sources: list[Source]
    verified_sources: list[Source]
    rejected_sources: list[Source]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def verify_sources_node(state: VerificationState) -> dict:
    """Verify the credibility of sources using a (mock) LLM."""
    raw_sources = state.get("sources", [])
    # Accept both string URLs and Source objects
    if raw_sources and isinstance(raw_sources[0], str):
        # String input (e.g. from tests): convert to dict format
        source_dicts = [{"url": s, "title": s, "snippet": ""} for s in raw_sources]
        verified_titles, rejected_titles = await mock_llm_verify(source_dicts)
        verified = []
        rejected = []
        for s in raw_sources:
            if s in verified_titles:
                verified.append(s)
            else:
                rejected.append(s)
        return {
            "verified_sources": verified,
            "rejected_sources": rejected,
        }
    else:
        # Source objects: verify using title matching
        sources = raw_sources  # type: list[Source]
        source_dicts = [
            {"url": s.url, "title": s.title, "snippet": s.snippet} for s in sources
        ]
        verified_titles, rejected_titles = await mock_llm_verify(source_dicts)
        verified = [
            Source(
                id=s.id,
                url=s.url,
                title=s.title,
                snippet=s.snippet,
                publisher=s.publisher,
                retrieved_at=s.retrieved_at,
                metadata=s.metadata,
            )
            for s in sources
            if s.title in verified_titles
        ]
        rejected = [
            Source(
                id=s.id,
                url=s.url,
                title=s.title,
                snippet=s.snippet,
                publisher=s.publisher,
                retrieved_at=s.retrieved_at,
                metadata=s.metadata,
            )
            for s in sources
            if s.title in rejected_titles
        ]
        return {
            "verified_sources": verified,
            "rejected_sources": rejected,
        }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


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
