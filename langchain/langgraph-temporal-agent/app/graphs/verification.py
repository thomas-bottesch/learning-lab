"""
LangGraph graph for the verification phase.

This graph verifies the credibility of sources returned by the research phase.
"""

import typing

from langgraph.graph import END, START, StateGraph

from app.infrastructure.llm import llm_verify as mock_llm_verify, search as mock_search

# ---------------------------------------------------------------------------
# Typed state for the verification graph
# ---------------------------------------------------------------------------


class VerificationState(typing.TypedDict, total=False):
    """State used by the verification graph."""

    question: str
    findings: list[str]
    sources: list[str]
    verified_sources: list[str]
    rejected_sources: list[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def verify_sources_node(state: VerificationState) -> dict:
    """Verify the credibility of sources using a (mock) LLM."""
    question = state.get("question", "")
    sources = await mock_search(question)
    verified, rejected = await mock_llm_verify(sources)
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
