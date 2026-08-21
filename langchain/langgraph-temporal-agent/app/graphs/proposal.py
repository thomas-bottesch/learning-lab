"""
LangGraph graph for the proposal generation phase.

This graph generates a research proposal from verified sources.
"""

import typing

from langgraph.graph import END, START, StateGraph

from app.domain.models import Source
from app.infrastructure.llm import llm_generate_proposal as mock_llm_generate_proposal

# ---------------------------------------------------------------------------
# Typed state for the proposal graph
# ---------------------------------------------------------------------------


class ProposalState(typing.TypedDict, total=False):
    """State used by the proposal graph."""

    question: str
    verified_sources: list[Source]
    rejected_sources: list[Source]
    title: str
    summary: str
    proposed_action: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def generate_proposal_node(state: ProposalState) -> dict:
    """Generate a proposal using a (mock) LLM."""
    verified_sources = state.get("verified_sources", [])
    rejected_sources = state.get("rejected_sources", [])
    verified_titles = [s.title for s in verified_sources]
    rejected_titles = [s.title for s in rejected_sources]
    return await mock_llm_generate_proposal(
        state.get("question", ""),
        verified_titles,
        rejected_titles,
    )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


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
