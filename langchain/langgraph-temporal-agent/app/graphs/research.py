"""
LangGraph graph for the research phase.

This graph handles search and summarisation of a research question.
It imports from `infrastructure.llm` for external I/O, which is fine
because graphs run inside Activities (not in the Workflow sandbox).
"""

import typing

from langgraph.graph import END, START, StateGraph

from app.infrastructure.llm import (
    llm_summarize as mock_llm_summarize,
    search as mock_search,
)
from app.domain.models import Source

# ---------------------------------------------------------------------------
# Typed state for the research graph
# ---------------------------------------------------------------------------


class ResearchState(typing.TypedDict, total=False):
    """State used by the research graph."""

    question: str
    findings: list[str]
    sources: list[Source]
    raw_results: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Node functions (async — invoke mock I/O)
# ---------------------------------------------------------------------------


async def search(state: ResearchState) -> dict:
    """Perform a (mock) web search and store results."""
    from datetime import datetime, timezone

    question = state.get("question", "")
    raw_results = await mock_search(question)
    return {
        "raw_results": raw_results,
        "sources": [
            Source(
                id=f"src-{i}",
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                retrieved_at=datetime.now(timezone.utc),
            )
            for i, r in enumerate(raw_results)
        ],
    }


async def summarize(state: ResearchState) -> dict:
    """Summarise findings using a (mock) LLM."""
    question = state.get("question", "")
    raw_results = state.get("raw_results", [])
    findings = await mock_llm_summarize(question, raw_results)
    return {"findings": findings}


# ---------------------------------------------------------------------------
# Graph builder
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
