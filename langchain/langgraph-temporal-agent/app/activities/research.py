"""
Temporal Activity: research

Runs the Research LangGraph to investigate a given question.

ARCHITECTURAL NOTE — Purity Invariant:

This module MUST NOT import LangChain, LangGraph, or any external I/O code.
All graph/infrastructure imports are lazy (inside the function body) to
preserve this invariant.  The Temporal worker registers this module at
import time, so the import-time dependency graph is:

    activities.research → domain.models ✅ (pure Pydantic)
    activities.research → temporalio ✅ (Temporal SDK)

    (at runtime) → graphs → infrastructure.llm ✅ (only when invoked)
"""

from temporalio import activity

from app.domain.models import ResearchRequest, ResearchResult

# ---------------------------------------------------------------------------
# Activity definition
# ---------------------------------------------------------------------------


@activity.defn
async def research(request: ResearchRequest) -> ResearchResult:
    """
    Run the Research LangGraph to investigate the given question.

    Graph imports are lazy to keep LangChain out of the workflow sandbox.
    """
    from app.graphs import build_research_graph  # lazy import

    activity.logger.info("research activity started", question=request.question)

    graph = build_research_graph()
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
