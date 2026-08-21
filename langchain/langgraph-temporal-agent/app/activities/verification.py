"""
Temporal Activity: verify_sources

Verifies the credibility of sources returned by the research phase.

ARCHITECTURAL NOTE — Purity Invariant:

This module MUST NOT import LangChain, LangGraph, or any external I/O code.
All graph/infrastructure imports are lazy (inside the function body).
"""

from temporalio import activity

from app.domain.models import ResearchResult, VerifiedResearch

# ---------------------------------------------------------------------------
# Activity definition
# ---------------------------------------------------------------------------


@activity.defn
async def verify_sources(research_result: ResearchResult) -> VerifiedResearch:
    """
    Verify the credibility of sources returned by the research phase.

    Graph imports are lazy to keep LangChain out of the workflow sandbox.
    """
    from app.graphs import build_verification_graph  # lazy import

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
