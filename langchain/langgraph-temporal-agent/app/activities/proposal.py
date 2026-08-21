"""
Temporal Activity: generate_proposal

Generates a research proposal from verified sources.

ARCHITECTURAL NOTE — Purity Invariant:

This module MUST NOT import LangChain, LangGraph, or any external I/O code.
All graph/infrastructure imports are lazy (inside the function body).
"""

from temporalio import activity

from app.domain.models import VerifiedResearch, Proposal

# ---------------------------------------------------------------------------
# Activity definition
# ---------------------------------------------------------------------------


@activity.defn
async def generate_proposal(verified: VerifiedResearch) -> Proposal:
    """
    Generate a research proposal from verified sources.

    Graph imports are lazy to keep LangChain out of the workflow sandbox.
    """
    from app.graphs import build_proposal_graph  # lazy import

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
