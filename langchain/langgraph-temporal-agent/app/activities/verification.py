"""
Temporal Activity: verify_sources

Verifies the credibility of sources returned by the research phase.

ARCHITECTURAL NOTE — Purity Invariant:

This module MUST NOT import LangChain, LangGraph, or any external I/O code.
All graph/infrastructure imports are lazy (inside the function body).

OBSERVABILITY (Phase 2):

This Activity is wrapped with ``observe_activity`` from the infrastructure
layer.  All Langfuse integration happens inside the wrapper — this module
knows nothing about Langfuse internals.
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
    Observability is handled by the infrastructure layer.
    """
    # Lazy imports — Activity purity invariant preserved
    from app.graphs import build_verification_graph
    from app.infrastructure.observability.callbacks import langfuse_callbacks
    from app.infrastructure.observability.config import get_application_metadata
    from app.infrastructure.observability.metadata import get_temporal_metadata
    from app.infrastructure.observability.redaction import sanitize
    from app.infrastructure.observability.tracing import observe_activity

    # Build metadata for observation
    app_meta = get_application_metadata()
    temporal_meta = get_temporal_metadata()
    extra_meta = {
        **app_meta,
        **temporal_meta,
        "workflow.type": "research",
        "activity.type": "verify_sources",
        "graph.name": "verification-graph",
        "graph.version": "1",
    }

    # Sanitize input before sending to observability (Rule #12)
    # Only include summary info, not full source bodies
    safe_input = {
        "question": research_result.question,
        "sources_count": len(research_result.sources),
    }

    # Get current workflow_id for session-level trace bridging
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="verify_sources",
        input_data=safe_input,
        activity_type="verify_sources",
        session_id=_wf_id,  # Bridges to workflow-level trace
    ) as obs:
        obs.add_metadata("metadata", extra_meta)

        try:
            graph = build_verification_graph()

            # Phase 3 integration: pass Langfuse callbacks into the graph
            result = await graph.ainvoke(
                {
                    "question": research_result.question,
                    "findings": research_result.findings,
                    "sources": research_result.sources,
                },
                config={
                    "callbacks": langfuse_callbacks(),
                    "run_name": "verification-graph",
                },
            )

            # Set output summary (Rule #11: don't dump large data into Langfuse)
            obs.set_output(
                {
                    "verified_sources_count": len(result.get("verified_sources", [])),
                    "rejected_sources_count": len(result.get("rejected_sources", [])),
                }
            )

            activity.logger.info(
                "verify_sources activity completed",
                verified_count=len(result.get("verified_sources", [])),
                rejected_count=len(result.get("rejected_sources", [])),
            )

            return VerifiedResearch(
                question=research_result.question,
                verified_sources=result.get("verified_sources", []),
                rejected_sources=result.get("rejected_sources", []),
            )

        except Exception as exc:
            obs.set_error(exc)
            raise
