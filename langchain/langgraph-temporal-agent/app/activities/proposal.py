"""
Temporal Activity: generate_proposal

Generates a research proposal from verified sources.

ARCHITECTURAL NOTE — Purity Invariant:

This module MUST NOT import LangChain, LangGraph, or any external I/O code.
All graph/infrastructure imports are lazy (inside the function body).

OBSERVABILITY (Phase 2):

This Activity is wrapped with ``observe_activity`` from the infrastructure
layer.  All Langfuse integration happens inside the wrapper — this module
knows nothing about Langfuse internals.
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
    Observability is handled by the infrastructure layer.
    """
    # Lazy imports — Activity purity invariant preserved
    from app.graphs import build_proposal_graph
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
        "activity.type": "generate_proposal",
        "graph.name": "proposal-graph",
        "graph.version": "1",
    }

    # Sanitize input before sending to observability (Rule #12)
    # Only include summary info, not full source bodies
    safe_input = {
        "question": verified.question,
        "verified_sources_count": len(verified.verified_sources),
        "rejected_sources_count": len(verified.rejected_sources),
    }

    # Get current workflow_id for session-level trace bridging
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="generate_proposal",
        input_data=safe_input,
        activity_type="generate_proposal",
        session_id=_wf_id,  # Bridges to workflow-level trace
    ) as obs:
        obs.add_metadata("metadata", extra_meta)

        try:
            graph = build_proposal_graph()

            # Phase 3 integration: pass Langfuse callbacks into the graph
            result = await graph.ainvoke(
                {
                    "question": verified.question,
                    "verified_sources": verified.verified_sources,
                    "rejected_sources": verified.rejected_sources,
                },
                config={
                    "callbacks": langfuse_callbacks(),
                    "run_name": "proposal-graph",
                },
            )

            # Set output summary (Rule #11: don't dump large data into Langfuse)
            obs.set_output(
                {
                    "title": result.get("title", ""),
                    "summary_length": len(result.get("summary", "")),
                    "proposed_action_length": len(result.get("proposed_action", "")),
                }
            )

            activity.logger.info(
                "generate_proposal activity completed",
                title=result.get("title", ""),
            )

            return Proposal(
                question=verified.question,
                title=result.get("title", ""),
                summary=result.get("summary", ""),
                proposed_action=result.get("proposed_action", ""),
            )

        except Exception as exc:
            obs.set_error(exc)
            raise
