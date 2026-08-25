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

OBSERVABILITY (Phase 2):

This Activity is wrapped with ``observe_activity`` from the infrastructure
layer.  All Langfuse integration happens inside the wrapper — this module
knows nothing about Langfuse internals.
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
    Observability is handled by the infrastructure layer.
    """
    # Lazy imports — Activity purity invariant preserved
    from app.graphs import build_research_graph
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
        "activity.type": "research",
        "graph.name": "research-graph",
        "graph.version": "1",
    }

    # Sanitize input before sending to observability (Rule #12)
    safe_input = sanitize({"question": request.question})

    # Get current workflow_id for session-level trace bridging
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="research",
        input_data=safe_input,
        activity_type="research",
        session_id=_wf_id,  # Bridges to workflow-level trace
    ) as obs:
        obs.add_metadata("metadata", extra_meta)

        try:
            graph = build_research_graph()

            # Phase 3 integration: pass Langfuse callbacks into the graph
            result = await graph.ainvoke(
                {"question": request.question},
                config={
                    "callbacks": langfuse_callbacks(),
                    "run_name": "research-graph",
                },
            )

            # Set output summary (Rule #11: don't dump large data into Langfuse)
            obs.set_output(
                {
                    "findings_count": len(result.get("findings", [])),
                    "sources_count": len(result.get("sources", [])),
                }
            )

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

        except Exception as exc:
            obs.set_error(exc)
            raise
