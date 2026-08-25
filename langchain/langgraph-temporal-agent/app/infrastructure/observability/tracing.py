"""
Activity-level tracing helpers with workflow-level trace bridging.

This module provides ``observe_activity`` — a context manager that wraps
each AI Activity with consistent observability AND connects it to a
unified Langfuse trace representing the entire workflow:

  - Temporal metadata propagation (rules #6, #8, #17)
  - Input/output capture (rule #11)
  - Duration and error tracking (rule #14)
  - Activity attempt tracking (rules #1, #15)
  - **Workflow-level trace bridging** — activity spans become children
    of the workflow trace (NEW: covers rule #1)

Langfuse v4 API usage:
  - Traces are identified by ``trace_id`` (generated via ``create_trace_id(seed=...)``)
  - Spans/generations are created via ``start_as_current_observation(trace_context=..., name=..., as_type=...)``
  - ``trace_context`` dict keys: ``id`` (required, trace_id), ``parent_span_id`` (optional), ``name`` (optional)

Trace Hierarchy::

    trace (workflow_id = session_id)
    ├── metadata: version, git_sha, environment
    ├── span: verify_sources (activity)
    │   ├── input / output / duration
    │   └── child spans (via callbacks): verification-graph
    ├── span: research (activity)
    │   └── child spans: research-graph → LLM generations
    └── span: generate_proposal (activity)
        └── child spans: proposal-graph

Usage::

    from app.infrastructure.observability import observe_activity

    @activity.defn
    async def research(request: ResearchRequest) -> ResearchResult:
        with observe_activity(
            name="research",
            input={"question": request.question},
            activity_type="research",
            session_id="<TEMPORAL_WORKFLOW_ID>",  # Bridges to workflow trace
        ) as obs:
            result = await graph.ainvoke({"question": request.question})
            obs.set_output({
                "findings_count": len(result["findings"]),
                "sources_count": len(result["sources"]),
            })
            return ResearchResult(...)
"""

from __future__ import annotations

import contextlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterator

from temporalio import activity

from app.infrastructure.observability.client import get_langfuse
from app.infrastructure.observability.config import (
    is_tracing_enabled,
    should_sample,
)
from app.infrastructure.observability.metadata import (
    build_execution_id,
    get_temporal_metadata,
)

# ---------------------------------------------------------------------------
# Trace ID propagation — production-safe design
# ---------------------------------------------------------------------------
#
# CROSS-THREAD PROPAGATION (Workflow → Activity):
#
# Temporal workers execute activities in separate threads/pools. Thread-local
# state DOES NOT propagate across threads. To bridge workflow→activity traces,
# we use a GLOBAL WORKFLOW TRACE MAP:
#
#   _workflow_trace_map: dict[workflow_id, trace_id]
#
# This map is populated by ``workflow_trace()`` before activities start,
# and read by ``observe_activity()`` using ``activity.info().workflow_id``.
#
# THREAD-LOCAL STATE (Intra-thread nesting):
#   Used ONLY for nested observations WITHIN the same activity execution
#   (e.g., when observe_activity wraps another observe_activity call).
#
# TEST-SAFE: Both mechanisms coexist; thread-local takes priority when
# available (tests), while the global map handles production scenarios.

import threading
from typing import Optional

# Global workflow→trace_id mapping (cross-thread propagation)
_workflow_trace_map: dict[str, str] = {}
_map_lock = threading.Lock()

# Thread-local storage (intra-thread nesting)
_local = threading.local()


def register_workflow_trace(workflow_id: str, trace_id: str) -> None:
    """Register a trace_id for a workflow_id (called by workflow_trace)."""
    with _map_lock:
        _workflow_trace_map[workflow_id] = trace_id


def unregister_workflow_trace(workflow_id: str) -> None:
    """Remove a workflow_id from the trace map (cleanup)."""
    with _map_lock:
        _workflow_trace_map.pop(workflow_id, None)


def resolve_trace_id(workflow_id: str) -> str | None:
    """Resolve the active trace_id for a given workflow_id.

    Priority:
    1. Thread-local active_trace_id (test/intra-thread context)
    2. Global workflow→trace_id map (production cross-thread context)
    """
    # Check thread-local first (fast path for tests)
    tl_trace_id = getattr(_local, "active_trace_id", None)
    if tl_trace_id:
        return tl_trace_id

    # Fall back to global map (production path)
    with _map_lock:
        return _workflow_trace_map.get(workflow_id)


def _set_active_trace_id(trace_id: str | None) -> None:
    """Set active trace_id for current thread (intra-thread propagation only)."""
    _local.active_trace_id = trace_id


def _get_active_trace_id() -> str | None:
    """Get active trace_id from thread-local (intra-thread only)."""
    return getattr(_local, "active_trace_id", None)


def _set_active_parent_span_id(span_id: str | None) -> None:
    """Set current parent span ID for nested observations within same thread."""
    _local.active_parent_span_id = span_id


def _clear_thread_local() -> None:
    """Clear all thread-local state (for testing/cleanup)."""
    _local.active_trace_id = None
    _local.active_parent_span_id = None


@dataclass
class ObservationContext:
    """
    Context object passed through the observe_activity block.

    Provides methods to set output, errors, and additional metadata
    on the Langfuse observation.
    """

    name: str
    execution_id: str
    trace_id: str | None = None
    session_id: str | None = None
    span_id: str | None = None
    _start_time: float = field(default_factory=time.monotonic)
    _output: dict[str, Any] | None = None
    _error: dict[str, Any] | None = None
    _metadata: dict[str, Any] = field(default_factory=dict)
    _langfuse_span: Any = None

    @property
    def duration(self) -> float:
        """Elapsed seconds since observation started."""
        return time.monotonic() - self._start_time

    def set_output(self, output: dict[str, Any]) -> None:
        """Set the output summary for this observation."""
        self._output = output

    def set_error(self, error: Exception) -> None:
        """Record an error for this observation."""
        self._error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    def add_metadata(self, key: str, value: Any) -> None:
        """Add arbitrary metadata to this observation."""
        self._metadata[key] = value

    def finalize(self) -> None:
        """Finalize the Langfuse span with output/error/metadata."""
        if not is_tracing_enabled() or not self._langfuse_span:
            return

        try:
            update_data: dict[str, Any] = {}

            if self._output:
                update_data["output"] = self._output

            if self._error:
                update_data["error"] = self._error
                update_data["level"] = "ERROR"

            meta = {**self._metadata, "duration_seconds": round(self.duration, 4)}
            if meta:
                update_data["metadata"] = meta

            # Update the span with final data
            self._langfuse_span.update(**update_data)

            # Mark span as ended
            if hasattr(self._langfuse_span, "end"):
                self._langfuse_span.end()

        except Exception:
            # Never let observability failures affect business logic
            pass


@contextlib.contextmanager
def observe_activity(
    *,
    name: str,
    input_data: dict[str, Any] | None = None,
    activity_type: str | None = None,
    session_id: str | None = None,
) -> Iterator[ObservationContext]:
    """
    Context manager that wraps an Activity execution with Langfuse tracing.

    When ``session_id`` is provided, activity spans are attached as children
    of the workflow-level Langfuse trace. Without it, each activity creates
    its own independent span (fallback mode).

    Parameters
    ----------
    name : str
        Human-readable name for this observation (e.g., "research").
    input_data : dict, optional
        Sanitized input to record. Should NOT contain secrets or large bodies.
    activity_type : str, optional
        Override for the activity type. Defaults to name.
    session_id : str, optional
        The workflow's session_id (usually the Temporal workflow_id).
        When provided, activity spans become children of the workflow trace.

    Yields
    ------
    ObservationContext
        Use this to set output, errors, and metadata.
    """
    # Rule #23: If sampling is disabled, return no-op immediately
    if not should_sample():
        ctx = ObservationContext(name=name, execution_id="unsampled")
        yield ctx
        return

    langfuse = get_langfuse()

    # Check if we have a real client (not NoOp)
    if (
        isinstance(langfuse, type(langfuse).__bases__[0])
        if langfuse.__class__.__bases__
        else True
    ):
        pass  # We'll detect NoOp by checking auth_check

    if hasattr(langfuse, "auth_check") and not langfuse.auth_check():
        # NoOp client
        execution_id = "noop"
        try:
            _act_info = activity.info()
            execution_id = build_execution_id(
                workflow_id=_act_info.workflow_id,
                activity_id=_act_info.activity_id,
                attempt=_act_info.attempt,
            )
        except RuntimeError:
            pass
        ctx = ObservationContext(name=name, execution_id=execution_id)
        yield ctx
        return

    # Attempt to fetch the current Temporal activity context.
    try:
        _act_info = activity.info()
    except RuntimeError:
        execution_id = "no-activity-context"
        ctx = ObservationContext(name=name, execution_id=execution_id)
        yield ctx
        return

    # Resolve session_id: explicit param > Temporal workflow_id
    resolved_session_id = session_id or _act_info.workflow_id

    execution_id = build_execution_id(
        workflow_id=_act_info.workflow_id,
        activity_id=_act_info.activity_id,
        attempt=_act_info.attempt,
    )

    act_type = activity_type or name

    # Build standard metadata (rule #22: observability contract)
    temporal_meta = get_temporal_metadata()
    base_metadata = {
        "workflow.type": "research",
        "activity.type": act_type,
        "activity.attempt": _act_info.attempt,
        **temporal_meta,
    }

    # Resolve trace_id: use workflow_id to look up from global map (production)
    # or thread-local (tests). Falls back to session_id as unique identifier.
    trace_id = resolve_trace_id(_act_info.workflow_id) or resolved_session_id

    span = None
    span_obj = None

    try:
        if is_tracing_enabled():
            trace_context: dict[str, Any] = {"id": trace_id}

            # Add parent span if we're in a nested context
            parent_span = getattr(_local, "active_parent_span_id", None)
            if parent_span:
                trace_context["parent_span_id"] = parent_span

            # Also add the workflow trace name
            if resolved_session_id:
                trace_context["name"] = f"workflow:{resolved_session_id}"

            # Create span using v4 API
            span_obj = langfuse.start_as_current_observation(
                trace_context=trace_context,
                name=f"{act_type}-activity",
                as_type="span",
                input=input_data or {},
                metadata={
                    **base_metadata,
                    "tracing.mode": "bridged" if resolved_session_id else "standalone",
                },
            )

            # Store span reference and update thread-local for nesting
            span = span_obj
            _set_active_parent_span_id(getattr(span_obj, "id", None))

    except Exception:
        span = None

    ctx = ObservationContext(
        name=name,
        execution_id=execution_id,
        trace_id=trace_id,
        session_id=resolved_session_id,
        span_id=getattr(span, "id", None) if span else None,
        _langfuse_span=span,
    )

    try:
        yield ctx
    except Exception as exc:
        ctx.set_error(exc)
        raise
    finally:
        # Restore previous parent span ID
        if span is not None:
            _set_active_parent_span_id(None)
            ctx.finalize()


@contextlib.contextmanager
def workflow_trace(
    *,
    workflow_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """
    Context manager that creates a workflow-level Langfuse trace.

    This is called ONCE at the start of the workflow to create the root
    trace. All subsequent ``observe_activity`` calls with matching
    ``session_id`` will attach their spans to this trace.

    Covers rules #1 and #2:
      - One trace = one logical workflow
      - Centralized infrastructure module

    Usage::

        from app.infrastructure.observability.tracing import workflow_trace

        @workflow.defn
        class ResearchWorkflow:
            @workflow.run
            async def run(self, request: ResearchRequest):
                with workflow_trace(
                    workflow_id=workflow.info().workflow_id,
                    extra_metadata={"user_id": request.user_id},
                ):
                    # All activities within this scope bridge to this trace
                    ...
    """
    if not is_tracing_enabled() or not should_sample():
        yield None
        return

    langfuse = get_langfuse()

    # Check if this is a NoOp client
    if hasattr(langfuse, "auth_check") and not langfuse.auth_check():
        yield None
        return

    app_meta = _get_application_metadata_for_trace()

    try:
        # Create a unique trace_id for this workflow
        trace_id = langfuse.create_trace_id(seed=workflow_id)

        # Register in global map for cross-thread propagation (production)
        register_workflow_trace(workflow_id, trace_id)

        # Also set thread-local for intra-thread contexts (tests)
        _set_active_trace_id(trace_id)

        # Create root span for the workflow
        trace_context = {"id": trace_id, "name": f"workflow:{workflow_id}"}
        root_span = langfuse.start_as_current_observation(
            trace_context=trace_context,
            name="research-workflow",
            as_type="span",
            input={"workflow_id": workflow_id},
            metadata={**app_meta, **(extra_metadata or {})},
        )

        yield root_span
    except Exception:
        yield None
    finally:
        # Clean up both thread-local and global map
        unregister_workflow_trace(workflow_id)
        _set_active_trace_id(None)


def _get_application_metadata_for_trace() -> dict[str, Any]:
    """Return application-level metadata for traces."""
    from app.infrastructure.observability.config import (
        APP_VERSION,
        ENVIRONMENT,
        GIT_SHA,
        LANGFUSE_TRACING_ENVIRONMENT,
    )

    result: dict[str, Any] = {
        "application.version": APP_VERSION,
        "git.sha": GIT_SHA,
        "environment": LANGFUSE_TRACING_ENVIRONMENT,
    }
    return result


def activity_observer(
    name: str,
    activity_type: str | None = None,
) -> Any:
    """
    Decorator-style factory for Activity observation.

    Because Activities must be decorated at definition time but we need
    lazy imports, this returns a pattern you use conceptually:

        @activity.defn
        async def research(...):
            with activity_observer_wrapper("research"):
                ...

    Note: Python decorators can't wrap async functions cleanly while also
    returning the original signature, so we provide the context manager
    directly rather than using this as a decorator.
    """
    return observe_activity(name=name, activity_type=activity_type)
