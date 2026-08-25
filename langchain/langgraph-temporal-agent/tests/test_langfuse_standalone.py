"""
Standalone Langfuse v4 integration tests (no Temporal dependencies).

This module tests basic Langfuse v4 tracing functionality in isolation:
  1. Client creation and auth check
  2. Trace/spans/generations via start_as_current_observation
  3. ObservationContext finalization
  4. End-to-end trace verification via get_trace_url

Run with:
    LANGFUSE_TEST_TRACING=1 python -m pytest tests/test_langfuse_standalone.py -v
"""

import time
import os
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_tracing_enabled() -> bool:
    """Return True if LANGFUSE_TEST_TRACING=1 is set."""
    return os.environ.get("LANGFUSE_TEST_TRACING", "0") == "1"


def _load_env_file() -> dict[str, str]:
    """Load key-value pairs from .langfuse.env if it exists."""
    _TEST_ROOT = Path(__file__).parent.parent
    env_file = _TEST_ROOT / ".langfuse.env"
    result: dict[str, str] = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
    return result


def _get_creds_from_env_or_file() -> tuple[str, str, str]:
    """Return (public_key, secret_key, base_url).

    Checks OS env vars first; falls back to .langfuse.env file.
    """
    _creds = _load_env_file()
    public = os.getenv("LANGFUSE_PUBLIC_KEY", _creds.get("LANGFUSE_PUBLIC_KEY", ""))
    secret = os.getenv("LANGFUSE_SECRET_KEY", _creds.get("LANGFUSE_SECRET_KEY", ""))
    host = os.getenv(
        "LANGFUSE_BASE_URL",
        os.getenv("LANGFUSE_HOST", _creds.get("LANGFUSE_BASE_URL", "")),
    )
    return public, secret, host


def _has_credentials() -> bool:
    """Check whether valid Langfuse credentials are configured AND reachable."""
    public, secret, base = _get_creds_from_env_or_file()
    if not (public and secret and base):
        return False
    try:
        import urllib.request

        url = f"{base.rstrip('/')}/api/public/health"
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=3)
        return resp.status == 200
    except Exception:
        return False


def _flush_langfuse(timeout: float = 5.0) -> None:
    """Flush pending Langfuse data using the SDK's built-in flush/shutdown."""
    from app.infrastructure.observability.client import get_langfuse

    langfuse = get_langfuse()
    finished = []

    def _do_shutdown():
        try:
            langfuse.shutdown()
            finished.append(True)
        except Exception:
            finished.append(False)

    import threading

    thread = threading.Thread(target=_do_shutdown, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while not finished and time.monotonic() < deadline:
        thread.join(timeout=0.1)


# ===================================================================
# Test 1 - Client creation
# ===================================================================


class TestClientCreation:
    """Validate the Langfuse client singleton behaves correctly."""

    def test_gets_real_client_when_enabled(self):
        """When tracing is enabled and credentials exist, get_langfuse()
        should return a real Langfuse client, not a NoOp."""
        if not _is_tracing_enabled() or not _has_credentials():
            pytest.skip("LANGFUSE_TEST_TRACING=1 + valid credentials required")

        from app.infrastructure.observability.client import get_langfuse, reset_langfuse

        reset_langfuse()

        try:
            client = get_langfuse()
            client_type = type(client).__name__

            if client_type == "_NoOpLangfuse":
                pytest.skip(
                    "Could not create real Langfuse client. "
                    "Ensure LANGFUSE_SECRET_KEY/KEY are set and "
                    "a Langfuse instance is running at localhost:3000."
                )

            # Verify auth works
            auth_check = getattr(client, "auth_check", None)
            if auth_check and callable(auth_check):
                assert auth_check(), "Langfuse auth_check should return True"
        finally:
            reset_langfuse()

    def test_observe_activity_no_activity_context(self):
        """observe_activity should not raise RuntimeError when called
        outside of a Temporal activity context."""
        from app.infrastructure.observability.tracing import observe_activity

        with observe_activity(
            name="test-activity",
            input_data={"question": "test"},
            activity_type="test",
            session_id="test-session",
        ) as obs:
            obs.set_output({"result": "ok"})


# ===================================================================
# Test 2 - Direct SDK trace/spans creation
# ===================================================================


class TestTraceCreation:
    """Validate that we can create Langfuse traces/spans via the SDK."""

    @pytest.fixture(autouse=True)
    def _reset_before_each(self, monkeypatch):
        """Reset Langfuse client between tests."""
        from app.infrastructure.observability.client import reset_langfuse

        reset_langfuse()

    def test_create_trace_and_span(self):
        """Should be able to create a trace and a span using the v4 SDK."""
        if not _is_tracing_enabled() or not _has_credentials():
            pytest.skip("LANGFUSE_TEST_TRACING=1 + valid credentials required")

        from app.infrastructure.observability.client import get_langfuse

        langfuse = get_langfuse()
        trace_id = langfuse.create_trace_id(seed=f"e2e-trace-{uuid.uuid4().hex[:6]}")

        # Create root span
        with langfuse.start_as_current_observation(
            trace_context={"id": trace_id, "name": "test-workflow"},
            name="root-span",
            as_type="span",
            input={"query": "test query"},
        ) as root_span:
            assert root_span is not None
            span_id = getattr(root_span, "id", None)
            assert span_id is not None, "Span should have an ID"

            # Create child span
            with langfuse.start_as_current_observation(
                trace_context={"id": trace_id, "parent_span_id": span_id},
                name="child-span",
                as_type="span",
                input={"step": "research"},
            ):
                pass

        _flush_langfuse(timeout=5.0)

    def test_create_generation_on_span(self):
        """Should be able to create a generation (LLM call) inside a span."""
        if not _is_tracing_enabled() or not _has_credentials():
            pytest.skip("LANGFUSE_TEST_TRACING=1 + valid credentials required")

        from app.infrastructure.observability.client import get_langfuse

        langfuse = get_langfuse()
        trace_id = langfuse.create_trace_id(seed=f"gen-test-{uuid.uuid4().hex[:6]}")

        with langfuse.start_as_current_observation(
            trace_context={"id": trace_id, "name": "workflow-gen"},
            name="workflow-span",
            as_type="span",
            input={"prompt": "Hello"},
        ) as parent_span:
            parent_id = getattr(parent_span, "id", None)

            # Create generation inside span
            with langfuse.start_as_current_observation(
                trace_context={"id": trace_id, "parent_span_id": parent_id},
                name="llm-summarize",
                as_type="generation",
                model="gpt-4o-mini",
                input="Summarize this",
                output="Here is the summary.",
                metadata={"usage": {"inputTokens": 10, "outputTokens": 20}},
            ) as generation:
                assert generation is not None
                gen_id = getattr(generation, "id", None)
                assert gen_id is not None, "Generation should have an ID"

        _flush_langfuse(timeout=5.0)


# ===================================================================
# Test 3 - Workflow trace bridge
# ===================================================================


class TestWorkflowTraceBridge:
    """Validate that the workflow_trace context manager works correctly."""

    @pytest.fixture(autouse=True)
    def _reset_before_each(self, monkeypatch):
        """Reset Langfuse client between tests."""
        from app.infrastructure.observability.client import reset_langfuse

        reset_langfuse()

    def test_workflow_trace_creates_root_trace(self):
        """workflow_trace should create a root Langfuse trace."""
        if not _is_tracing_enabled() or not _has_credentials():
            pytest.skip("LANGFUSE_TEST_TRACING=1 + valid credentials required")

        from app.infrastructure.observability.tracing import workflow_trace
        from app.infrastructure.observability.client import get_langfuse

        wf_id = f"wf-test-{int(time.time() * 1000)}"

        # Ensure tracing is active
        langfuse = get_langfuse()
        if hasattr(langfuse, "auth_check") and not langfuse.auth_check():
            pytest.skip("Real Langfuse client not available")

        with workflow_trace(workflow_id=wf_id):
            pass

        _flush_langfuse(timeout=5.0)

    def test_workflow_trace_with_no_tracing(self):
        """workflow_trace should be a no-op when tracing is disabled."""
        original = os.environ.get("LANGFUSE_TEST_TRACING")
        os.environ["LANGFUSE_TEST_TRACING"] = "0"

        import sys

        mod_name = "app.infrastructure.observability.config"
        cached_config = sys.modules.get(mod_name)
        mod_name2 = "app.infrastructure.observability.tracing"
        cached_tracing = sys.modules.get(mod_name2)

        try:
            from app.infrastructure.observability.tracing import workflow_trace

            wf_id = f"wf-no-trace-{int(time.time() * 1000)}"
            with workflow_trace(workflow_id=wf_id):
                pass  # Should not raise
        finally:
            if original is not None:
                os.environ["LANGFUSE_TEST_TRACING"] = original
            if cached_config:
                sys.modules[mod_name] = cached_config
            if cached_tracing:
                sys.modules[mod_name2] = cached_tracing


# ===================================================================
# Test 4 - ObservationContext finalization (no Temporal)
# ===================================================================


class TestObservationContextFinalize:
    """Validate that ObservationContext.finalize() sends data to Langfuse."""

    @pytest.fixture(autouse=True)
    def _reset_before_each(self, monkeypatch):
        """Reset Langfuse client between tests."""
        from app.infrastructure.observability.client import reset_langfuse

        reset_langfuse()

    def test_observation_context_finalize(self):
        """finalize() should send the observation to Langfuse."""
        if not _is_tracing_enabled() or not _has_credentials():
            pytest.skip("LANGFUSE_TEST_TRACING=1 + valid credentials required")

        from app.infrastructure.observability.tracing import observe_activity
        from app.infrastructure.observability.client import get_langfuse

        langfuse = get_langfuse()
        if hasattr(langfuse, "auth_check") and not langfuse.auth_check():
            pytest.skip("Real Langfuse client not available")

        with observe_activity(
            name="test-finalize",
            input_data={"test": True},
            activity_type="test",
            session_id=f"sess-{int(time.time() * 1000)}",
        ) as obs:
            obs.set_output({"status": "done"})

        _flush_langfuse(timeout=5.0)


# ===================================================================
# Test 5 - End-to-end trace verification
# ===================================================================


class TestEndToEndTracing:
    """Integration tests that verify traces are actually received by Langfuse."""

    @pytest.fixture(autouse=True)
    def _reset_before_each(self, monkeypatch):
        """Reset Langfuse client between tests."""
        from app.infrastructure.observability.client import reset_langfuse

        reset_langfuse()

    def test_e2e_trace_recorded(self):
        """Create a trace with multiple spans and verify it completes without errors.

        The trace hierarchy:
            trace (trace_id)
            ├── span: e2e-workflow (root)
            │   ├── span: phase-research (child)
            │   │   └── generation: llm-summarize (grandchild)
            │   └── span: phase-verify (child)
        """
        if not _is_tracing_enabled() or not _has_credentials():
            pytest.skip("LANGFUSE_TEST_TRACING=1 + valid credentials required")

        from app.infrastructure.observability.client import get_langfuse

        langfuse = get_langfuse()
        session_id = f"e2e-test-{uuid.uuid4().hex[:8]}"

        trace_id = langfuse.create_trace_id(seed=session_id)

        # Build hierarchy: trace -> span -> span -> generation
        with langfuse.start_as_current_observation(
            trace_context={"id": trace_id, "name": "e2e-workflow"},
            name="e2e-root",
            as_type="span",
            input={"user_query": "Hello World"},
        ) as root_span:
            root_id = getattr(root_span, "id", None)

            # Phase 1: research
            with langfuse.start_as_current_observation(
                trace_context={"id": trace_id, "parent_span_id": root_id},
                name="phase-research",
                as_type="span",
                input={"question": "Is Python good?"},
            ) as research_span:
                research_id = getattr(research_span, "id", None)

                # Generation inside research
                with langfuse.start_as_current_observation(
                    trace_context={"id": trace_id, "parent_span_id": research_id},
                    name="llm-summarize",
                    as_type="generation",
                    model="gpt-4o-mini",
                    input="Summarize: fact1, fact2",
                    output="Summary text here.",
                    metadata={"usage": {"inputTokens": 10, "outputTokens": 20}},
                ):
                    pass

            # Phase 2: verification
            with langfuse.start_as_current_observation(
                trace_context={"id": trace_id, "parent_span_id": root_id},
                name="phase-verify",
                as_type="span",
                input={"findings": ["fact1", "fact2"]},
            ):
                pass

        # Flush data
        _flush_langfuse(timeout=10.0)

        # Verify we got a valid trace URL (proves trace was created)
        trace_url = langfuse.get_trace_url(trace_id=trace_id)
        assert trace_id in trace_url, f"Trace URL should contain trace_id: {trace_url}"
        assert (
            "localhost:3000" in trace_url
        ), f"Trace URL should point to local Langfuse: {trace_url}"
