"""
Shared Langfuse v4 client singleton.

This module provides a single shared Langfuse client via ``get_langfuse()``.
The SDK manages its own initialization, batching, and background flushing —
do NOT create a new Langfuse instance per Activity or request.

Langfuse v4 API:
  - Traces are identified by ``trace_id`` (generated via ``create_trace_id(seed=...)``)
  - Spans/generations are created via ``start_as_current_observation(trace_context=..., name=..., as_type=...)``
  - ``trace_context`` is a dict with keys: ``id`` (required), ``parent_span_id`` (optional), ``name`` (optional)
  - Data is sent via OpenTelemetry (OTLP), not REST API endpoints

Usage::

    from app.infrastructure.observability.client import get_langfuse

    langfuse = get_langfuse()  # shared singleton

IMPORTANT: If Langfuse environment variables are not set (e.g., in tests),
this returns a no-op client that does nothing rather than raising an error.
This ensures observability failures never block business logic.
"""

from __future__ import annotations

import os
import time
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langfuse import Langfuse

# Module-level cache for the singleton client.
_langfuse_client: Langfuse | None = None


def _create_langfuse() -> Any:
    """Create a Langfuse v4 client using credentials cached by the config module.

    Credentials are read via :func:`config.get_langfuse_client_kwargs` rather
    than directly from ``os.environ`` so that this function works correctly
    inside Temporal's workflow sandbox.

    Supports three modes:

    1. **Production**: Real SDK key starting with ``sk-lf-``
    2. **Testing**: Localhost target with any non-empty key
    3. **No-op**: No credentials → returns ``None``
    """
    from app.infrastructure.observability.config import (
        get_langfuse_client_kwargs,
        is_tracing_enabled,
    )

    if not is_tracing_enabled():
        return None

    kwargs = get_langfuse_client_kwargs()
    _secret = kwargs.get("secret_key", "")
    _base = kwargs.get("base_url", "")

    with suppress(Exception):
        from langfuse import Langfuse

        is_production = _secret.startswith("sk-lf-")
        is_testing = "localhost" in (_base or "") or "127.0.0.1" in (_base or "")

        if is_production or is_testing:
            client = Langfuse(**kwargs)
            if not _health_check(client):
                return None
            return client

    return None


def _health_check(client: Any, timeout: float = 2.0) -> bool:  # noqa: ARG001
    """Verify the Langfuse client is properly configured and functional.

    For Langfuse v4 we use ``auth_check()`` which verifies credentials are valid.
    Network connectivity is implied since the SDK was initialized without error.

    Parameters
    ----------
    client : Any
        The Langfuse client instance to validate.
    timeout : float
        Maximum seconds to wait (reserved for future use).

    Returns
    -------
    bool
        ``True`` if both checks pass; ``False`` otherwise.
    """
    try:
        # Auth check confirms credentials are valid for the connected server
        if hasattr(client, "auth_check"):
            return bool(client.auth_check())
        # Fallback: just check that core methods exist
        for attr_name in ("flush", "shutdown", "create_trace_id"):
            if not hasattr(client, attr_name):
                return False
        return True
    except Exception:
        return False


def get_langfuse() -> Langfuse:
    """
    Get the shared Langfuse client singleton.

    Returns a real client if configured, otherwise returns a no-op client
    that silently discards all calls. This design ensures:

      1. Observability is non-authoritative (Langfuse down ≠ worker broken)
      2. No retry storms caused by telemetry failures
      3. Clean test environments without Langfuse dependencies

    The client is initialized lazily on first call.
    """
    global _langfuse_client

    if _langfuse_client is None:
        _langfuse_client = _create_langfuse() or _NoOpLangfuse()

    return _langfuse_client  # type: ignore[return-value]


def reset_langfuse():
    """
    Reset the Langfuse client singleton.

    Useful for testing where you need to re-initialize the client
    with different configuration between test runs.

    To prevent the test runner from stalling on a hung ``shutdown()``,
    the call is executed in a daemon thread with a 2-second deadline.
    After that the client is discarded regardless of whether it finished.
    """
    global _langfuse_client

    if _langfuse_client is not None and isinstance(_langfuse_client, _NoOpLangfuse):
        _langfuse_client = None
        return

    if _langfuse_client is None:
        return

    def _do_shutdown():
        try:
            if hasattr(_langfuse_client, "shutdown"):
                _langfuse_client.shutdown()
        except Exception:
            pass

    t = threading.Thread(target=_do_shutdown, daemon=True)
    t.start()
    t.join(timeout=2.0)

    _langfuse_client = None


# ===================================================================
# No-op implementations for Langfuse v4 API
# ===================================================================


class _NoOpLangfuse:
    """
    No-op Langfuse v4 client for environments without tracing configuration.

    Implements the v4 API surface:
      - ``auth_check()`` → returns False
      - ``create_trace_id(seed=...)`` → returns deterministic string
      - ``start_as_current_observation(...)`` → returns no-op context manager
      - ``flush()``, ``shutdown()`` → no-ops
    """

    def auth_check(self) -> bool:
        """Return False to indicate no real client is available."""
        return False

    def create_trace_id(self, *, seed: str | None = None) -> str:
        """Return a fake trace ID."""
        return f"noop-trace-{seed or 'default'}"

    def start_as_current_observation(
        self,
        trace_context: dict[str, Any] | None = None,
        name: str = "noop",
        as_type: str = "span",
        **kwargs: Any,
    ) -> "_NoOpSpan":
        """Return a no-op span context manager."""
        return _NoOpSpan()

    def flush(self, *args: Any, **kwargs: Any) -> None:
        pass

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_trace_url(self, trace_id: str, **kwargs: Any) -> str:
        return f"noop://trace/{trace_id}"

    def score_trace(self, *args: Any, **kwargs: Any) -> "_NoOpScore":
        return _NoOpScore()

    def update_current_span(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update_current_generation(self, *args: Any, **kwargs: Any) -> None:
        pass

    def create_event(self, *args: Any, **kwargs: Any) -> "_NoOpResult":
        return _NoOpResult()

    def start_observation(self, *args: Any, **kwargs: Any) -> "_NoOpSpan":
        return _NoOpSpan()

    def set_current_trace_as_public(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_current_trace_io(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoOpSpan:
    """No-op span returned by start_as_current_observation(as_type='span')."""

    id: str = "noop-span"

    def __init__(self, **kwargs: Any):
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass

    def score(self, *args: Any, **kwargs: Any) -> "_NoOpScore":
        return _NoOpScore()

    def score_trace(self, *args: Any, **kwargs: Any) -> "_NoOpScore":
        return _NoOpScore()

    def create_event(self, *args: Any, **kwargs: Any) -> "_NoOpResult":
        return _NoOpResult()

    def start_as_current_observation(self, **kwargs: Any) -> "_NoOpSpan":
        return _NoOpSpan()

    def start_observation(self, **kwargs: Any) -> "_NoOpSpan":
        return _NoOpSpan()

    def set_trace_as_public(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_trace_io(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoOpGeneration(_NoOpSpan):
    """No-op generation returned by start_as_current_observation(as_type='generation')."""

    id: str = "noop-generation"


class _NoOpScore:
    """No-op score."""

    id: str = "noop-score"

    def __enter__(self) -> "_NoOpScore":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpResult:
    """Generic no-op return value."""

    def __repr__(self) -> str:
        return "<NoOp>"

    def __enter__(self) -> "_NoOpResult":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
