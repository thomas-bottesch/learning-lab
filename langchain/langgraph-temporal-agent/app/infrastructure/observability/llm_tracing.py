"""
LLM tracing helpers for Langfuse generation observations.

This module provides ``traced_llm_call`` — a context manager that wraps
actual LLM API calls and captures:

  - model name
  - provider
  - input tokens
  - output tokens
  - total tokens
  - latency
  - temperature
  - estimated cost

Covers rules #13 and #20:
  - Instrument actual LLM calls, not just graphs
  - Capture model, tokens, cost in the trace tree

Example trace hierarchy with real LLM:

    research-activity
       └── research-graph
            ├── search [tool]
            │   ├── query
            │   ├── results_count
            │   └── latency
            │
            └── summarize [chain]
                 │
                 └── generation        ← traced_llm_call creates this
                      ├── model: gpt-4
                      ├── input_tokens: 1,421
                      ├── output_tokens: 318
                      ├── latency: 1.8s
                      └── cost: $...

Usage:

    from app.infrastructure.observability.llm_tracing import traced_llm_call

    async def llm_summarize(question: str, sources: list[dict]) -> list[str]:
        async with traced_llm_call(
            model="gpt-4",
            provider="openai",
            input={"question": question, "sources_count": len(sources)},
        ) as tracer:
            response = await chat_model.ainvoke(...)
            tracer.set_usage(response.usage)
            return response.content
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.infrastructure.observability.client import get_langfuse
from app.infrastructure.observability.config import is_tracing_enabled


@dataclass
class TokenUsage:
    """Token usage information from an LLM response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def summary(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMTracer:
    """
    Tracer for a single LLM generation call.

    Captures all metrics needed for rule #13 observability.
    """

    model: str
    provider: str
    input_data: dict[str, Any]
    trace_name: str = "llm-generation"

    _start_time: float = field(default_factory=time.monotonic)
    _usage: TokenUsage | None = None
    _cost: float | None = None
    _response: str | None = None
    _error: dict[str, Any] | None = None
    _metadata: dict[str, Any] = field(default_factory=dict)
    _langfuse_generation: Any = None

    @property
    def duration(self) -> float:
        return time.monotonic() - self._start_time

    def set_usage(self, usage: TokenUsage) -> None:
        """Record token usage from the LLM response."""
        self._usage = usage

    def set_cost(self, cost: float) -> None:
        """Record the estimated cost of this call in USD."""
        self._cost = cost

    def set_response(self, response: str) -> None:
        """Record the response text (optional, may be large)."""
        self._response = response

    def set_error(self, error: Exception) -> None:
        """Record an error from the LLM call."""
        self._error = {
            "type": type(error).__name__,
            "message": str(error),
        }

    def add_metadata(self, key: str, value: Any) -> None:
        """Add arbitrary metadata."""
        self._metadata[key] = value

    def finalize(self) -> None:
        """Send the generation observation to Langfuse."""
        if not is_tracing_enabled() or not self._langfuse_generation:
            return

        try:
            data: dict[str, Any] = {
                "name": self.trace_name,
                "model": self.model,
                "model_parameters": {
                    "temperature": self._metadata.get("temperature"),
                    "max_tokens": self._metadata.get("max_tokens"),
                },
                "input": self.input_data,
                "metadata": {
                    **self._metadata,
                    "provider": self.provider,
                    "duration_seconds": round(self.duration, 4),
                },
            }

            if self._usage:
                data["usage"] = self._usage.summary

            if self._cost is not None:
                data["metadata"]["cost_usd"] = round(self._cost, 6)

            if self._response:
                # Only include response if it's reasonably sized (< 10KB)
                if len(self._response) < 10000:
                    data["output"] = self._response

            if self._error:
                data["status_message"] = self._error["message"]

            self._langfuse_generation.update(**data)

        except Exception:
            pass  # Never let observability failures affect business logic


# ---------------------------------------------------------------------------
# Price estimates per token (USD) — update as needed
# ---------------------------------------------------------------------------

_MODEL_PRICES: dict[str, dict[str, float]] = {
    "gpt-4": {"prompt": 0.00003, "completion": 0.00006},
    "gpt-4o": {"prompt": 0.000005, "completion": 0.000015},
    "gpt-4o-mini": {"prompt": 0.00000015, "completion": 0.0000006},
    "claude-3-opus": {"prompt": 0.000015, "completion": 0.000075},
    "claude-3-sonnet": {"prompt": 0.000003, "completion": 0.000015},
    "claude-3-haiku": {"prompt": 0.00000025, "completion": 0.00000125},
}


def estimate_cost(
    model: str,
    usage: TokenUsage,
) -> float | None:
    """
    Estimate the cost of an LLM call based on token usage.

    Uses rough per-token prices. Update MODEL_PRICES for your models.

    Returns None if the model price is unknown.
    """
    if usage.total_tokens is None:
        return None

    prices = _MODEL_PRICES.get(model)
    if not prices:
        return None

    prompt_cost = (usage.prompt_tokens or 0) * prices["prompt"]
    completion_cost = (usage.completion_tokens or 0) * prices["completion"]
    return prompt_cost + completion_cost


@asynccontextmanager
async def traced_llm_call(
    *,
    model: str,
    provider: str,
    input_data: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    parent_observation: Any = None,
) -> AsyncIterator[LLMTracer]:
    """
    Context manager that traces an LLM generation call.

    Creates a Langfuse generation observation under the current trace/span.

    Parameters
    ----------
    model : str
        The model name (e.g., "gpt-4", "claude-3-opus").
    provider : str
        The provider name (e.g., "openai", "anthropic").
    input_data : dict
        The input to the model (sanitized, no secrets).
    temperature : float, optional
        The temperature parameter.
    max_tokens : int, optional
        The max_tokens parameter.
    parent_observation : any, optional
        Parent Langfuse span/trace to attach this generation to.
        If None, attaches to the current implicit context.

    Yields
    ------
    LLMTracer
        Use this to record usage, cost, and errors.

    Examples
    --------
    >>> async with traced_llm_call(
    ...     model="gpt-4",
    ...     provider="openai",
    ...     input_data={"question": "What is...?"},
    ...     temperature=0.7,
    ... ) as tracer:
    ...     response = await model.ainvoke("...")
    ...     tracer.set_usage(TokenUsage(
    ...         prompt_tokens=100,
    ...         completion_tokens=50,
    ...         total_tokens=150,
    ...     ))
    ...     result = response.content
    """
    langfuse = get_langfuse()
    start = time.monotonic()

    tracer = LLMTracer(
        model=model,
        provider=provider,
        input_data=input_data,
        trace_name=f"{provider}-{model}",
    )

    if temperature is not None:
        tracer.add_metadata("temperature", temperature)
    if max_tokens is not None:
        tracer.add_metadata("max_tokens", max_tokens)

    gen = None
    try:
        if is_tracing_enabled():
            kwargs: dict[str, Any] = {
                "name": f"{provider}-generation",
                "model": model,
                "input": input_data,
                "metadata": {
                    "provider": provider,
                    "model": model,
                },
            }
            if parent_observation is not None:
                kwargs["parent_observation_id"] = getattr(
                    parent_observation, "id", None
                )
            gen = langfuse.generation(**kwargs)
            tracer._langfuse_generation = gen
    except Exception:
        gen = None
        tracer._langfuse_generation = None

    try:
        yield tracer
    except Exception as exc:
        tracer.set_error(exc)
        raise
    finally:
        if gen is not None:
            tracer.finalize()
