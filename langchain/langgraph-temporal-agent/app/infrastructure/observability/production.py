"""
Production-hardened Activity wrapper with observability failure isolation.

This module provides ``safe_activity`` — an async context manager that wraps
Activity execution with these guarantees (rules #23, #24, #25):

  1. Business result ≠ Langfuse result
  2. Observability outage does NOT cause retry storms
  3. Tracing is NON-AUTHORITATIVE — worker continues if Langfuse fails

Key invariant:

    try:
        business_result = await do_business_work()
        await send_to_langfuse(...)  # Best effort only
    except Exception:
        raise business_result  # Never expose telemetry errors as business errors

Usage:

    from app.infrastructure.observability.production import safe_activity

    @activity.defn
    async def my_activity(request: MyRequest) -> MyResult:
        with safe_activity(
            name="my-activity",
            input_data=sanitize({"question": request.question}),
        ) as obs:
            result = await graph.ainvoke({"question": request.question})
            obs.set_output(summary(result))
            return result

The difference from bare ``observe_activity``:
  - ``safe_activity`` catches ALL exceptions during Langfuse finalize()
  - Even if Langfuse crashes, the Activity completes normally
  - Only logs the error internally, never raises
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from app.infrastructure.observability.config import (
    IS_TRACING_ENABLED,
    should_sample,
)
from app.infrastructure.observability.tracing import (
    ObservationContext,
    observe_activity,
)

logger = logging.getLogger(__name__)


@contextmanager
def safe_activity(
    *,
    name: str,
    input_data: dict[str, Any] | None = None,
    activity_type: str | None = None,
) -> Iterator[ObservationContext]:
    """
    Context manager that wraps an Activity with safe observability.

    This is the PRODUCTION-SAFE version of ``observe_activity``.
    It ensures:
      1. The Activity body executes regardless of Langfuse status
      2. Langfuse failures are logged but NEVER raised
      3. Sampling is checked before creating any traces

    Parameters
    ----------
    name : str
        Activity name for the observation.
    input_data : dict, optional
        Sanitized input data to record.
    activity_type : str, optional
        Override activity type.

    Yields
    ------
    ObservationContext
        Use this to set output, errors, metadata.

    Examples
    --------
    >>> with safe_activity(name="research", input_data={"question": "..."}):
    ...     result = await graph.ainvoke(...)
    ...     obs.set_output({"count": len(result)})
    """
    # Rule #26: Check sampling first (avoid unnecessary work)
    if not should_sample():
        ctx = ObservationContext(name=name, execution_id="unsampled")
        yield ctx
        return

    observer = None
    try:
        observer = observe_activity(
            name=name,
            input_data=input_data,
            activity_type=activity_type,
        )
        ctx = next(observer)
        yield ctx

        # Finalize safely — catch everything
        if hasattr(ctx, "finalize"):
            try:
                ctx.finalize()
            except Exception as e:
                logger.warning(
                    "Observability finalization failed for '%s': %s. "
                    "Business work was unaffected.",
                    name,
                    e,
                )
    except StopIteration:
        pass
    finally:
        # Close the generator to ensure cleanup runs
        if observer is not None:
            try:
                observer.throw(None)
            except StopIteration:
                pass
            except Exception as e:
                logger.warning(
                    "Observability cleanup failed for '%s': %s. "
                    "Business work was unaffected.",
                    name,
                    e,
                )
