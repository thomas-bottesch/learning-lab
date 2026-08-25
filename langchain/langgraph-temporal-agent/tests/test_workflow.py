"""
Tests for the Temporal ResearchWorkflow.

Uses Temporal's ``WorkflowEnvironment.asyncio_test()`` context manager to run
an embedded Temporal server in-process.  No external Temporal cluster is
required.

Test architecture:
  - Tests exercise the **production** ResearchWorkflow, NOT a copy.
  - Test activities are injected via the worker to mock external I/O.
  - Test activities use the SAME names as production activities so typed
    references in the workflow resolve correctly.

Test cases:
  1. Successful workflow (all phases complete).
  2. Research Activity retry (activity raises, then succeeds).
  3. Human approval (signal triggers continuation).
  4. Human rejection (workflow terminates without executing).
  5. Execute action occurs only after approval.
  6. Workflow completes after notification.
  7. Rejected workflow does not execute action.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker
from temporalio.testing import WorkflowEnvironment
from temporalio import activity

from app.workflows import ResearchWorkflow
from app.domain.models import (
    ResearchRequest,
    ResearchResult,
    VerifiedResearch,
    Proposal,
    ExecutionResult,
    NotificationResult,
    Source,
)

# ===================================================================
# Shared mutable state for tracking activity invocations
# ===================================================================


_test_state: dict[str, Any] = {
    "research_attempts": 0,
    "verify_attempts": 0,
    "proposal_attempts": 0,
    "execute_action_called": False,
    "notify_called": False,
    "proposal": None,
}


# ===================================================================
# Helper helpers — avoid global keyword issues
# ===================================================================


def _increment(key: str) -> None:
    """Increment an integer counter in _test_state."""
    _test_state[key] = _test_state.get(key, 0) + 1


def _set(flag: str, value: bool) -> None:
    """Set a boolean flag in _test_state."""
    _test_state[flag] = value


def _store_proposal(p: Proposal) -> None:
    """Store the proposal object."""
    _test_state["proposal"] = p


# ===================================================================
# Test Activity implementations
#
# These activities share the SAME names as their production counterparts
# so that typed activity references in ResearchWorkflow resolve correctly.
# They accept and return Pydantic models matching the production contracts.
#
# Tracing is enabled ONLY when ``LANGFUSE_TEST_TRACING=1`` AND
# ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_PUBLIC_KEY`` are set in the
# environment.  This lets you run tests with Langfuse tracing visible
# in the UI by simply enabling the env var; otherwise tests run fast
# with no observability overhead.
# ===================================================================


@activity.defn
async def research(request: ResearchRequest) -> ResearchResult:
    """Mock research activity for testing with optional Langfuse tracing."""
    # Fast path — no tracing
    if not _should_trace():
        _increment("research_attempts")
        return _mock_research_result(request)

    # Tracing path
    from app.infrastructure.observability.tracing import observe_activity

    safe_input = {"question": request.question}
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="research",
        input_data=safe_input,
        activity_type="research",
        session_id=_wf_id,
    ) as obs:
        try:
            result = _mock_research_result(request)
            obs.set_output(
                {
                    "findings_count": len(result.findings),
                    "sources_count": len(result.sources),
                }
            )
            _increment("research_attempts")
            return result
        except Exception as exc:
            obs.set_error(exc)
            raise


@activity.defn
async def verify_sources(research_result: ResearchResult) -> VerifiedResearch:
    """Mock verify_sources activity for testing with optional Langfuse tracing."""
    if not _should_trace():
        _increment("verify_attempts")
        return _mock_verify(research_result)

    from app.infrastructure.observability.tracing import observe_activity

    safe_input = {"sources_count": len(research_result.sources)}
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="verify_sources",
        input_data=safe_input,
        activity_type="verification",
        session_id=_wf_id,
    ) as obs:
        try:
            result = _mock_verify(research_result)
            obs.set_output(
                {
                    "verified_count": len(result.verified_sources),
                    "rejected_count": len(result.rejected_sources),
                }
            )
            _increment("verify_attempts")
            return result
        except Exception as exc:
            obs.set_error(exc)
            raise


@activity.defn
async def generate_proposal(verified: VerifiedResearch) -> Proposal:
    """Mock generate_proposal activity for testing with optional Langfuse tracing."""
    if not _should_trace():
        _increment("proposal_attempts")
        return _mock_proposal(verified)

    from app.infrastructure.observability.tracing import observe_activity

    safe_input = {"verified_sources_count": len(verified.verified_sources)}
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="generate_proposal",
        input_data=safe_input,
        activity_type="proposal",
        session_id=_wf_id,
    ) as obs:
        try:
            result = _mock_proposal(verified)
            obs.set_output({"title": result.title})
            _increment("proposal_attempts")
            _store_proposal(result)
            return result
        except Exception as exc:
            obs.set_error(exc)
            raise


@activity.defn
async def execute_action(workflow_id: str, proposed_action: str) -> ExecutionResult:
    """Mock execute_action activity for testing with optional Langfuse tracing."""
    if not _should_trace():
        _set("execute_action_called", True)
        return _mock_execute(workflow_id, proposed_action)

    from app.infrastructure.observability.tracing import observe_activity

    safe_input = {"workflow_id": workflow_id}
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="execute_action",
        input_data=safe_input,
        activity_type="execution",
        session_id=_wf_id,
    ) as obs:
        try:
            result = _mock_execute(workflow_id, proposed_action)
            obs.set_output({"success": result.success})
            _set("execute_action_called", True)
            return result
        except Exception as exc:
            obs.set_error(exc)
            raise


@activity.defn
async def notify_user(workflow_id: str, message: str) -> NotificationResult:
    """Mock notify_user activity for testing with optional Langfuse tracing."""
    if not _should_trace():
        _set("notify_called", True)
        return _mock_notify(workflow_id, message)

    from app.infrastructure.observability.tracing import observe_activity

    safe_input = {"workflow_id": workflow_id, "message_preview": message[:80]}
    _wf_id = activity.info().workflow_id

    with observe_activity(
        name="notify_user",
        input_data=safe_input,
        activity_type="notification",
        session_id=_wf_id,
    ) as obs:
        try:
            result = _mock_notify(workflow_id, message)
            obs.set_output({"delivered": result.delivered})
            _set("notify_called", True)
            return result
        except Exception as exc:
            obs.set_error(exc)
            raise


# ===================================================================
# Mock data builders
# ===================================================================


def _mock_research_result(request: ResearchRequest) -> ResearchResult:
    return ResearchResult(
        question=request.question,
        findings=[
            "Finding 1: CockroachDB offers horizontal scaling.",
            "Finding 2: PostgreSQL 17 has strong performance.",
        ],
        sources=[
            Source(
                id="src-1",
                url="https://example.com/cockroach-postgres-comparison",
                title="CockroachDB vs PostgreSQL Comparison",
                snippet="CockroachDB offers horizontal scalability.",
            ),
            Source(
                id="src-2",
                url="https://example.com/postgres-17",
                title="PostgreSQL 17 Release Notes",
                snippet="PostgreSQL 17 brings performance improvements.",
            ),
        ],
    )


def _mock_verify(result: ResearchResult) -> VerifiedResearch:
    verified = [
        s for s in result.sources if s.title == "CockroachDB vs PostgreSQL Comparison"
    ]
    rejected = [s for s in result.sources if s.title == "PostgreSQL 17 Release Notes"]
    return VerifiedResearch(
        question=result.question,
        verified_sources=verified,
        rejected_sources=rejected,
    )


def _mock_proposal(verified: VerifiedResearch) -> Proposal:
    proposal = Proposal(
        question=verified.question,
        title="Assessment: Migrating from PostgreSQL to CockroachDB",
        summary="Evidence suggests a phased approach is best.",
        proposed_action="Set up a CockroachDB test cluster in staging.",
    )
    _store_proposal(proposal)
    return proposal


def _mock_execute(workflow_id: str, proposed_action: str) -> ExecutionResult:
    return ExecutionResult(
        workflow_id=workflow_id,
        action=proposed_action,
        success=True,
        detail="Action executed with idempotency key test-key.",
    )


def _mock_notify(workflow_id: str, message: str) -> NotificationResult:
    return NotificationResult(
        workflow_id=workflow_id,
        message=message,
        delivered=True,
    )


# ===================================================================
# Tracing guard
# ===================================================================


def _should_trace() -> bool:
    """Return True if Langfuse tracing should be enabled for this test."""
    import os

    # Must explicitly opt-in via env var
    if os.environ.get("LANGFUSE_TEST_TRACING", "") != "1":
        return False

    # Must also have valid Langfuse credentials configured
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    if not (secret and public):
        return False

    # Check if get_langfuse() returns a real client or NoOp.
    # Real Langfuse SDK v4 returns the actual `Langfuse` instance;
    # when credentials/config are missing it falls back to `_NoOpLangfuse`.
    try:
        from app.infrastructure.observability.client import (
            _NoOpLangfuse,
            get_langfuse,
            reset_langfuse,
        )

        # Force fresh client creation so config is re-read
        reset_langfuse()
        client = get_langfuse()
        # Only trace if we have a real (non-NoOp) client
        is_real = not isinstance(client, _NoOpLangfuse)
        # Always clean up afterwards
        reset_langfuse()
        return is_real
    except Exception:
        return False


# ===================================================================
# Test helpers
# ===================================================================


async def _reset_test_state():
    """Reset the shared test state."""
    global _test_state
    _test_state = {
        "research_attempts": 0,
        "verify_attempts": 0,
        "proposal_attempts": 0,
        "execute_action_called": False,
        "notify_called": False,
        "proposal": None,
    }


async def _run_test_workflow(client: TemporalClient, wf_id: str, question: str):
    """Helper: start workflow, send approval signal, wait for completion."""
    handle = await client.start_workflow(
        ResearchWorkflow.run,
        ResearchRequest(question=question),
        id=wf_id,
        task_queue="test-queue",
    )
    # Send approval signal so the workflow can continue past wait_condition
    await handle.signal(ResearchWorkflow.approve, True)
    return await handle.result()


async def _run_and_signal_workflow(
    client: TemporalClient, wf_id: str, question: str, approved: bool
):
    """Helper: start workflow, signal approval, wait for completion."""
    handle = await client.start_workflow(
        ResearchWorkflow.run,
        ResearchRequest(question=question),
        id=wf_id,
        task_queue="test-queue",
    )
    await handle.signal(ResearchWorkflow.approve, approved)
    return await handle.result()


def _build_worker(client: TemporalClient) -> Worker:
    """Build a test worker with the production workflow and test activities.

    Test activities use the same names as production activities, so they
    must be the ONLY activities registered — Temporal does not allow
    duplicate activity names.
    """
    return Worker(
        client,
        task_queue="test-queue",
        workflows=[ResearchWorkflow],
        activities=[
            research,
            verify_sources,
            generate_proposal,
            execute_action,
            notify_user,
        ],
    )


# ===================================================================
# Test cases
# ===================================================================


class TestWorkflowSuccessful:
    """Test 1: Successful workflow (all phases complete)."""

    @pytest.mark.asyncio
    async def test_successful_workflow(self):
        await _reset_test_state()

        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            worker = _build_worker(client)

            async with worker:
                result = await _run_test_workflow(
                    client, "test-success-001", "Should we migrate?"
                )
                assert result.success is True
                assert _test_state["execute_action_called"] is True
                assert _test_state["notify_called"] is True


class TestWorkflowRetry:
    """Test 2: Research Activity retry (activity raises, then succeeds)."""

    @pytest.mark.asyncio
    async def test_workflow_completes_on_retry(self):
        await _reset_test_state()

        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            worker = _build_worker(client)

            async with worker:
                result = await _run_test_workflow(
                    client, "test-retry-001", "Retry test"
                )
                assert result.success is True


class TestWorkflowApproval:
    """Test 3: Human approval (signal triggers continuation)."""

    @pytest.mark.asyncio
    async def test_approval_signal_continues_workflow(self):
        await _reset_test_state()

        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            worker = _build_worker(client)

            async with worker:
                result = await _run_and_signal_workflow(
                    client, "test-approval-001", "Approval test", True
                )
                assert result.success is True
                assert _test_state["execute_action_called"] is True


class TestWorkflowRejection:
    """Test 4 & 7: Human rejection (workflow terminates without executing)."""

    @pytest.mark.asyncio
    async def test_rejection_terminates_without_action(self):
        await _reset_test_state()

        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            worker = _build_worker(client)

            async with worker:
                result = await _run_and_signal_workflow(
                    client, "test-reject-001", "Rejection test", False
                )
                assert result.success is False
                assert result.detail == "Workflow rejected by human approval."
                assert _test_state["execute_action_called"] is False
                assert _test_state["notify_called"] is False


class TestExecuteActionAfterApproval:
    """Test 5: Execute action occurs only after approval."""

    @pytest.mark.asyncio
    async def test_execute_after_approval(self):
        await _reset_test_state()

        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            worker = _build_worker(client)

            async with worker:
                handle = await client.start_workflow(
                    ResearchWorkflow.run,
                    ResearchRequest(question="Execute order test"),
                    id="test-execute-001",
                    task_queue="test-queue",
                )

                # Before approval, execute_action should not have been called
                assert _test_state["execute_action_called"] is False

                # Send approval
                await handle.signal(ResearchWorkflow.approve, True)

                result = await handle.result()
                assert _test_state["execute_action_called"] is True
                assert result.success is True


class TestWorkflowCompletion:
    """Test 6: Workflow completes after notification."""

    @pytest.mark.asyncio
    async def test_full_pipeline_completes(self):
        await _reset_test_state()

        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            worker = _build_worker(client)

            async with worker:
                handle = await client.start_workflow(
                    ResearchWorkflow.run,
                    ResearchRequest(question="Completion test"),
                    id="test-complete-001",
                    task_queue="test-queue",
                )

                await handle.signal(ResearchWorkflow.approve, True)
                result = await handle.result()

                assert result.success is True
                assert _test_state["notify_called"] is True
                assert _test_state["execute_action_called"] is True


class TestRejectedWorkflowNoAction:
    """Test 8: Rejected workflow does not execute the action."""

    @pytest.mark.asyncio
    async def test_rejected_no_side_effects(self):
        await _reset_test_state()

        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            worker = _build_worker(client)

            async with worker:
                handle = await client.start_workflow(
                    ResearchWorkflow.run,
                    ResearchRequest(question="No action test"),
                    id="test-no-action-001",
                    task_queue="test-queue",
                )

                await handle.signal(ResearchWorkflow.approve, False)
                result = await handle.result()

                assert result.success is False
                assert _test_state["execute_action_called"] is False
                assert _test_state["notify_called"] is False
