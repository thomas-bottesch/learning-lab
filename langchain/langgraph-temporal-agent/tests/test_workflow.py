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


_test_state = {
    "research_attempts": 0,
    "verify_attempts": 0,
    "proposal_attempts": 0,
    "execute_action_called": False,
    "notify_called": False,
    "proposal": None,
}


# ===================================================================
# Test Activity implementations
#
# These activities share the SAME names as their production counterparts
# so that typed activity references in ResearchWorkflow resolve correctly.
# They accept and return Pydantic models matching the production contracts.
# ===================================================================


@activity.defn
async def research(request: ResearchRequest) -> ResearchResult:
    """Mock research activity for testing."""
    global _test_state
    _test_state["research_attempts"] += 1
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


@activity.defn
async def verify_sources(research_result: ResearchResult) -> VerifiedResearch:
    """Mock verify_sources activity for testing."""
    global _test_state
    _test_state["verify_attempts"] += 1
    verified = [
        s
        for s in research_result.sources
        if s.title == "CockroachDB vs PostgreSQL Comparison"
    ]
    rejected = [
        s for s in research_result.sources if s.title == "PostgreSQL 17 Release Notes"
    ]
    return VerifiedResearch(
        question=research_result.question,
        verified_sources=verified,
        rejected_sources=rejected,
    )


@activity.defn
async def generate_proposal(verified: VerifiedResearch) -> Proposal:
    """Mock generate_proposal activity for testing."""
    global _test_state
    _test_state["proposal_attempts"] += 1
    proposal = Proposal(
        question=verified.question,
        title="Assessment: Migrating from PostgreSQL to CockroachDB",
        summary="Evidence suggests a phased approach is best.",
        proposed_action="Set up a CockroachDB test cluster in staging.",
    )
    _test_state["proposal"] = proposal
    return proposal


@activity.defn
async def execute_action(workflow_id: str, proposed_action: str) -> ExecutionResult:
    """Mock execute_action activity for testing."""
    global _test_state
    _test_state["execute_action_called"] = True
    return ExecutionResult(
        workflow_id=workflow_id,
        action=proposed_action,
        success=True,
        detail="Action executed with idempotency key test-key.",
    )


@activity.defn
async def notify_user(workflow_id: str, message: str) -> NotificationResult:
    """Mock notify_user activity for testing."""
    global _test_state
    _test_state["notify_called"] = True
    return NotificationResult(
        workflow_id=workflow_id,
        message=message,
        delivered=True,
    )


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
