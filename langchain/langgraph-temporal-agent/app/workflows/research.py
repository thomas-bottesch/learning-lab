"""
Temporal Workflow: ResearchWorkflow

ARCHITECTURAL PRINCIPLE — Temporal Determinism:

Temporal Workflow code MUST NOT:
  - call an LLM
  - make HTTP requests
  - access a database
  - read arbitrary files
  - use random numbers directly
  - depend on current wall-clock time except through Temporal's APIs
  - access environment-dependent state
  - perform external I/O
  - call arbitrary nondeterministic Python code

These operations belong **inside Activities**.

The Workflow is ONLY responsible for:
  - deterministic orchestration (calling Activities in sequence)
  - Temporal-native operations (signals, conditions, timers, child workflows)

LANGGRAPH INTEGRATION:

LangGraph is used **inside** Activities (research, verify_sources,
generate_proposal) for bounded agent reasoning.  Temporal owns the
durable outer workflow; LangGraph owns the inner graph logic.

ACTIVITY REFERENCES — TYPED:

This workflow imports activity callables directly (not by string name).
This is safe because each activity module uses **lazy imports** for
graph/infrastructure code — the import-time dependency graph is:

    activities.research → domain.models ✅ (pure Pydantic)
    activities.research → temporalio ✅ (Temporal SDK)

    (at runtime) → graphs → infrastructure.llm ✅ (only when invoked)

Typed references are required for Pydantic serialization to work correctly
with Temporal's `pydantic_data_converter`.

Dependency graph:

                    ┌──────────────┐
                    │   domain     │
                    │    models    │
                    └──────▲───────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
       ┌─────┴─────┐               ┌─────┴─────┐
       │ workflows │               │ activities │
       └─────┬─────┘               └─────┬─────┘
             │                           │
             │                           ▼
             │                       ┌─────────┐
             │                       │ graphs  │
             │                       └─────────┘
             │
             ▼
          Temporal
"""

from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

from app.activities import (
    execute_action,
    generate_proposal,
    notify_user,
    research,
    verify_sources,
)
from app.domain.models import (
    ExecutionResult,
    NotificationResult,
    Proposal,
    ResearchRequest,
    ResearchResult,
    VerifiedResearch,
)

# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------


@workflow.defn
class ResearchWorkflow:
    """
    Durable research-and-approval workflow.

    Execution flow:

        research (LangGraph inside Activity)
            ↓
        verify_sources (LangGraph inside Activity)
            ↓
        generate_proposal (LangGraph inside Activity)
            ↓
        WAIT FOR HUMAN APPROVAL (Temporal Signal)
            ↓
        execute_action
            ↓
        notify_user
            ↓
        DONE
    """

    def __init__(self) -> None:
        # Human approval state — set by the ``approve`` signal.
        self.approval_received: bool | None = None
        self.approved: bool = False

    # ------------------------------------------------------------------
    # Signal: human approval
    # ------------------------------------------------------------------

    @workflow.signal
    async def approve(self, approved: bool) -> None:
        """
        Signal to approve or reject the generated proposal.

        This signal can arrive at any time — even days after the workflow
        reaches the WAIT state.  Temporal owns the durable wait; no Python
        process needs to stay alive.

        Parameters
        ----------
        approved : bool
            ``True``  → proceed with execution.
            ``False`` → terminate the workflow without executing.
        """
        self.approval_received = True
        self.approved = approved
        workflow.logger.info(
            "approval signal received",
            approved=approved,
        )

    # ------------------------------------------------------------------
    # Workflow definition
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, request: ResearchRequest) -> ExecutionResult:
        """
        Execute the full research-and-approval workflow.
        """
        workflow.logger.info(
            "ResearchWorkflow started",
            question=request.question,
        )

        # ---- Phase 1: Research (LangGraph inside Activity) ----
        research_result: ResearchResult = await workflow.execute_activity(
            research,
            request,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=2),
                maximum_attempts=5,
            ),
        )

        # ---- Phase 2: Verify sources (LangGraph inside Activity) ----
        verified: VerifiedResearch = await workflow.execute_activity(
            verify_sources,
            research_result,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=2),
                maximum_attempts=5,
            ),
        )

        # ---- Phase 3: Generate proposal (LangGraph inside Activity) ----
        proposal: Proposal = await workflow.execute_activity(
            generate_proposal,
            verified,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=2),
                maximum_attempts=3,
            ),
        )

        workflow.logger.info(
            "Proposal generated",
            title=proposal.title,
        )

        # ---- Phase 4: Wait for human approval (Temporal Signal) ----
        #
        # This wait is DURABLE.  If the Python worker crashes, restarts,
        # or goes offline, Temporal retains the workflow in a WAITING
        # state.  When a worker comes back online (or a new one starts),
        # Temporal will resume the workflow from this point.
        #
        # No Python process is kept busy during the wait.

        await workflow.wait_condition(
            lambda: self.approval_received is not None,
        )

        if not self.approved:
            workflow.logger.info(
                "Workflow rejected by human approval. Terminating.",
            )
            return ExecutionResult(
                workflow_id=workflow.info().workflow_id,
                action=proposal.proposed_action,
                success=False,
                detail="Workflow rejected by human approval.",
            )

        workflow.logger.info("Workflow approved — proceeding to execution.")

        # ---- Phase 5: Execute action ----
        #
        # Retry policy: short retries for transient failures, but no
        # infinite retry loop.  Permanent failures (invalid request,
        # auth failure, business rule violation) are raised with
        # non_retryable=True inside the Activity and will fail immediately.
        execution_result: ExecutionResult = await workflow.execute_activity(
            execute_action,
            args=(workflow.info().workflow_id, proposal.proposed_action),
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
            ),
        )

        # ---- Phase 6: Notify user ----
        #
        # Notification Activity uses internal deduplication keys.
        # A small retry budget protects against transient delivery failures.
        await workflow.execute_activity(
            notify_user,
            args=(
                workflow.info().workflow_id,
                (
                    f"Your research on '{request.question}' has been completed. "
                    f"Action: {execution_result.detail}"
                ),
            ),
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=5),
                maximum_attempts=2,
            ),
        )

        workflow.logger.info(
            "ResearchWorkflow completed successfully.",
            workflow_id=workflow.info().workflow_id,
        )

        return execution_result
