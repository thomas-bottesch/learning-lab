"""
Temporal Workflow Client: CLI interface

Provides commands to:
  - start a new research workflow
  - list workflows waiting for approval
  - approve an existing workflow
  - reject an existing workflow
  - show the status of a workflow

Usage:

    # Start a new workflow
    python -m app.client start \\
        "Should our company migrate from PostgreSQL to CockroachDB?"

    # List workflows waiting for approval
    python -m app.client list-pending

    # Approve a workflow
    python -m app.client approve research-company-123

    # Reject a workflow
    python -m app.client reject research-company-123

    # Check status
    python -m app.client status research-company-123
"""

import argparse
import asyncio
import os
import sys
import uuid

from temporalio.client import Client as TemporalClient
from temporalio.contrib.pydantic import pydantic_data_converter

from app.domain.models import ResearchRequest
from app.workflows import ResearchWorkflow


async def _build_client() -> TemporalClient:
    """Build a Temporal client from environment variables."""
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    return await TemporalClient.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )


async def cmd_start(args: argparse.Namespace) -> None:
    """Start a new research workflow and return immediately."""
    client = await _build_client()

    request = ResearchRequest(question=args.question)

    # Start (fire-and-forget): returns a handle without blocking on completion.
    handle = await client.start_workflow(
        ResearchWorkflow.run,
        request,
        id=f"research-{uuid.uuid4().hex}",
        task_queue=os.environ.get("TEMPORAL_TASK_QUEUE", "research-agent"),
    )

    print(f"Workflow started:")
    print(f"  workflow_id: {handle.id}")
    print(f"  run_id:      {handle.result_run_id}")
    print()
    print("Use these commands to interact with the running workflow:")
    print(f"  python -m app.client approve  {handle.id}   # approve the proposal")
    print(f"  python -m app.client reject   {handle.id}   # reject the proposal")
    print(f"  python -m app.client status   {handle.id}   # check current status")


async def cmd_signal(args: argparse.Namespace, signal_name: str) -> None:
    """Send an approve/reject signal to an existing workflow."""
    client = await _build_client()

    handle = client.get_workflow_handle(args.workflow_id)

    approved = signal_name == "approve"

    await handle.signal(
        ResearchWorkflow.approve,
        approved,
    )

    print(f"Signal '{signal_name}' sent to workflow '{args.workflow_id}'.")


async def cmd_list_pending(args: argparse.Namespace) -> None:
    """List all ResearchWorkflow executions waiting for human approval."""
    client = await _build_client()
    task_queue = os.environ.get("TEMPORAL_TASK_QUEUE", "research-agent")

    # Query for open (running) workflows on our task queue.
    # Temporal stores open workflows, so we filter by task_queue
    # and exclude any that have already completed.
    page = []
    async for wf in client.list_workflows(
        f'StartTime > "{_iso_now()}" OR StartTime <= "{_iso_now()}"'
    ):
        page.append(wf)

    if not page:
        print("No workflows found.")
        return

    # Inspect each open workflow to find ones waiting for approval.
    # We do this by describing the workflow — if it is RUNNING and has
    # pending signals or a state flag, we flag it as pending.
    pending: list[dict] = []

    for wf_info in page[:50]:  # cap at 50 to avoid flooding stdout
        handle = client.get_workflow_handle(wf_info.id)
        desc = await handle.describe()

        if desc.status.name != "RUNNING":
            continue

        # Count unsignal events / pending tasks to determine if waiting.
        # A workflow that received the approve signal will complete quickly.
        # Workflows with pending workflow tasks that include signals are
        # candidates; however the most reliable heuristic here is:
        #   RUNNING + has input payload matching ResearchWorkflow.run
        # We simply list all RUNNING workflows owned by us and let the user
        # decide.  The status subcommand gives finer detail.
        pending.append(
            {
                "id": wf_info.id,
                "run_id": getattr(desc, "run_id", "N/A"),
                "status": desc.status.name,
                "start_time": str(getattr(desc, "start_time", "N/A")),
            }
        )

    if not pending:
        print("No workflows currently running (all idle / completed).")
        return

    print(f"Found {len(pending)} running workflow(s):")
    print("-" * 80)
    print(f"{'Workflow ID':<55} {'Status':<12} {'Run ID'}")
    print("-" * 80)
    for entry in pending:
        wid = entry["id"]
        if len(wid) > 54:
            wid = wid[:51] + "..."
        print(f"{wid:<55} {entry['status']:<12} {entry['run_id']}")
    print("-" * 80)
    print()
    print("Use these commands to interact with a running workflow:")
    print(f"  python -m app.client approve  <workflow_id>   # approve the proposal")
    print(f"  python -m app.client reject   <workflow_id>   # reject the proposal")
    print(f"  python -m app.client status   <workflow_id>   # check details")


async def cmd_status(args: argparse.Namespace) -> None:
    """Show the current status of a workflow."""
    client = await _build_client()

    handle = client.get_workflow_handle(args.workflow_id)

    try:
        result = await handle.result()
        print(f"Workflow completed:")
        print(f"  workflow_id: {handle.id}")
        print(f"  success:     {result.success}")
        print(f"  detail:      {result.detail}")
    except Exception as exc:
        # Workflow may still be running or completed with failure
        status = await handle.describe()
        print(f"Workflow status:")
        print(f"  workflow_id: {handle.id}")
        print(f"  status:      {status.status.name}")
        if status.error:
            print(f"  error:       {status.error}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph + Temporal Research Agent Client",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- start --
    start_parser = subparsers.add_parser("start", help="Start a new research workflow")
    start_parser.add_argument("question", help="The research question")
    start_parser.set_defaults(func=cmd_start)

    # -- list-pending --
    list_parser = subparsers.add_parser(
        "list-pending",
        help="List workflows running (waiting for approval)",
    )
    list_parser.set_defaults(func=cmd_list_pending)

    # -- approve --
    approve_parser = subparsers.add_parser("approve", help="Approve a workflow")
    approve_parser.add_argument("workflow_id", help="The workflow ID to approve")
    approve_parser.set_defaults(func=lambda a: cmd_signal(a, "approve"))

    # -- reject --
    reject_parser = subparsers.add_parser("reject", help="Reject a workflow")
    reject_parser.add_argument("workflow_id", help="The workflow ID to reject")
    reject_parser.set_defaults(func=lambda a: cmd_signal(a, "reject"))

    # -- status --
    status_parser = subparsers.add_parser("status", help="Show workflow status")
    status_parser.add_argument("workflow_id", help="The workflow ID to check")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
