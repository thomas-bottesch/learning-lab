"""
Temporal Workflow Client: CLI interface

Provides commands to:
  - start a new research workflow
  - approve an existing workflow
  - reject an existing workflow

Usage:

    # Start a new workflow
    python -m app.client start \
        "Should our company migrate from PostgreSQL to CockroachDB?"

    # Approve a workflow
    python -m app.client approve research-company-123

    # Reject a workflow
    python -m app.client reject research-company-123
"""

import argparse
import asyncio
import os
import sys

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
    """Start a new research workflow."""
    client = await _build_client()

    request = ResearchRequest(question=args.question)

    handle = await client.execute_workflow(
        ResearchWorkflow.run,
        request,
        id=f"research-{hash(args.question) & 0xFFFFFFFF:08x}",
        task_queue=os.environ.get("TEMPORAL_TASK_QUEUE", "research-agent"),
    )

    print(f"Workflow started:")
    print(f"  workflow_id: {handle.id}")
    print(f"  run_id:      {handle.result_run_id}")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph + Temporal Research Agent Client",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- start --
    start_parser = subparsers.add_parser("start", help="Start a new research workflow")
    start_parser.add_argument("question", help="The research question")
    start_parser.set_defaults(func=cmd_start)

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
