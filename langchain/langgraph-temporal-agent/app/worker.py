"""
Temporal Worker: research-agent

Reads Temporal configuration from environment variables, connects to the
existing Temporal Server, registers the ResearchWorkflow and all Activities,
and listens on the configured task queue indefinitely.

Usage:

    python -m app.worker

Multiple instances of this worker can run simultaneously against the same
task queue.  Temporal will distribute tasks across them.
"""

from __future__ import annotations

import asyncio
import logging
import os

from temporalio.client import Client as TemporalClient
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker
from temporalio.worker._workflow_instance import UnsandboxedWorkflowRunner

from app.activities import (
    execute_action,
    generate_proposal,
    notify_user,
    research,
    verify_sources,
)
from app.workflow import ResearchWorkflow


async def build_client() -> TemporalClient:
    """
    Build a Temporal client from environment variables.

    Required env vars:
        TEMPORAL_ADDRESS   — e.g. ``temporal.example.internal:7233``
        TEMPORAL_NAMESPACE — e.g. ``default``

    Optional env vars:
        TEMPORAL_TASK_QUEUE — e.g. ``research-agent`` (default: ``research-agent``)
    """
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")

    return await TemporalClient.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )


async def run_worker() -> None:
    """
    Start the Temporal worker and listen indefinitely.
    """
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    task_queue = os.environ.get("TEMPORAL_TASK_QUEUE", "research-agent")

    client = await build_client()

    # Use UnsandboxedWorkflowRunner because the workflow module imports
    # activity functions at module level, which causes the sandbox to
    # validate all transitive imports (langchain -> requests -> http.client).
    # The workflow code itself is deterministic and performs no I/O, so
    # unsandboxed execution is safe.
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ResearchWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        activities=[
            research,
            verify_sources,
            generate_proposal,
            execute_action,
            notify_user,
        ],
    )

    logging.info(
        "Worker started on task queue '%s', connecting to %s",
        task_queue,
        address,
    )

    await worker.run()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
