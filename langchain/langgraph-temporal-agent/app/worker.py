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

import asyncio
import logging
import os

from temporalio.client import Client as TemporalClient
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from app.workflows import ResearchWorkflow


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

    # The workflow references activities by string name (not by import),
    # so it never drags LangChain/LangGraph into the sandbox.
    # Activities are registered individually so Temporal can resolve
    # them by the fully-qualified names used in the workflow.
    from app.activities.research import research
    from app.activities.verification import verify_sources
    from app.activities.proposal import generate_proposal
    from app.activities.side_effects import execute_action, notify_user

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ResearchWorkflow],
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
