"""Forgejo Actions MCP server — run with: python -m mcp_servers.forgejo"""

from io import StringIO
from typing import Annotated
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from . import client as fg

mcp = FastMCP("forgejo-actions")

_user, _password = fg.get_credentials()


# ── Return type models ────────────────────────────────────────────────────────


class RunInfo(BaseModel):
    run_index: int = Field(description="Sequential run number shown in the Forgejo UI")
    workflow: str = Field(description="Workflow filename, e.g. 'submit.yaml'")
    status: str = Field(
        description="One of: success, failure, cancelled, waiting, running"
    )
    title: str = Field(description="Commit message that triggered the run")
    time: str = Field(description="Start time in 'YYYY-MM-DD HH:MM:SS UTC' format")
    url: str = Field(description="Full URL to the run in the Forgejo web UI")


class StepInfo(BaseModel):
    index: int = Field(description="Zero-based step index within the job")
    name: str = Field(description="Step name as defined in the workflow yaml")
    status: str = Field(description="One of: success, failure, skipped, cancelled")
    duration: str = Field(description="Human-readable duration, e.g. '1s', '2m30s'")


class JobInfo(BaseModel):
    id: int = Field(description="Internal Forgejo job ID")
    name: str = Field(description="Job name as defined in the workflow yaml")
    status: str = Field(description="One of: success, failure, skipped, cancelled")
    duration: str = Field(description="Human-readable duration of the job")


class JobSummary(BaseModel):
    title: str = Field(description="Commit message that triggered the run")
    status: str = Field(description="Overall run status")
    jobs: list[JobInfo] = Field(description="List of jobs in this run")
    steps: list[StepInfo] = Field(description="Steps of the first (current) job")


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def list_repos() -> list[str]:
    """
    List all repositories accessible to the authenticated user.

    Returns a list of repository names in 'owner/repo' format.
    """
    return fg.list_repos(_user, _password)


@mcp.tool()
def list_workflows(
    owner_repo: Annotated[
        str,
        Field(
            description="Repository in 'owner/repo' format, e.g. 'team-tron/pipelines'"
        ),
    ],
) -> list[str]:
    """
    List all workflow files defined in a Forgejo repository.

    Workflows live under .forgejo/workflows/ in the repo. Returns their filenames
    (e.g. ['ci.yaml', 'submit.yaml']). Includes workflows that have never been run.
    """
    return fg.list_workflows(owner_repo, _user, _password)


@mcp.tool()
def get_runs(
    owner_repo: Annotated[
        str,
        Field(
            description="Repository in 'owner/repo' format, e.g. 'team-tron/pipelines'"
        ),
    ],
    workflow: Annotated[
        str, Field(description="Workflow filename to filter by, e.g. 'submit.yaml'")
    ],
) -> list[RunInfo]:
    """
    Return all runs for a specific workflow, ordered oldest to newest.

    Returns an empty list if the workflow exists but has never been triggered.
    Use run_index from the result to fetch jobs or logs for a specific run.
    """
    return [
        RunInfo(**fg.format_run(r))
        for r in fg.fetch_runs_for_workflow(owner_repo, workflow, _user, _password)
    ]


@mcp.tool()
def get_jobs(
    owner_repo: Annotated[
        str,
        Field(
            description="Repository in 'owner/repo' format, e.g. 'team-tron/pipelines'"
        ),
    ],
    run_index: Annotated[
        int, Field(description="Run number from get_runs (the run_index field), e.g. 2")
    ],
) -> JobSummary:
    """
    Return the job and step summary for a specific run without fetching full logs.

    Use this to quickly check which step failed before deciding whether to call
    get_logs. Steps are listed in execution order with their individual statuses.
    """
    state = fg.get_job_state(owner_repo, run_index, 0, _user, _password)
    run = state["state"]["run"]
    return JobSummary(
        title=run["title"],
        status=run["status"],
        jobs=[
            JobInfo(
                id=j["id"], name=j["name"], status=j["status"], duration=j["duration"]
            )
            for j in run.get("jobs", [])
        ],
        steps=[
            StepInfo(
                index=i, name=s["summary"], status=s["status"], duration=s["duration"]
            )
            for i, s in enumerate(state["state"]["currentJob"].get("steps", []))
        ],
    )


@mcp.tool()
def get_logs(
    owner_repo: Annotated[
        str,
        Field(
            description="Repository in 'owner/repo' format, e.g. 'team-tron/pipelines'"
        ),
    ],
    run_index: Annotated[
        int, Field(description="Run number from get_runs (the run_index field), e.g. 2")
    ],
    job_index: Annotated[
        int, Field(description="Zero-based job index within the run, almost always 0")
    ] = 0,
) -> str:
    """
    Return the full log output for every step of a run job as a single string.

    Steps are separated by header lines showing the step name and status.
    Skipped steps appear with an empty body. This can be long — use get_jobs
    first to identify which step failed, then use this only when you need
    the actual error output.
    """
    result = fg.get_step_logs(owner_repo, run_index, job_index, _user, _password)
    steps, step_map = result["steps"], result["logs"]
    if not steps:
        return "No steps found."
    out = StringIO()
    for i, step in enumerate(steps):
        out.write(f"\n{'='*60}\n")
        out.write(
            f"Step {i}: {step['summary']}  [{step['status']}]  {step['duration']}\n"
        )
        out.write(f"{'='*60}\n")
        step_log = step_map.get(i)
        for line in (step_log.get("lines", []) if step_log else []):
            out.write(f"  {line['message']}\n")
    return out.getvalue()


@mcp.tool()
def get_latest_logs(
    owner_repo: Annotated[
        str,
        Field(
            description="Repository in 'owner/repo' format, e.g. 'team-tron/pipelines'"
        ),
    ],
    workflow: Annotated[
        str, Field(description="Workflow filename, e.g. 'submit.yaml'")
    ],
) -> str:
    """
    Return the full logs for the most recent run of a workflow.

    Convenience shortcut for: get_runs → pick last → get_logs.
    Returns an error string if the workflow has never been run.
    """
    runs = fg.fetch_runs_for_workflow(owner_repo, workflow, _user, _password)
    if not runs:
        return f"No runs found for {workflow}"
    latest = runs[-1]
    run_index = latest["index_in_repo"]
    header = f"Latest run: #{run_index}  [{latest['status']}]  {latest['title']}\n\n"
    return header + get_logs(owner_repo, run_index)


@mcp.tool()
def wait_for_run(
    owner_repo: Annotated[
        str,
        Field(
            description="Repository in 'owner/repo' format, e.g. 'team-tron/pipelines'"
        ),
    ],
    workflow: Annotated[
        str, Field(description="Workflow filename to watch, e.g. 'submit.yaml'")
    ],
    commit_message: Annotated[
        str, Field(description="Exact commit message of the run to wait for")
    ],
    timeout_seconds: Annotated[
        int, Field(description="How long to wait before giving up (default: 30)")
    ] = 30,
) -> RunInfo:
    """
    Poll every 2 seconds until a run triggered by the given commit message appears.

    Useful after a git push to confirm the pipeline was picked up by the runner.
    Raises TimeoutError if no matching run appears within timeout_seconds.
    Returns the run info as soon as the run is registered — it may still be
    in-progress at that point. Follow up with get_jobs to track its status.
    """
    run = fg.wait_for_run(
        owner_repo, workflow, commit_message, _user, _password, timeout_seconds
    )
    return RunInfo(**fg.format_run(run))


if __name__ == "__main__":
    mcp.run(transport="stdio")
