"""
Forgejo HTTP client — credential loading, REST API, and web UI calls.

Also usable as a CLI without the MCP server running:

  python -m mcp_servers.forgejo.client <owner/repo> list workflows
  python -m mcp_servers.forgejo.client <owner/repo> get runs <workflow.yaml>
  python -m mcp_servers.forgejo.client <owner/repo> get jobs <run_index>
  python -m mcp_servers.forgejo.client <owner/repo> get logs <run_index> [job_index]
  python -m mcp_servers.forgejo.client <owner/repo> get latest logs <workflow.yaml>
  python -m mcp_servers.forgejo.client <owner/repo> wait run <workflow.yaml> <commit message> [timeout_seconds]

Examples:
  python -m mcp_servers.forgejo.client ml-platform/ml-components list workflows
  python -m mcp_servers.forgejo.client ml-platform/ml-components get runs ci.yaml
  python -m mcp_servers.forgejo.client ml-platform/ml-components get jobs 2
  python -m mcp_servers.forgejo.client ml-platform/ml-components get logs 2
  python -m mcp_servers.forgejo.client ml-platform/ml-components get latest logs ci.yaml
  python -m mcp_servers.forgejo.client ml-platform/ml-components wait run ci.yaml "my commit" 30
"""

import re
import sys
import json
import base64
import time
import urllib.request
import urllib.error
from datetime import datetime

FORGEJO_URL = "http://localhost:4000"
SECRET_FILE = "k8s_yamls/forgejo/02-secret.yaml"


# ── HTTP primitives ───────────────────────────────────────────────────────────


def get_credentials() -> tuple[str, str]:
    """Read admin username and password from the k8s secret YAML."""
    with open(SECRET_FILE) as f:
        content = f.read()
    user = re.search(r"admin-username:\s*\"?([^\"\n]+)\"?", content).group(1).strip()
    password = (
        re.search(r"admin-password:\s*\"?([^\"\n]+)\"?", content).group(1).strip()
    )
    return user, password


def _auth_header(user: str, password: str) -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def api_get(path: str, user: str, password: str) -> dict | list:
    """GET against the Forgejo REST API (v1); returns parsed JSON."""
    url = f"{FORGEJO_URL}/api/v1/{path}"
    req = urllib.request.Request(url, headers=_auth_header(user, password))
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def web_post(path: str, user: str, password: str, payload: dict | None = None) -> dict:
    """POST against Forgejo's internal web UI endpoints, which expose data not available via the REST API."""
    url = f"{FORGEJO_URL}/{path}"
    data = json.dumps(payload or {}).encode()
    headers = {**_auth_header(user, password), "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# ── Domain functions ──────────────────────────────────────────────────────────


def list_repos(user: str, password: str) -> list[str]:
    """Return a list of all repositories accessible to the authenticated user."""
    repos = api_get("user/repos?type=admin", user, password)
    return [f"{r['owner']['login']}/{r['name']}" for r in repos]


def list_workflows(owner_repo: str, user: str, password: str) -> list[str]:
    """Return sorted filenames of all workflow files under .forgejo/workflows/."""
    owner, repo = owner_repo.split("/", 1)
    files = api_get(f"repos/{owner}/{repo}/contents/.forgejo/workflows", user, password)
    return sorted(f["name"] for f in files if f["type"] == "file")


def fetch_runs(owner_repo: str, user: str, password: str) -> list[dict]:
    """Return all action runs for a repo across all workflows, oldest first."""
    owner, repo = owner_repo.split("/", 1)
    page, runs = 1, []
    while True:
        data = api_get(
            f"repos/{owner}/{repo}/actions/runs?limit=50&page={page}", user, password
        )
        batch = data.get("workflow_runs", [])
        runs.extend(batch)
        if len(batch) < 50:
            break
        page += 1
    return runs


def fetch_runs_for_workflow(
    owner_repo: str, workflow: str, user: str, password: str
) -> list[dict]:
    """Return all runs for a specific workflow filename, oldest first."""
    return [
        r
        for r in fetch_runs(owner_repo, user, password)
        if r["workflow_id"] == workflow
    ]


def get_job_state(
    owner_repo: str, run_index: int, job_index: int, user: str, password: str
) -> dict:
    """Return the full state dict for a job, including run metadata and step list."""
    owner, repo = owner_repo.split("/", 1)
    path = f"{owner}/{repo}/actions/runs/{run_index}/jobs/{job_index}/attempt/1"
    return web_post(path, user, password)


def get_step_logs(
    owner_repo: str, run_index: int, job_index: int, user: str, password: str
) -> dict:
    """Return steps and per-step log lines for a job as {"steps": [...], "logs": {step_index: ...}}."""
    owner, repo = owner_repo.split("/", 1)
    path = f"{owner}/{repo}/actions/runs/{run_index}/jobs/{job_index}/attempt/1"
    state = web_post(path, user, password)
    steps = state["state"]["currentJob"].get("steps", [])
    if not steps:
        return {"steps": [], "logs": {}}
    cursors = [{"step": i, "cursor": 0, "expanded": True} for i in range(len(steps))]
    log_data = web_post(path, user, password, {"logCursors": cursors})
    step_map = {s["step"]: s for s in log_data["logs"].get("stepsLog", [])}
    return {"steps": steps, "logs": step_map}


def wait_for_run(
    owner_repo: str,
    workflow: str,
    commit_message: str,
    user: str,
    password: str,
    timeout: int = 30,
) -> dict:
    """Poll until a run matching commit_message appears; returns as soon as registered, not when complete."""
    deadline = time.monotonic() + timeout
    while True:
        for run in fetch_runs_for_workflow(owner_repo, workflow, user, password):
            if run["title"] == commit_message:
                return run
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"No '{workflow}' run with message \"{commit_message}\" appeared within {timeout}s"
            )
        time.sleep(min(2, remaining))


def format_run(run: dict) -> dict:
    """Normalize a raw API run dict into the flat structure used by RunInfo and the CLI."""
    started = run.get("started", "")
    if started and started != "0001-01-01T00:00:00Z":
        ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
    else:
        ts = datetime.fromisoformat(run["created"].replace("Z", "+00:00"))
    return {
        "run_index": run["index_in_repo"],
        "workflow": run["workflow_id"],
        "status": run["status"],
        "title": run["title"],
        "time": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "url": run["html_url"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_run(run: dict) -> None:
    r = format_run(run)
    print(
        f"run #{r['run_index']:>3}  [{r['status']:<10}]  {r['time']}  {r['title'][:60]}"
    )


def _cmd_list_repos(user, password):
    for repo in list_repos(user, password):
        print(repo)


def _cmd_list_workflows(owner_repo, user, password):
    for w in list_workflows(owner_repo, user, password):
        print(w)


def _cmd_get_runs(owner_repo, workflow, user, password):
    runs = fetch_runs_for_workflow(owner_repo, workflow, user, password)
    if not runs:
        print(f"No runs found for {workflow}")
        return
    for run in runs:
        _print_run(run)


def _cmd_get_jobs(owner_repo, run_index, user, password):
    state = get_job_state(owner_repo, run_index, 0, user, password)
    run = state["state"]["run"]
    print(f"Run #{run_index}: {run['title']}")
    print(f"Status: {run['status']}")
    print()
    for job in run.get("jobs", []):
        print(
            f"  Job [{job['id']}] {job['name']}  status={job['status']}  duration={job['duration']}"
        )
    print()
    for i, step in enumerate(state["state"]["currentJob"].get("steps", [])):
        print(
            f"  step {i}: {step['summary']:<40}  {step['status']:<10}  {step['duration']}"
        )


def _cmd_get_logs(owner_repo, run_index, job_index, user, password):
    result = get_step_logs(owner_repo, run_index, job_index, user, password)
    steps, step_map = result["steps"], result["logs"]
    if not steps:
        print("No steps found.")
        return
    for i, step in enumerate(steps):
        print(f"\n{'='*60}")
        print(f"Step {i}: {step['summary']}  [{step['status']}]  {step['duration']}")
        print(f"{'='*60}")
        step_log = step_map.get(i)
        for line in (step_log.get("lines", []) if step_log else []):
            print(f"  {line['message']}")


def _cmd_latest_logs(owner_repo, workflow, user, password):
    runs = fetch_runs_for_workflow(owner_repo, workflow, user, password)
    if not runs:
        print(f"No runs found for {workflow}")
        return
    latest = runs[-1]
    run_index = latest["index_in_repo"]
    print(f"Latest run: #{run_index}  [{latest['status']}]  {latest['title']}\n")
    _cmd_get_logs(owner_repo, run_index, 0, user, password)


def _cmd_wait_run(owner_repo, workflow, commit_message, timeout, user, password):
    print(
        f"Waiting up to {timeout}s for a '{workflow}' run with message: \"{commit_message}\"",
        flush=True,
    )
    run = wait_for_run(owner_repo, workflow, commit_message, user, password, timeout)
    r = format_run(run)
    print(f"Found: run #{r['run_index']}  [{r['status']}]  {r['title']}")


def _usage():
    print("Usage:")
    print("  python -m mcp_servers.forgejo.client list repos")
    print("  python -m mcp_servers.forgejo.client <owner/repo> <command>")
    print()
    print("Commands (requires <owner/repo>):")
    print("  list workflows")
    print("  get runs <workflow.yaml>")
    print("  get jobs <run_index>")
    print("  get logs <run_index> [job_index]")
    print("  get latest logs <workflow.yaml>")
    print("  wait run <workflow.yaml> <commit message> [timeout_seconds]")
    sys.exit(1)


def main():
    user, password = get_credentials()

    if len(sys.argv) < 3:
        _usage()

    command = sys.argv[1:]

    try:
        if command == ["list", "repos"]:
            _cmd_list_repos(user, password)
            return

        if len(sys.argv) < 4:
            _usage()

        owner_repo = sys.argv[2]
        command = sys.argv[3:]

        if command == ["list", "workflows"]:
            _cmd_list_workflows(owner_repo, user, password)

        elif len(command) == 3 and command[:2] == ["get", "runs"]:
            _cmd_get_runs(owner_repo, command[2], user, password)

        elif len(command) == 3 and command[:2] == ["get", "jobs"]:
            _cmd_get_jobs(owner_repo, int(command[2]), user, password)

        elif len(command) >= 3 and command[:2] == ["get", "logs"]:
            _cmd_get_logs(
                owner_repo,
                int(command[2]),
                int(command[3]) if len(command) > 3 else 0,
                user,
                password,
            )

        elif len(command) == 4 and command[:3] == ["get", "latest", "logs"]:
            _cmd_latest_logs(owner_repo, command[3], user, password)

        elif len(command) >= 4 and command[:2] == ["wait", "run"]:
            timeout = int(command[4]) if len(command) > 4 else 30
            _cmd_wait_run(owner_repo, command[2], command[3], timeout, user, password)

        else:
            _usage()

    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Secret file not found: {SECRET_FILE}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
