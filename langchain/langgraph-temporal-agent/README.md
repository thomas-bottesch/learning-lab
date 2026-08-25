# LangGraph + Temporal Durable Agent

A production-oriented long-running AI agent demonstrating **Temporal** for durable workflow orchestration combined with **LangGraph** for bounded agent reasoning.

## Architecture

```mermaid
graph TB
    subgraph "Temporal Cluster"
        TW["Temporal Workflow"]
        WA["Worker: research"]
        VW["Worker: verify_sources"]
        PW["Worker: generate_proposal"]
        HA["Human Approval Signal"]
        EA["Worker: execute_action"]
        NA["Worker: notify_user"]
    end

    subgraph "LangGraph (inside Activities)"
        RG["Research Graph"]
        VG["Verification Graph"]
        PG["Proposal Graph"]
    end

    subgraph "Observability Layer"
        LT["Langfuse v4 Trace"]
        WS["Workflow Span"]
        SA1["research Activity Span"]
        SA2["verify Activity Span"]
        SA3["proposal Activity Span"]
        SA4["execute Action Span"]
        SA5["notify User Span"]
    end

    subgraph "External Systems"
        LLM["LLM / Search API"]
        DB["External Storage / DB"]
        NOTIFY["Email / Webhook / Slack"]
    end

    TW -->|Activity| WA
    WA -->|invoke| RG
    RG -->|mock| LLM

    TW -->|Activity| VW
    VW -->|invoke| VG
    VG -->|mock| LLM

    TW -->|Activity| PW
    PW -->|invoke| PG
    PG -->|mock| LLM

    TW -->|Signal| HA
    HA -->|condition| TW

    TW -->|Activity| EA
    EA -->|side effect| DB

    TW -->|Activity| NA
    NA -->|send| NOTIFY

    %% Observability connections
    TW -.- wfTrace .-> LT
    LT --> WS
    WA -.- obsAct .-> SA1
    VW -.- obsAct .-> SA2
    PW -.- obsAct .-> SA3
    EA -.- obsAct .-> SA4
    NA -.- obsAct .-> SA5
    SA1 --> LT
    SA2 --> LT
    SA3 --> LT
    SA4 --> LT
    SA5 --> LT
    WS --> LT
```

## Why This Architecture?

### Temporal is the Outer Workflow Engine

Temporal owns the **durable state machine** that orchestrates the entire research-and-approval lifecycle. Key benefits:

- **Crash resilience**: If the Python worker process crashes, Temporal retains the workflow state. When a worker restarts, Temporal replays the workflow history and resumes from the last completed Activity.
- **Durable waits**: The human approval step can wait indefinitely without keeping a Python process busy. Temporal stores the "waiting" state in its persistence layer.
- **Automatic retries**: Temporal retries failed Activities with configurable backoff policies — no manual retry loops needed.
- **Horizontal scaling**: Multiple workers can share the same task queue. Temporal distributes tasks across them.

### LangGraph is Used Inside Activities

LangGraph owns the **bounded agent reasoning** inside specific phases:

| Activity          | LangGraph Purpose                    |
| ----------------- | ------------------------------------ |
| `research`        | Search → Summarise pipeline          |
| `verify_sources`  | Source credibility verification      |
| `generate_proposal` | Proposal generation with LLM       |

LangGraph is **not** a durable workflow engine. It manages graph state during a single execution. By placing LangGraph inside Activities, we get:

- **Bounded scope**: Each graph has a clear input/output contract.
- **No persistence conflicts**: Temporal owns the long-lived state; LangGraph owns the short-lived graph state.
- **Testability**: LangGraph graphs can be tested independently of Temporal.

### Why External I/O Belongs in Activities

**Never** perform external I/O (LLM calls, HTTP requests, database writes, file reads) directly inside Workflow code. Workflow code is **deterministic** — it is replayed every time a worker loads the workflow state. If Workflow code made an HTTP request on every replay, you'd get:

- Duplicate API calls
- Non-deterministic results (different LLM responses on each replay)
- Corrupted state

Activities are **not** replayed — they execute once, and Temporal records their result. All external I/O belongs in Activities.

## Workflow Execution Flow

```
Temporal Workflow
│
├── 1. research Activity
│       └── Research LangGraph (search → summarize)
│
├── 2. verify_sources Activity
│       └── Verification LangGraph (verify_sources)
│
├── 3. generate_proposal Activity
│       └── Proposal LangGraph (generate_proposal)
│
├── 4. WAIT FOR HUMAN APPROVAL
│       └── Temporal Signal (durable — no Python process needed)
│
├── 5. execute_action Activity
│       └── Idempotent side effect
│
└── 6. notify_user Activity
        └── External notification
```

## Retry Policies

Temporal retries failed Activities automatically. Different Activities have different retry characteristics based on their nature:

| Activity             | Max Attempts | Initial Backoff | Max Backoff | Rationale                         |
| -------------------- | ------------ | --------------- | ----------- | --------------------------------- |
| `research`           | 5            | 2 s             | 2 min       | Transient network/LLM failures    |
| `verify_sources`     | 5            | 2 s             | 2 min       | Same as research                  |
| `generate_proposal`  | 3            | 2 s             | 2 min       | Fewer retries to limit cost       |
| `execute_action`     | Default      | 1 s             | ∞           | Conservative — side effect        |
| `notify_user`        | Default      | 1 s             | ∞           | Notification should eventually deliver |

**Do not** implement manual retry loops inside Workflow code. Use Temporal's `RetryPolicy`.

## Human Approval

Human approval uses a **Temporal Signal**:

1. The workflow reaches `wait_condition` and pauses durably.
2. A client sends a signal (`approve` or `reject`) to the workflow handle.
3. Temporal delivers the signal to the workflow when a worker replays it.
4. The workflow resumes execution based on the signal value.

```python
@workflow.signal
async def approve(self, approved: bool):
    self.approval_received = True
    self.approved = approved

# In the workflow:
await workflow.wait_condition(
    lambda: self.approval_received is not None,
)
```

**Key point**: The workflow can wait for days, months, or years. No Python process is kept alive. Temporal stores the waiting state in its database.

## Worker Failure

When the Python worker process crashes:

1. Temporal detects that the Activity heartbeat has stopped.
2. If the Activity had a `start_to_close_timeout`, Temporal marks it as failed after the timeout expires.
3. Temporal retries the Activity (up to the configured `maximum_attempts`).
4. When a new worker starts (or the crashed one recovers), Temporal assigns the retried Activity to an available worker.
5. The workflow replays its history to the point of failure, then executes the retried Activity.

**Important**: Workflow code is deterministic and replayed. Activity code executes once (unless retried by Temporal).

## Multiple Workers

Multiple instances of `python -m app.worker` can run simultaneously:

```bash
# Terminal 1
python -m app.worker

# Terminal 2
python -m app.worker

# Terminal 3
python -m app.worker
```

All connect to the same Temporal Server and listen on the same task queue (`research-agent`). Temporal distributes tasks across workers automatically.

## DevContainer

This project runs inside a **DevContainer**. All dependencies and services are installed within the container environment. To work on this project:

1. Open the project in VS Code with the Dev Containers extension installed.
2. The container will automatically build using the configuration in `.devcontainer/`.

## Prerequisites

Before opening the DevContainer, ensure the following are set up on the **host machine**:

1. **Kubernetes Cluster** — Install k3s on the host (Docker must be running):
   ```bash
   python host_config/k8s_cluster.py --start
   ```
   This installs a fresh k3s cluster bound to `172.17.0.1` (the Docker bridge IP).

2. **Docker** — Required for both k3s and running Langfuse/Temporal as containers.

### Installing Services Inside the DevContainer

Once inside the DevContainer, run the install scripts:

```bash
# Install Local Temporal Server
./install_temporal.sh install

# Install Local Langfuse Server
./install_langfuse.sh install
```

These scripts configure and start the respective services using the k3s cluster on the host.

## Installation (Package)

```bash
cd langgraph-temporal-agent
pip install -e .
```

```bash
cd langgraph-temporal-agent
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:

```env
TEMPORAL_ADDRESS=temporal.example.internal:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=research-agent

# Optional future LLM integration
OPENAI_API_KEY=
```

### Environment Variables

| Variable             | Description                          | Default               |
| -------------------- | ------------------------------------ | --------------------- |
| `TEMPORAL_ADDRESS`   | Temporal Server host:port            | `localhost:7233`      |
| `TEMPORAL_NAMESPACE` | Temporal namespace                   | `default`             |
| `TEMPORAL_TASK_QUEUE`| Task queue name                      | `research-agent`      |
| `OPENAI_API_KEY`     | OpenAI API key (optional)            | _(empty)_             |

## Usage

### Start the Worker

```bash
export TEMPORAL_ADDRESS=172.17.0.1:7233
export TEMPORAL_NAMESPACE=default
export TEMPORAL_TASK_QUEUE=research-agent

python -m app.worker
```

### Start a Workflow

```bash
python -m app.client start \
    "Should our company migrate from PostgreSQL to CockroachDB?"
```

Output:

```
Workflow started:
  workflow_id: research-abc12345
  run_id:      def67890
```

### Approve a Workflow

```bash
python -m app.client approve research-abc12345
```

### Reject a Workflow

```bash
python -m app.client reject research-abc12345
```

### Check Workflow Status

```bash
python -m app.client status research-abc12345
```

## Running Tests

### LangGraph Tests

Tests the three LangGraph graphs in isolation:

```bash
pytest tests/test_graphs.py -v
```

### Temporal Workflow Tests

Uses Temporal's `TestingEnvironment` (no real Temporal Server required):

```bash
pytest tests/test_workflow.py -v
```

### All Tests

```bash
pytest -v
```

## Large Data Handling

**Do not pass large documents through Temporal.** Temporal history has size limits, and storing large artifacts in workflow state is inefficient.

Instead, use external object storage:

```
Temporal Workflow
        │
        └── document_id (e.g., "s3://bucket/research/abc123")
                │
                ▼
        Object Storage (S3 / GCS / Azure Blob)
```

The models in [`app/domain/models.py`](app/domain/models.py) include an `external_doc_ids` field for this purpose.

## LangGraph Persistence

**Do not add LangGraph checkpointing for the entire long-running workflow.** Temporal is the source of truth for the workflow lifecycle between Activities. Use:

| System            | Purpose                      |
| ----------------- | ---------------------------- |
| Temporal          | Durable workflow state       |
| LangGraph         | Bounded agent execution      |
| External DB/Storage | Large application artifacts |

Redundant persistence systems that all attempt to own the same state create confusion and inconsistency.

## Idempotency

External side-effecting Activities can be retried by Temporal. The `execute_action` Activity demonstrates correct idempotency handling:

```
Activity executes external action
        ↓
external action succeeds
        ↓
worker crashes before Temporal receives completion
        ↓
Temporal retries Activity
        ↓
same idempotency key
        ↓
external system returns existing result
```

The idempotency key is derived from the Workflow ID:

```python
idempotency_key = f"{workflow_id}:execute-action"
```

**Temporal does NOT make arbitrary external APIs exactly-once.** Idempotency keys are the developer's responsibility.

## Project Structure

```
langgraph-temporal-agent/
│
├── app/
│   ├── __init__.py               # Package marker
│   ├── domain/                   # Domain layer
│   │   └── models.py             # Pydantic data models
│   ├── workflows/                # Temporal Workflows
│   │   ├── __init__.py
│   │   └── research.py           # ResearchWorkflow definition
│   ├── activities/               # Temporal Activities
│   │   ├── __init__.py
│   │   ├── research.py           # research activity
│   │   ├── verification.py       # verify_sources activity
│   │   ├── proposal.py           # generate_proposal activity
│   │   └── side_effects.py       # execute_action, notify_user
│   ├── graphs/                   # LangGraph graphs (inside Activities)
│   │   ├── __init__.py
│   │   ├── research.py
│   │   ├── verification.py
│   │   └── proposal.py
│   ├── infrastructure/           # Infrastructure layer
│   │   ├── __init__.py
│   │   ├── llm.py                # LLM client abstraction
│   │   └── observability/        # Observability layer (Langfuse v4)
│   │       ├── __init__.py
│   │       ├── client.py         # Client singleton + NoOp fallback
│   │       ├── config.py         # Config loader (.langfuse.env)
│   │       ├── tracing.py        # observe_activity, workflow_trace
│   │       ├── context.py        # Observation context helpers
│   │       ├── metadata.py       # Execution ID helpers
│   │       ├── production.py     # Production readiness checks
│   │       ├── redaction.py      # Input sanitization
│   │       ├── callbacks.py      # LangGraph callback integration
│   │       ├── graphs.py         # Graph observation wrappers
│   │       ├── llm_tracing.py    # LLM-specific tracing helpers
│   │       └── evaluations.py    # LLM evaluation hooks
│   ├── client.py                 # CLI client
│   └── worker.py                 # Worker entry point
│
├── tests/
│   ├── conftest.py               # Pytest fixtures (_setup_test_env, _reset_langfuse)
│   ├── test_graphs.py            # LangGraph unit tests
│   ├── test_workflow.py          # Temporal workflow tests
│   └── test_langfuse_standalone.py  # Langfuse v4 integration tests
│
├── .langfuse.env                 # Langfuse credentials (generated by install_langfuse.sh)
├── .env.example                  # Environment template
├── pyproject.toml                # Project dependencies
├── install_langfuse.sh           # Local Langfuse server installer
├── install_temporal.sh           # Local Temporal server installer
└── README.md                     # This file
```

## Local Development Cluster (k3s)

This project uses **k3s** (lightweight Kubernetes) as the local development cluster. The `host_config/k8s_cluster.py` script manages the cluster lifecycle on the host machine:

```bash
# Start the k3s cluster on the host (run once)
python host_config/k8s_cluster.py --start

# Stop the cluster when done
python host_config/k8s_cluster.py --stop
```

The cluster binds to the Docker bridge IP (`172.17.0.1`) so that services installed inside the DevContainer are reachable via the host network.

### Deploying Workers to k3s

For production or staging deployments, workers can be deployed to any Kubernetes cluster:

1. Build a container image:
   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip install --no-cache-dir -e .
   CMD ["python", "-m", "app.worker"]
   ```

2. Create a Kubernetes Deployment:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: research-agent
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: research-agent
     template:
       metadata:
         labels:
           app: research-agent
       spec:
         containers:
         - name: worker
           image: research-agent:latest
           envFrom:
           - secretRef:
               name: temporal-credentials
   ```

3. Store Temporal connection details in a Kubernetes Secret.

Both Langfuse and Temporal Servers are managed as part of the dev environment via `install_langfuse.sh` and `install_temporal.sh`.

## Observability — Langfuse v4 Integration

The project includes a production-grade observability layer using **Langfuse v4** (OpenTelemetry-based). Tracing is configured via `.langfuse.env` and activated with `LANGFUSE_TEST_TRACING=1`.

### Architecture

```
Trace ID = create_trace_id(seed=workflow_id)
│
├─ Root Span: "research-workflow" (workflow_trace context manager)
│   │
│   ├─ Span: "research-activity" (activity.research → observe_activity)
│   │   └─ LLM generations from LangGraph callbacks
│   │
│   ├─ Span: "verify_sources-activity" (activity.verify_sources)
│   │   └─ LLM generations from LangGraph callbacks
│   │
│   ├─ Span: "generate_proposal-activity" (activity.generate_proposal)
│   │   └─ LLM generations from LangGraph callbacks
│   │
│   ├─ Span: "execute_action-activity" (activity.execute_action)
│   │
│   └─ Span: "notify_user-activity" (activity.notify_user)
```

### How It Works

**Workflow creates root trace** ([`app/workflows/research.py`](app/workflows/research.py)):
```python
with workflow_trace(workflow_id=wf_id):
    # Registers trace_id in global map + creates root span
    ...
```

**Activities join the trace** ([`app/infrastructure/observability/tracing.py`](app/infrastructure/observability/tracing.py)):
```python
with observe_activity(
    name="research",
    input_data=safe_input,
    session_id=_wf_id,  # Bridges to workflow trace via workflow_id lookup
) as obs:
    obs.set_output({"findings_count": len(findings)})
```

### Cross-Thread Propagation

Temporal workers execute activities in separate threads. The observability layer uses a **global workflow→trace_id map** to propagate trace IDs across thread boundaries:

1. `workflow_trace()` calls `register_workflow_trace(wf_id, trace_id)` before executing activities
2. `observe_activity()` calls `resolve_trace_id(activity.workflow_id)` to find the active trace
3. Thread-local state handles intra-thread nesting (within same activity)

### Configuration

Copy `.langfuse.env` (generated by `install_langfuse.sh`) or set environment variables:

```env
LANGFUSE_PUBLIC_KEY=lf_pk_xxx
LANGFUSE_SECRET_KEY=lf_sk_xxx
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_TEST_TRACING=1  # Enable tracing in tests
```

### Running Tests with Tracing

```bash
# Without tracing (default)
pytest tests/test_workflow.py -v

# With Langfuse tracing enabled
LANGFUSE_TEST_TRACING=1 pytest tests/test_workflow.py -v

# Standalone integration tests (requires Langfuse server)
LANGFUSE_TEST_TRACING=1 pytest tests/test_langfuse_standalone.py -v
```

### Trace URL

After running with tracing, verify traces at:
```
http://localhost:3000/project/local-project/traces/<trace_id>
```

### Module Structure

```
app/infrastructure/observability/
├── __init__.py               # Package marker
├── client.py                 # Client singleton + NoOp fallback
├── config.py                 # Config loader (.langfuse.env, env vars)
├── tracing.py                # observe_activity, workflow_trace
├── context.py                # Observation context helpers
├── metadata.py               # Execution ID helpers
├── production.py             # Production readiness checks
├── redaction.py              # Input sanitization
├── callbacks.py              # LangGraph callback integration
├── graphs.py                 # Graph observation wrappers
├── llm_tracing.py            # LLM-specific tracing helpers
└── evaluations.py            # LLM evaluation hooks
```

## Temporal Determinism — Critical Rules

**Workflow code MUST NOT:**
- Call an LLM
- Make HTTP requests
- Access a database
- Read arbitrary files
- Use random numbers directly
- Depend on current wall-clock time (use Temporal's `workflow.now()`)
- Access environment-dependent state
- Perform external I/O
- Call arbitrary nondeterministic Python code

**Workflow code SHOULD:**
- Call Activities (by name)
- Use Temporal signals, conditions, and timers
- Perform deterministic data transformations
- Log (using `workflow.logger`)

All external operations belong in **Activities**.
