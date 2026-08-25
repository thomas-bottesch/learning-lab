"""
Observability configuration and application metadata.

This module provides:
  - Application version and git SHA (for trace correlation)
  - Environment classification (local / staging / production)
  - Sampling configuration
  - Whether tracing is enabled

PRODUCTION NOTE — Module Purity:

  This module MUST have zero side effects at import time that block execution.
  No blocking network requests or subprocess calls. In production, APP_VERSION
  and GIT_SHA are set via CI/CD environment variables. The _detect_* functions
  are provided only as a local-development convenience and are NOT called at
  module level.

LOCAL DEVELOPMENT:

  Credentials are automatically loaded from .langfuse.env in the project root.
  OS environment variables take precedence for production overrides.

Usage:

    from app.infrastructure.observability.config import (
        APP_VERSION,
        ENVIRONMENT,
        GIT_SHA,
        is_tracing_enabled,
        get_application_metadata,
    )
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Environment classification
# ---------------------------------------------------------------------------

ENVIRONMENT = os.environ.get("APP_ENV", "local")

LANGFUSE_TRACING_ENVIRONMENT = os.environ.get(
    "LANGFUSE_TRACING_ENVIRONMENT", ENVIRONMENT
)

# ---------------------------------------------------------------------------
# Application version & git SHA
# ---------------------------------------------------------------------------
# Set by CI/CD in production. Falls back to static defaults at import time.

APP_VERSION: str = os.environ.get("APP_VERSION", "0.0.0-dev")
GIT_SHA: str = os.environ.get("GIT_SHA", "unknown")


# ---------------------------------------------------------------------------
# Local development helpers (NOT called at module level)
# ---------------------------------------------------------------------------


def _detect_version_from_git() -> str:
    """Attempt to derive version from git tag. LOCAL DEV ONLY."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "0.0.0-dev"


def _detect_git_sha() -> str:
    """Attempt to derive git SHA from repository. LOCAL DEV ONLY."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:8]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Sampling rule (#26: production control)
# ---------------------------------------------------------------------------

TRACING_SAMPLE_RATE = float(os.environ.get("TRACING_SAMPLE_RATE", "1.0"))


# ---------------------------------------------------------------------------
# Tracing enabled flag (#23: make tracing optional)
# ---------------------------------------------------------------------------

# CRITICAL: These values are read ONCE at module import time because the
# Temporal workflow sandbox blocks os.environ access inside function bodies.
# Functions that check tracing must use these cached globals.

_LANGFUSE_SECRET_KEY: str = ""
_LANGFUSE_PUBLIC_KEY: str = ""
_LANGFUSE_BASE_URL: str = ""


def _load_langfuse_env():
    """Load Langfuse credentials from .langfuse.env if present.

    OS env vars take precedence over values from the file.
    This runs once at module import time — no runtime cost.
    """
    global _LANGFUSE_SECRET_KEY, _LANGFUSE_PUBLIC_KEY, _LANGFUSE_BASE_URL

    # Find project root (three levels up: observability -> infrastructure -> app -> project-root)
    project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    project_root = os.path.normpath(project_root)
    env_path = os.path.join(project_root, ".langfuse.env")

    file_defaults: dict[str, str] = {}

    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key in (
                    "LANGFUSE_SECRET_KEY",
                    "LANGFUSE_PUBLIC_KEY",
                    "LANGFUSE_BASE_URL",
                ):
                    file_defaults[key] = value

    # OS env vars override file defaults
    _LANGFUSE_SECRET_KEY = os.environ.get(
        "LANGFUSE_SECRET_KEY", file_defaults.get("LANGFUSE_SECRET_KEY", "")
    )
    _LANGFUSE_PUBLIC_KEY = os.environ.get(
        "LANGFUSE_PUBLIC_KEY", file_defaults.get("LANGFUSE_PUBLIC_KEY", "")
    )
    _LANGFUSE_BASE_URL = os.environ.get(
        "LANGFUSE_BASE_URL", file_defaults.get("LANGFUSE_BASE_URL", "")
    )


_load_langfuse_env()


def is_tracing_enabled() -> bool:
    """
    Return True if Langfuse tracing is configured and active.

    **Sandbox-safe**: reads cached module-level variables set at import time.
    Never accesses os.environ inside the function body.

    Production: ``LANGFUSE_SECRET_KEY`` set via OS env var.
    Local dev: automatically loaded from .langfuse.env at import time.
    """
    return bool(_LANGFUSE_SECRET_KEY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_application_metadata() -> dict[str, Any]:
    """Return a standard metadata dict for every trace."""
    return {
        "application.version": APP_VERSION,
        "git.sha": GIT_SHA,
        "environment": LANGFUSE_TRACING_ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def should_sample() -> bool:
    """Decide whether to sample this trace based on TRACING_SAMPLE_RATE."""
    if TRACING_SAMPLE_RATE >= 1.0:
        return True
    if TRACING_SAMPLE_RATE <= 0.0:
        return False
    second = datetime.now(timezone.utc).second
    return (second % 100) < int(TRACING_SAMPLE_RATE * 100)


def get_langfuse_client_kwargs() -> dict:
    """Return kwargs suitable for constructing a Langfuse client.

    Used by the client singleton factory. Reads cached globals (no os.environ).
    """
    kwargs: dict = {}
    if _LANGFUSE_PUBLIC_KEY:
        kwargs["public_key"] = _LANGFUSE_PUBLIC_KEY
    kwargs["secret_key"] = _LANGFUSE_SECRET_KEY
    if _LANGFUSE_BASE_URL:
        kwargs["base_url"] = _LANGFUSE_BASE_URL
    return kwargs
