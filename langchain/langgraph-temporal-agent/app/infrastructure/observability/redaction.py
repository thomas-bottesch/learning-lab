"""
Redaction utilities for safe observability logging.

This module provides ``sanitize()`` — a function that removes sensitive data
from inputs/outputs before they are sent to Langfuse.

Covers rule #12:
  - Introduce explicit redaction at the infrastructure layer
  - Don't log raw prompts/responses by accident
  - Distinguish API keys, passwords, PII from normal data

Usage:

    from app.infrastructure.observability.redaction import sanitize

    safe_input = sanitize(request.model_dump())
    safe_output = sanitize(result.model_dump())

    with observe_activity(name="research", input=safe_input):
        ...
"""

from __future__ import annotations

import re
from typing import Any

# Patterns that indicate sensitive fields
SENSITIVE_KEYS = frozenset(
    [
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "secret_key",
        "secret-key",
        "password",
        "passwd",
        "pwd",
        "token",
        "auth_token",
        "access_token",
        "refresh_token",
        "authorization",
        "authorization_header",
        "auth-header",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "private_key",
        "privatekey",
        "private-key",
        "ssn",
        "social_security",
        "credit_card",
        "cc_number",
        "card_number",
        "phone",
        "phone_number",
        "email",  # Uncomment if you want to redact emails (PII concern)
    ]
)

# Regex pattern for header-like keys that might contain auth tokens
AUTH_HEADER_PATTERN = re.compile(r"(authorization|auth|x-api-key)", re.IGNORECASE)

# Value patterns that suggest something is sensitive but truncated or masked
MASKED_VALUE_PATTERNS = [
    r"\[REDACTED\]",
    r"sk-[a-zA-Z0-9]{20,}",  # API keys like sk-lf-...
    r"pk-[a-zA-Z0-9]{20,}",
    r"Bearer [a-zA-Z0-9\._\-=]+",
    r"eyJ[a-zA-Z0-9\._\-=]+",  # JWT tokens
]


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests it contains sensitive data."""
    lower = key.lower().replace("-", "_").replace(" ", "_")
    return any(s in lower for s in SENSITIVE_KEYS) or AUTH_HEADER_PATTERN.search(key)


def _is_masked_value(value: Any) -> bool:
    """Check if a value matches known masked/sensitive patterns."""
    str_val = str(value)
    for pattern in MASKED_VALUE_PATTERNS:
        if re.search(pattern, str_val):
            return True
    return False


def _sanitize_value(key: str, value: Any) -> Any:
    """Sanitize a single value if it appears to be sensitive."""
    if _is_sensitive_key(key) or _is_masked_value(value):
        return "[REDACTED]"
    return value


def _sanitize_nested(obj: Any, depth: int = 0) -> Any:
    """Recursively sanitize a nested structure."""
    max_depth = 5  # Prevent infinite recursion on circular refs

    if depth > max_depth:
        return "<max_depth_exceeded>"

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            sanitized_v = _sanitize_nested(v, depth + 1)
            result[k] = _sanitize_value(k, sanitized_v)
        return result

    if isinstance(obj, (list, tuple)):
        return [_sanitize_nested(item, depth + 1) for item in obj]

    return obj


def sanitize(data: Any) -> Any:
    """
    Sanitize data for safe logging to observability backends.

    This removes or masks:
      - Fields with sensitive names (api_key, password, token, etc.)
      - Values matching known secret patterns
      - Auth headers and bearer tokens

    Parameters
    ----------
    data : Any
        The data to sanitize. Can be any JSON-serializable type.

    Returns
    -------
    Any
        The sanitized data with sensitive values replaced by "[REDACTED]".

    Examples
    --------
    >>> sanitize({"api_key": "sk-abc123", "question": "hello"})
    {"api_key": "[REDACTED]", "question": "hello"}
    """
    if data is None:
        return None

    # Handle Pydantic models and dataclasses
    if hasattr(data, "model_dump"):
        # Pydantic v2
        data = data.model_dump()
    elif hasattr(data, "dict"):
        # Pydantic v1
        data = data.dict()
    elif hasattr(data, "__dict__"):
        data = vars(data)

    if not isinstance(data, dict):
        return data

    return _sanitize_nested(data)
