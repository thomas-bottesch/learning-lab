"""
Infrastructure layer for external I/O operations.

This module contains all external I/O code — LLM calls, HTTP requests,
search APIs. These are **not** called directly from Workflow code;
instead, they are invoked inside Temporal Activities.

In production, replace the mock implementations with real APIs
(e.g. Tavily for search, OpenAI/Anthropic for LLM calls).
"""

from __future__ import annotations


async def search(question: str) -> list[dict[str, str]]:
    """
    Mock search function.

    TODO: Replace with a real async search API (Tavily, SerpAPI, etc.).
    """
    return [
        {
            "title": "CockroachDB vs PostgreSQL: A Comparison",
            "url": "https://example.com/cockroach-postgres-comparison",
            "snippet": "CockroachDB offers horizontal scalability and strong consistency, while PostgreSQL excels in feature richness and ecosystem maturity.",
        },
        {
            "title": "Migrating from PostgreSQL to Distributed SQL",
            "url": "https://example.com/migrating-postgres-distributed",
            "snippet": "Organisations migrating to CockroachDB often do so for global distribution and automatic sharding. However, migration complexity and tooling gaps should be considered.",
        },
        {
            "title": "PostgreSQL 17 Released with Performance Improvements",
            "url": "https://example.com/postgres-17",
            "snippet": "PostgreSQL 17 brings significant performance improvements, making it a strong candidate for many workloads previously requiring distributed databases.",
        },
    ]


async def llm_summarize(question: str, sources: list[dict]) -> list[str]:
    """
    Mock LLM summarisation.

    TODO: Replace with a real async LLM call (OpenAI, Anthropic, etc.).
    """
    return [
        f"Based on analysis of {len(sources)} sources regarding '{question}':",
        "CockroachDB provides automatic sharding and multi-region consistency "
        "but sacrifices some of the rich ecosystem and tooling that PostgreSQL offers.",
        "PostgreSQL 17 delivers significant performance improvements and may "
        "suffice for most workloads without the complexity of a distributed database.",
        "Migration from PostgreSQL to CockroachDB requires careful planning, "
        "as driver compatibility, migration tools, and operational expertise "
        "are non-trivial factors.",
    ]


async def llm_verify(sources: list[dict]) -> tuple[list[str], list[str]]:
    """
    Mock source verification.

    Returns (verified, rejected) source titles.

    TODO: Replace with a real async LLM verification call.
    """
    verified: list[str] = []
    rejected: list[str] = []
    for src in sources:
        # Simulate a simple heuristic: reject sources with "migrating" in URL
        if "migrating" in src.get("url", ""):
            rejected.append(src["title"])
        else:
            verified.append(src["title"])
    # If nothing was rejected, reject the least informative one
    if not rejected and sources:
        rejected.append(sources[-1]["title"])
    return verified, rejected


async def llm_generate_proposal(
    question: str,
    verified_sources: list[str],
    rejected_sources: list[str],
) -> dict[str, str]:
    """
    Mock proposal generation.

    TODO: Replace with a real async LLM call.
    """
    return {
        "title": f"Assessment: Migrating from PostgreSQL to CockroachDB",
        "summary": (
            f"After researching '{question}' and verifying {len(verified_sources)} "
            f"sources (rejecting {len(rejected_sources)}), the evidence suggests that "
            "a full migration to CockroachDB is not immediately warranted. "
            "PostgreSQL 17 offers strong performance, and a phased approach "
            "— such as read replicas or eventual migration — is recommended."
        ),
        "proposed_action": (
            "Set up a CockroachDB test cluster in staging, run benchmark "
            "workloads matching production patterns, and evaluate results "
            "before committing to a migration."
        ),
    }
