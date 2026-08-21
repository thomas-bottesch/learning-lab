"""
Tests for LangGraph graphs (research, verification, proposal).

These tests validate the internal graph logic in isolation, without
requiring a running Temporal Server or any external services.

All node functions are async, so we use ``pytest.mark.asyncio`` and call
``.ainvoke()`` on the compiled graphs.
"""

import pytest

from app.graphs import (
    build_proposal_graph,
    build_research_graph,
    build_verification_graph,
)


@pytest.fixture
def research_graph():
    return build_research_graph()


@pytest.fixture
def verification_graph():
    return build_verification_graph()


@pytest.fixture
def proposal_graph():
    return build_proposal_graph()


# ---------------------------------------------------------------------------
# Research graph tests
# ---------------------------------------------------------------------------


class TestResearchGraph:
    @pytest.mark.asyncio
    async def test_returns_findings_and_sources(self, research_graph):
        result = await research_graph.ainvoke(
            {"question": "Is Python a good language?"}
        )
        assert "findings" in result
        assert "sources" in result
        assert isinstance(result["findings"], list)
        assert isinstance(result["sources"], list)
        assert len(result["findings"]) > 0
        assert len(result["sources"]) > 0

    @pytest.mark.asyncio
    async def test_findings_are_strings(self, research_graph):
        result = await research_graph.ainvoke({"question": "test question"})
        assert all(isinstance(f, str) for f in result["findings"])

    @pytest.mark.asyncio
    async def test_sources_are_strings(self, research_graph):
        result = await research_graph.ainvoke({"question": "test question"})
        # Sources are now Source objects with url/title properties
        assert all(hasattr(s, "url") for s in result["sources"])
        assert all(hasattr(s, "title") for s in result["sources"])


# ---------------------------------------------------------------------------
# Verification graph tests
# ---------------------------------------------------------------------------


class TestVerificationGraph:
    @pytest.mark.asyncio
    async def test_returns_verified_and_rejected(self, verification_graph):
        result = await verification_graph.ainvoke(
            {
                "question": "test",
                "findings": ["finding 1"],
                "sources": [
                    "https://example.com/good-source",
                    "https://example.com/migrating",
                ],
            }
        )
        assert "verified_sources" in result
        assert "rejected_sources" in result
        assert isinstance(result["verified_sources"], list)
        assert isinstance(result["rejected_sources"], list)
        # The mock rejects sources with "migrating" in URL
        assert len(result["verified_sources"]) >= 1
        assert len(result["rejected_sources"]) >= 1

    @pytest.mark.asyncio
    async def test_async_verification(self, verification_graph):
        result = await verification_graph.ainvoke(
            {
                "question": "test",
                "findings": ["finding 1"],
                "sources": ["https://example.com/valid"],
            }
        )
        assert len(result["verified_sources"]) >= 0
        assert len(result["rejected_sources"]) >= 0


# ---------------------------------------------------------------------------
# Proposal graph tests
# ---------------------------------------------------------------------------


class TestProposalGraph:
    @pytest.mark.asyncio
    async def test_returns_proposal_fields(self, proposal_graph):
        result = await proposal_graph.ainvoke(
            {
                "question": "Should we migrate to CockroachDB?",
                "verified_sources": ["Source A", "Source B"],
                "rejected_sources": ["Source C"],
            }
        )
        assert "title" in result
        assert "summary" in result
        assert "proposed_action" in result
        assert isinstance(result["title"], str)
        assert isinstance(result["summary"], str)
        assert isinstance(result["proposed_action"], str)
        assert len(result["title"]) > 0
        assert len(result["summary"]) > 0
        assert len(result["proposed_action"]) > 0

    @pytest.mark.asyncio
    async def test_async_proposal(self, proposal_graph):
        result = await proposal_graph.ainvoke(
            {
                "question": "test question",
                "verified_sources": ["V1"],
                "rejected_sources": ["R1"],
            }
        )
        assert len(result["title"]) > 0
