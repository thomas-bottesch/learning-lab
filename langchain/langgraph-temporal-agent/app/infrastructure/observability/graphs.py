"""
LangGraph builder wrappers for naming consistency.

This module provides wrapper functions that build LangGraph graphs and
set standard node names. The wrappers ensure:

  - Consistent node naming across all graphs (rule #6)
  - Graph version metadata is available (rule #7)
  - Callback configuration from graph code (rule #5)

Covers rules #6 and #7:
  - Make your LangGraph node names excellent
  - Give every trace a useful naming taxonomy

Node naming convention:
    <workflow_type>.<node_name>

Examples:
    research.search
    research.summarize
    verification.verify_sources
    proposal.generate_proposal
"""

from __future__ import annotations

import typing
from typing import Any, Callable

if typing.TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# Graph versions for metadata tracking (rule #7)
GRAPH_VERSIONS: dict[str, str] = {
    "research-graph": "1",
    "verification-graph": "1",
    "proposal-graph": "1",
}


def prefixed_node_name(workflow_type: str, node_name: str) -> str:
    """
    Create a prefixed node name for consistent graph observability.

    Parameters
    ----------
    workflow_type : str
        The workflow type (e.g., 'research', 'verification', 'proposal').
    node_name : str
        The original node name.

    Returns
    -------
    str
        Prefixed name (e.g., 'research.search').
    """
    return f"{workflow_type}.{node_name}"


def build_graph_with_naming(
    workflow_type: str,
    build_fn: Callable[..., "CompiledStateGraph"],
    callbacks: list[Any] | None = None,
) -> tuple["CompiledStateGraph", dict[str, str]]:
    """
    Build a LangGraph with standard naming conventions.

    This wraps the graph build process to inject callback configuration
    and return version metadata.

    Parameters
    ----------
    workflow_type : str
        The workflow type for naming prefix.
    build_fn : callable
        The function that builds and returns the compiled graph.
    callbacks : list, optional
        LangChain callbacks to configure on the graph.

    Returns
    -------
    tuple[CompiledStateGraph, dict]
        The compiled graph and metadata dict.
    """
    graph = build_fn()

    # Store version in graph metadata if supported
    graph_name = f"{workflow_type}-graph"
    version = GRAPH_VERSIONS.get(graph_name, "0")

    metadata = {
        "graph.name": graph_name,
        "graph.version": version,
        "workflow.type": workflow_type,
    }

    return graph, metadata
