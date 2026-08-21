"""Temporal Activities for the research agent.

Each activity is defined in its own module for clean separation of concerns.
All activity modules import graph builders lazily (inside the function body)
to preserve the Temporal sandbox purity invariant.
"""

from app.activities.proposal import generate_proposal
from app.activities.research import research
from app.activities.side_effects import execute_action, notify_user
from app.activities.verification import verify_sources

__all__ = [
    "research",
    "verify_sources",
    "generate_proposal",
    "execute_action",
    "notify_user",
]
