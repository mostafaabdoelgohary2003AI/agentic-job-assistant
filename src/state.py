"""Shared state that flows through the agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class JobApplicationState(TypedDict, total=False):
    # inputs
    job_text: str
    cv_text: str
    auto_approve: bool
    # working artifacts
    job_info: dict[str, Any]
    cv_profile: dict[str, Any]
    match_report: dict[str, Any]
    cover_letter: str
    critique: dict[str, Any]
    revision_count: int
    # outcome
    status: str  # "drafting" | "awaiting_human_review" | "approved"
    # Annotated reducer: LangGraph concatenates each node's list instead of
    # overwriting it, so `history` accumulates the full execution path.
    history: Annotated[list[str], operator.add]
