"""Framework-free runner that executes the same flow as the LangGraph graph.

This file exists on purpose: it shows exactly what the graph in graph.py
does underneath (a loop over node functions with one conditional edge),
so the framework never becomes magic. Use graph.py in production; read
this file to understand it.
"""

from __future__ import annotations

from .agents import MAX_REVISIONS, make_nodes, route_after_critique
from .state import JobApplicationState


def run(job_text: str, cv_text: str, auto_approve: bool = False, llm=None) -> JobApplicationState:
    nodes = make_nodes(llm)
    state: JobApplicationState = {
        "job_text": job_text,
        "cv_text": cv_text,
        "auto_approve": auto_approve,
        "revision_count": 0,
        "history": [],
    }

    def apply(update: dict) -> None:
        step = update.pop("history", [])
        state.update(update)
        state["history"].extend(step)

    apply(nodes["analyze_job"](state))
    apply(nodes["profile_candidate"](state))
    apply(nodes["match"](state))
    while True:
        apply(nodes["draft"](state))
        apply(nodes["critique"](state))
        if route_after_critique(state) == "review" or state["revision_count"] >= MAX_REVISIONS:
            break
    apply(nodes["human_review"](state))
    return state
