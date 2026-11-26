"""Agent nodes. Each takes the shared state and returns a partial update.

With a real LLM configured, the drafting node asks it to polish the
deterministic first draft. With the MockLLM (default), everything stays
deterministic so the graph is fully testable offline.
"""

from __future__ import annotations

from . import tools
from .llm import MockLLM
from .state import JobApplicationState

MAX_REVISIONS = 2


def make_nodes(llm=None):
    llm = llm or MockLLM()

    def analyze_job(state: JobApplicationState) -> dict:
        return {"job_info": tools.parse_job_posting(state["job_text"]),
                "history": ["analyze_job"]}

    def profile_candidate(state: JobApplicationState) -> dict:
        return {"cv_profile": tools.profile_cv(state["cv_text"]),
                "history": ["profile_candidate"]}

    def match(state: JobApplicationState) -> dict:
        report = tools.compute_match(state["job_info"], state["cv_profile"])
        return {"match_report": report, "history": ["match"]}

    def draft(state: JobApplicationState) -> dict:
        feedback = state.get("critique", {}).get("issues") or None
        letter = tools.compose_cover_letter(
            state["job_info"], state["match_report"], state["cv_profile"], feedback=feedback
        )
        if not isinstance(llm, MockLLM):
            polished = llm.complete(
                "You polish cover letters. Keep every factual claim identical; "
                "improve only flow and tone. Never add claims or metrics.",
                letter,
            )
            if polished and not polished.startswith("[mock-llm"):
                letter = polished
        return {
            "cover_letter": letter,
            "revision_count": state.get("revision_count", 0) + (1 if feedback else 0),
            "history": ["draft" + ("(revision)" if feedback else "")],
        }

    def critique(state: JobApplicationState) -> dict:
        result = tools.critique_letter(state["cover_letter"], state["job_info"], state["match_report"])
        return {"critique": result, "history": ["critique"]}

    def human_review(state: JobApplicationState) -> dict:
        approved = bool(state.get("auto_approve"))
        return {
            "status": "approved" if approved else "awaiting_human_review",
            "history": ["human_review"],
        }

    return {
        "analyze_job": analyze_job,
        "profile_candidate": profile_candidate,
        "match": match,
        "draft": draft,
        "critique": critique,
        "human_review": human_review,
    }


def route_after_critique(state: JobApplicationState) -> str:
    """Revise while there are fixable issues and revisions left."""
    has_issues = bool(state.get("critique", {}).get("issues"))
    if has_issues and state.get("revision_count", 0) < MAX_REVISIONS:
        return "revise"
    return "review"
