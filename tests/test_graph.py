import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

BASE = Path(__file__).resolve().parents[1]
JOB = BASE.joinpath("examples/sample_job.txt").read_text(encoding="utf-8")
CV = BASE.joinpath("examples/sample_cv.txt").read_text(encoding="utf-8")


def test_runner_end_to_end():
    from src.runner import run

    state = run(JOB, CV, auto_approve=True)
    assert state["status"] == "approved"
    assert state["cover_letter"].strip()
    assert state["critique"]["passed"]
    assert state["history"][0] == "analyze_job"


def test_langgraph_matches_runner():
    langgraph = pytest.importorskip("langgraph")
    from src.graph import build_graph

    state = build_graph().invoke(
        {"job_text": JOB, "cv_text": CV, "auto_approve": True,
         "revision_count": 0, "history": []}
    )
    assert state["status"] == "approved"
    assert state["match_report"]["score"] >= 0.5


if __name__ == "__main__":
    test_runner_end_to_end()
    test_langgraph_matches_runner()
    print("graph tests passed")
