"""Run the agent workflow on the sample job posting and CV.

Usage:
    python demo.py                     # offline, deterministic, asks you to approve
    python demo.py --auto-approve      # skip the interactive approval
    python demo.py --save              # write examples/sample_output.md
    LLM_BASE_URL=... LLM_API_KEY=... python demo.py --llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.llm import MockLLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()

    llm = None
    if args.llm:
        from src.llm import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM()

    examples = Path(__file__).parent / "examples"
    job_text = (examples / "sample_job.txt").read_text(encoding="utf-8")
    cv_text = (examples / "sample_cv.txt").read_text(encoding="utf-8")

    try:
        from src.graph import build_graph

        graph = build_graph(llm)
        state = graph.invoke(
            {"job_text": job_text, "cv_text": cv_text,
             "auto_approve": args.auto_approve, "revision_count": 0, "history": []}
        )
        engine = "langgraph"
    except ImportError:
        from src.runner import run

        state = run(job_text, cv_text, auto_approve=args.auto_approve, llm=llm)
        engine = "plain-runner (langgraph not installed)"

    match = state["match_report"]
    print(f"engine: {engine}")
    print(f"flow:   {' -> '.join(state['history'])}\n")
    print(f"role:   {state['job_info']['title']} @ {state['job_info']['company']}")
    print(f"match:  {match['score']:.0%} ({match['verdict']})")
    print(f"  matched: {', '.join(match['matched_skills'])}")
    print(f"  missing: {', '.join(match['missing_skills']) or 'none'}\n")
    print("cover letter draft")
    print("=" * 72)
    print(state["cover_letter"])
    print("=" * 72)
    critique = state["critique"]
    print(f"critique: {'clean' if critique['passed'] else critique['issues']} "
          f"({critique['word_count']} words)")

    if state["status"] == "awaiting_human_review" and not args.auto_approve and sys.stdin.isatty():
        answer = input("\napprove this letter? [y/N] ").strip().lower()
        state["status"] = "approved" if answer == "y" else "awaiting_human_review"
    print(f"status: {state['status']}")

    if args.save:
        out = examples / "sample_output.md"
        out.write_text(
            "# Sample run output\n\n"
            f"**Engine:** {engine}\n\n"
            f"**Flow:** {' -> '.join(state['history'])}\n\n"
            f"**Match:** {match['score']:.0%} ({match['verdict']}) — "
            f"matched: {', '.join(match['matched_skills'])}; "
            f"missing: {', '.join(match['missing_skills']) or 'none'}\n\n"
            "## Cover letter\n\n" + state["cover_letter"] + "\n\n"
            f"**Critique:** {'clean' if critique['passed'] else '; '.join(critique['issues'])}"
            f" ({critique['word_count']} words)\n\n**Status:** {state['status']}\n",
            encoding="utf-8",
        )
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
