<p align="center"><img src="assets/logo.png" width="128" alt="agentic-job-assistant logo"></p>

# agentic-job-assistant

A multi-agent workflow, built with LangGraph, that reads a job posting,
profiles a CV, computes an honest skill match, drafts a cover letter, and
runs it through a critique-and-revise loop before handing it to a human
for approval. Nothing is ever sent automatically.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![langgraph](https://img.shields.io/badge/langgraph-0.2%2B-blueviolet)
![license](https://img.shields.io/badge/license-MIT-green)

## Why

Job applications are a natural agent problem: several distinct skills
(parsing, matching, writing, reviewing) have to hand structured state to
each other, loop when quality is low, and stop for a human decision at the
end. This project demonstrates the pattern end-to-end — and it is honest
by construction: the letter composer only has access to facts extracted
from the posting and the CV, so it cannot invent claims.

## Agent concepts demonstrated

- **Shared state** — a single typed state object flows through the graph.
- **Specialized nodes** — analyze → profile → match → draft → critique →
  human review, each doing one job.
- **Conditional routing** — the critique node routes back to drafting when
  the letter has fixable issues (overclaiming language, unverifiable
  metrics, missing skills), bounded by a revision limit.
- **Deterministic tools, swappable LLM** — parsing, extraction, matching
  and critique are plain Python; the LLM (any OpenAI-compatible endpoint)
  only polishes prose. With the built-in `MockLLM` the whole graph runs
  and tests offline.
- **Human in the loop** — the final node marks the draft
  `awaiting_human_review`; a production deployment would use a LangGraph
  `interrupt()` here.

## Graph

```text
START ─► analyze_job ─► profile_candidate ─► match ─► draft ─► critique
                                                              │
                              ┌── fixable issues, revisions ──┤
                              │              left             │
                              ▼                               ▼
                            draft                      human_review ─► END
```

## Quickstart

```bash
pip install -r requirements.txt
python demo.py                 # deterministic, offline, asks you to approve
python demo.py --auto-approve  # skip the interactive prompt
python demo.py --llm           # polish with a real LLM (LLM_BASE_URL/LLM_API_KEY)
```

## What a run looks like (real output)

```text
engine: langgraph
flow:   analyze_job -> profile_candidate -> match -> draft -> critique -> human_review

role:   Junior AI Engineer @ Najm AI (SAMPLE fictional company)
match:  94% (strong)
  matched: agents, arabic, english, evaluation, fastapi, hugging face, ...
  missing: docker

critique: clean (162 words)
status: approved
```

The drafted letter names the matched skills, walks through real projects,
and — when the posting asks for something missing — says so plainly
instead of overselling. See `examples/sample_output.md`.

## No framework magic

`src/runner.py` executes the exact same flow as a plain Python loop. It
exists so the LangGraph version in `src/graph.py` is never a black box:
read the runner, then read the graph, and the framework's job (state
management, routing, checkpoints) becomes obvious.

## Project structure

```text
src/
  state.py          # JobApplicationState (with a LangGraph history reducer)
  skill_catalog.py  # canonical skill names + aliases
  tools.py          # parsing, skill extraction, matching, letter critique
  llm.py            # MockLLM (offline) + OpenAI-compatible client
  agents.py         # the six node functions + routing
  graph.py          # LangGraph wiring
  runner.py         # framework-free equivalent of the graph
examples/           # sample job posting, sample CV, sample output
tests/              # offline tests for tools, runner and the graph
```

## Roadmap

- [ ] Real `interrupt()`-based human review with graph persistence
- [ ] Web search tool for live postings (with source logging)
- [ ] Per-application tracking (SQLite) and follow-up reminders
- [ ] Streamlit UI

## Notes

- Developed with AI pair-programming assistance; design and review decisions
  are my own.
- The sample company in `examples/` is fictional and marked as such.
- License: MIT.

## Real LLM run (Gemini API free tier)
`LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai LLM_MODEL=gemini-3.1-flash-lite LLM_API_KEY=... python demo.py --llm` ran end to end on Nov 26, 2026: the LLM polished the cover letter, the Critic still flagged "company name not mentioned" and the run stopped at `awaiting_human_review`. The first attempt hit a transient 503 from the free tier, so the client now retries 429/5xx with backoff.
