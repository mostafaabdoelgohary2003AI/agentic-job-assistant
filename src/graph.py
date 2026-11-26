"""The LangGraph wiring.

Graph shape:

    START -> analyze_job -> profile_candidate -> match -> draft -> critique
                                                                  |
                                          +--- issues & revisions left ---+
                                          |                               |
                                          ▼                               ▼
                                        draft                      human_review -> END

`human_review` is where a production deployment would use a LangGraph
`interrupt()` so a person approves the letter before anything is sent.
Here it flips `status` based on the `auto_approve` input flag.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import make_nodes, route_after_critique
from .state import JobApplicationState


def build_graph(llm=None):
    nodes = make_nodes(llm)
    builder = StateGraph(JobApplicationState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "analyze_job")
    builder.add_edge("analyze_job", "profile_candidate")
    builder.add_edge("profile_candidate", "match")
    builder.add_edge("match", "draft")
    builder.add_edge("draft", "critique")
    builder.add_conditional_edges(
        "critique", route_after_critique, {"revise": "draft", "review": "human_review"}
    )
    builder.add_edge("human_review", END)
    return builder.compile()
