"""LangGraph workflow for the deterministic local Agent fallback."""
from __future__ import annotations
import re
from typing import Any, TypedDict
try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = START = StateGraph = None
from .models import Evidence
from .agent import answer_from_evidence, search_keyword_tool, search_semantic_tool, verify_coverage_tool


def extract_goal_timestamps_ms(goal: str) -> list[int]:
    """Extract explicit mm:ss or second-based anchors from a user question."""
    candidates: list[tuple[int, int]] = []
    occupied: list[tuple[int, int]] = []

    def add(match: Any, milliseconds: int) -> None:
        start, end = match.span()
        if milliseconds < 0 or any(start < right and end > left for left, right in occupied):
            return
        occupied.append((start, end))
        candidates.append((start, milliseconds))

    for match in re.finditer(r"(?<!\d)(\d{1,2}):(\d{2}):(\d{2})(?!\d)", goal):
        add(match, (int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))) * 1000)
    for match in re.finditer(r"(?<!\d)(\d{1,3}):(\d{2})(?!\d)", goal):
        add(match, (int(match.group(1)) * 60 + int(match.group(2))) * 1000)
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:秒|s)\b", goal, flags=re.I):
        add(match, int(float(match.group(1)) * 1000))
    for match in re.finditer(
        r"(?:(?:\u7b2c|\u7ea6|\u5927\u7ea6|\u5728|around|about|at)\s*)?"
        r"(\d+(?:\.\d+)?)\s*(?:\u5206\u949f|\u5206|minutes?|mins?)"
        r"(?:\s*(?:\u5de6\u53f3|\u9644\u8fd1|around|about))?",
        goal,
        flags=re.I,
    ):
        add(match, int(float(match.group(1)) * 60 * 1000))
    for match in re.finditer(
        r"(?:(?:\u7b2c|\u7ea6|\u5927\u7ea6|\u5728|around|about|at)\s*)?"
        r"(\d+(?:\.\d+)?)\s*(?:\u79d2|seconds?|secs?)"
        r"(?:\s*(?:\u5de6\u53f3|\u9644\u8fd1|around|about))?",
        goal,
        flags=re.I,
    ):
        add(match, int(float(match.group(1)) * 1000))
    return [value for _, value in sorted(candidates)][:5]

class AgentState(TypedDict):
    question: str
    video_id: str | None
    evidence: list[Evidence]
    answer: str
    trace: list[str]
    grounded: bool
    citations: list[dict[str, Any]]
    provider: str
    adequate: bool


AGENT_TOOL_NAMES = [
    "get_video_metadata",
    "get_timeline_overview",
    "search_timeline",
    "get_evidence_window",
    "verify_citations",
    "generate_report",
]


class ToolCall:
    def __init__(self, tool_name: str, args: dict[str, Any]):
        self.tool_name = tool_name
        self.args = args

    def __repr__(self) -> str:
        return f"ToolCall({self.tool_name}, {self.args})"


AVAILABLE_TOOLS = {
    "search_semantic": {
        "description": "Search for evidence using semantic/vector similarity",
        "params": ["question", "video_id"],
    },
    "search_keyword": {
        "description": "Search for evidence using keyword/lexical matching",
        "params": ["question", "video_id"],
    },
    "verify_coverage": {
        "description": "Check if retrieved evidence adequately covers the question",
        "params": ["question", "evidence"],
    },
}

def retrieve_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    video_id = state["video_id"]
    
    trace_msg = f"Retrieve: Invoking search tools for '{question[:40]}...'"
    state_trace = (state.get("trace", []) or []) + [trace_msg]
    
    semantic_results = search_semantic_tool(question, video_id)
    trace_msg = f"Tool: semantic_search returned {len(semantic_results)} results"
    state_trace.append(trace_msg)
    keyword_results = search_keyword_tool(question, video_id)
    trace_msg = f"Tool: keyword_search returned {len(keyword_results)} results"
    state_trace.append(trace_msg)
    combined: list[Evidence] = []
    seen_ids: set[int] = set()
    for item in semantic_results + keyword_results:
        if item.id not in seen_ids:
            combined.append(item)
            seen_ids.add(item.id)
    if not combined:
        state_trace.append("Retrieve: No evidence matched the question")
    return {"evidence": combined[:5], "trace": state_trace}


def verify_node(state: AgentState) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    question = state["question"]
    
    coverage = verify_coverage_tool(question, evidence)
    trace_msg = f"Verify: {coverage['reason']} (adequate={coverage['adequate']})"
    state_trace = (state.get("trace", []) or []) + [trace_msg]
    
    if not coverage["adequate"]:
        return {
            "answer": "No sufficient evidence found to answer your question.",
            "grounded": False,
            "citations": [],
            "provider": "refusal",
            "adequate": False,
            "trace": state_trace,
        }
    
    return {"adequate": True, "trace": state_trace}


def answer_node(state: AgentState) -> dict[str, Any]:
    result = answer_from_evidence(state["question"], state["evidence"])
    provider = result["provider"]
    trace_msg = f"Answer: {provider} generated grounded={result['grounded']}"
    return {
        "answer": result["answer"],
        "grounded": result["grounded"],
        "citations": result["citations"],
        "provider": provider,
        "adequate": True,
        "trace": (state.get("trace", []) or []) + [trace_msg],
    }


def should_answer(state: AgentState) -> str:
    if not state.get("adequate", False):
        return "refuse"
    return "answer"


def build_agent_graph():
    if StateGraph is None or END is None or START is None:
        return None
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("answer", answer_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "verify")
    workflow.add_conditional_edges(
        "verify",
        should_answer,
        {"refuse": END, "answer": "answer"},
    )
    workflow.add_edge("answer", END)
    return workflow.compile()


AGENT_GRAPH = build_agent_graph()
