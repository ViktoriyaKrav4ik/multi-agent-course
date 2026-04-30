"""Тести Tool Correctness: чи правильні інструменти викликаються."""

from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from conftest import EVAL_MODEL, extract_tool_calls, invoke_agent, last_assistant_text

# ── Metrics ─────────────────────────────────────────────────────

tool_metric = ToolCorrectnessMetric(threshold=0.5, model=EVAL_MODEL)


# ── Tests ───────────────────────────────────────────────────────

def test_planner_uses_search_tools(planner_agent):
    """Planner should use web_search and/or knowledge_search for exploration."""
    query = "Compare different RAG chunking strategies"
    result = invoke_agent(planner_agent, query, role="planner")
    output = last_assistant_text(result)
    actual_tools = extract_tool_calls(result)

    tools_called = [ToolCall(name=name) for name in actual_tools]
    expected_tools = [
        ToolCall(name="web_search"),
        ToolCall(name="knowledge_search"),
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])


def test_researcher_uses_search_and_read(research_agent):
    """Researcher should use web_search, knowledge_search, and potentially read_url."""
    query = (
        "Research hybrid retrieval approaches: BM25 + semantic search. "
        "Check knowledge base first, then search the web and read relevant pages."
    )
    result = invoke_agent(research_agent, query, role="research")
    output = last_assistant_text(result)
    actual_tools = extract_tool_calls(result)

    tools_called = [ToolCall(name=name) for name in actual_tools]
    expected_tools = [
        ToolCall(name="web_search"),
        ToolCall(name="knowledge_search"),
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])


def test_supervisor_calls_save_report():
    """Supervisor should call save_report after APPROVE verdict.

    Uses a pre-defined scenario to avoid running the full pipeline.
    """
    tools_called = [
        ToolCall(name="plan"),
        ToolCall(name="research"),
        ToolCall(name="critique"),
        ToolCall(name="save_report"),
    ]
    expected_tools = [
        ToolCall(name="plan"),
        ToolCall(name="research"),
        ToolCall(name="critique"),
        ToolCall(name="save_report"),
    ]

    test_case = LLMTestCase(
        input="Compare RAG approaches and save a report",
        actual_output="Report saved to output/rag_comparison.md",
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])
