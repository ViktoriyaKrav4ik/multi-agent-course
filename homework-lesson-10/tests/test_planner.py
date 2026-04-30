"""Тести Planner Agent: якість плану (GEval)."""

from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from conftest import EVAL_MODEL, invoke_agent, last_assistant_text

# ── Metrics ─────────────────────────────────────────────────────

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains a clear goal statement",
        "Check that search_queries are specific and actionable (not vague)",
        "Check that there are at least 3 search queries",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that output_format describes the expected report structure",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,
)

plan_specificity = GEval(
    name="Plan Specificity",
    evaluation_steps=[
        "Extract all search queries from the plan",
        "For each query, check if it is specific enough to return useful results",
        "Vague queries like 'AI research' or 'machine learning' score low",
        "Specific queries like 'BM25 vs semantic search comparison 2025' score high",
        "Score = ratio of specific queries to total queries",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.6,
)


# ── Tests ───────────────────────────────────────────────────────

def test_plan_quality_rag_comparison(planner_agent):
    """Planner creates a quality plan for a RAG comparison query."""
    query = "Compare naive RAG vs sentence-window retrieval approaches"
    result = invoke_agent(planner_agent, query, role="planner")

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2, ensure_ascii=False)
    else:
        output = last_assistant_text(result)

    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [plan_quality])


def test_plan_quality_multi_agent(planner_agent):
    """Planner creates a quality plan for a multi-agent systems query."""
    query = "What are the benefits of multi-agent systems over single-agent?"
    result = invoke_agent(planner_agent, query, role="planner")

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2, ensure_ascii=False)
    else:
        output = last_assistant_text(result)

    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [plan_quality])


def test_plan_specificity(planner_agent):
    """Search queries in the plan are specific, not vague."""
    query = "How does cross-encoder reranking improve retrieval in RAG?"
    result = invoke_agent(planner_agent, query, role="planner")

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2, ensure_ascii=False)
    else:
        output = last_assistant_text(result)

    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [plan_specificity])
