"""Тести Research Agent: groundedness відповіді (GEval)."""

from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from conftest import EVAL_MODEL, invoke_agent, last_assistant_text

# ── Metrics ─────────────────────────────────────────────────────

groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model=EVAL_MODEL,
    threshold=0.7,
)

research_completeness = GEval(
    name="Research Completeness",
    evaluation_steps=[
        "Check whether the research output addresses all aspects of the input query",
        "Check that the output contains specific facts, not just general statements",
        "Check that sources or references are mentioned",
        "Penalize responses that only partially cover the topic",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.6,
)


# ── Tests ───────────────────────────────────────────────────────

def test_research_grounded(research_agent):
    """Researcher's output is grounded in retrieved context."""
    query = (
        "Research the differences between BM25 and semantic search. "
        "Use knowledge_search and web_search to gather information."
    )
    result = invoke_agent(research_agent, query, role="research")
    output = last_assistant_text(result)

    retrieval_context = []
    from langchain_core.messages import ToolMessage
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.content:
            retrieval_context.append(msg.content[:2000])

    if not retrieval_context:
        retrieval_context = ["No retrieval context captured"]

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=retrieval_context,
    )
    assert_test(test_case, [groundedness])


def test_research_completeness(research_agent):
    """Researcher covers all aspects of the query."""
    query = (
        "Research how cross-encoder reranking works and its benefits for RAG systems. "
        "Include specific examples and comparisons."
    )
    result = invoke_agent(research_agent, query, role="research")
    output = last_assistant_text(result)

    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [research_completeness])
