"""Тести Critic Agent: якість критики (GEval)."""

from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from conftest import EVAL_MODEL, invoke_agent, last_assistant_text

# ── Metrics ─────────────────────────────────────────────────────

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one concrete revision_request",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,
)

critique_consistency = GEval(
    name="Critique Consistency",
    evaluation_steps=[
        "If is_fresh, is_complete, and is_well_structured are all true, verdict should be APPROVE",
        "If any of these flags is false and there are significant gaps, verdict should be REVISE",
        "Check that strengths and gaps lists are non-empty and specific",
        "The verdict must be logically consistent with the flags and gaps",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,
)


# ── Tests ───────────────────────────────────────────────────────

GOOD_RESEARCH = """
## Findings: BM25 vs Semantic Search

### BM25 (Okapi BM25)
BM25 is a probabilistic keyword-based ranking algorithm. It scores documents using term frequency,
inverse document frequency, and document length normalization. Parameters k1 (typically 1.2-2.0)
and b (typically 0.75) control saturation and length normalization.
Source: Robertson & Zaragoza, 2009.

### Semantic Search
Semantic search uses dense vector embeddings (e.g., from sentence-transformers) to capture meaning.
Queries and documents are encoded into the same vector space, and similarity is computed via cosine
or dot product. Models like all-MiniLM-L6-v2 produce 384-dim embeddings.
Source: Reimers & Gurevych, 2019.

### Hybrid Approach
Combining BM25 + semantic via ensemble retrieval (e.g., weighted 0.5/0.5) improves recall.
Cross-encoder reranking (e.g., BAAI/bge-reranker-base) as a second stage further boosts precision.
Source: Benchmarks from MTEB leaderboard, 2024.
"""

WEAK_RESEARCH = """
BM25 and semantic search are different.
BM25 uses keywords.
Semantic search uses vectors.
They can be combined.
"""


def test_critique_approves_good_research(critic_agent):
    """Critic should APPROVE well-structured, complete research."""
    result = invoke_agent(critic_agent, GOOD_RESEARCH, role="critic")

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2, ensure_ascii=False)
    else:
        output = last_assistant_text(result)

    test_case = LLMTestCase(input=GOOD_RESEARCH, actual_output=output)
    assert_test(test_case, [critique_quality])


def test_critique_revises_weak_research(critic_agent):
    """Critic should REVISE shallow, unsourced research."""
    result = invoke_agent(critic_agent, WEAK_RESEARCH, role="critic")

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2, ensure_ascii=False)
    else:
        output = last_assistant_text(result)

    test_case = LLMTestCase(input=WEAK_RESEARCH, actual_output=output)
    assert_test(test_case, [critique_quality, critique_consistency])
