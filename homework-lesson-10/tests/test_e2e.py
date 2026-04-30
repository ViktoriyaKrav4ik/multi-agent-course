"""End-to-end тести: повний pipeline на golden dataset."""

from __future__ import annotations

import uuid

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from conftest import EVAL_MODEL, load_golden_dataset

# ── Metrics ─────────────────────────────────────────────────────

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL)

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict 'expected output'",
        "Penalize omission of critical details from expected output",
        "Different wording of the same concept is acceptable",
        "Extra correct information beyond expected output is acceptable",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.6,
)

# Custom metric: чи містить відповідь джерела/посилання
citation_presence = GEval(
    name="Citation Presence",
    evaluation_steps=[
        "Check if the actual output mentions specific sources (URLs, paper titles, authors)",
        "At least one concrete source reference should be present for a research response",
        "Vague references like 'according to research' without specifics score low",
        "Score 0 if no sources at all, 0.5 for vague references, 1.0 for specific citations",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.5,
)


def _run_supervisor(user_text: str) -> str:
    """Run the full Supervisor pipeline and return the final text (without HITL)."""
    from supervisor import build_supervisor
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from config import SUPERVISOR_SYSTEM_PROMPT, Settings
    from tools import web_search, knowledge_search, read_url

    s = Settings()

    # Simplified supervisor without HITL for automated testing
    llm = ChatOpenAI(
        model=s.model_name,
        api_key=s.api_key.get_secret_value(),
        temperature=0,
    )

    from agents.planner import build_planner_agent
    from agents.research import build_research_agent
    from agents.critic import build_critic_agent
    from langchain_core.tools import tool

    planner_agent = build_planner_agent()
    research_agent_inst = build_research_agent()
    critic_agent = build_critic_agent()

    @tool
    def plan(request: str) -> str:
        """Build a structured research plan."""
        cfg = {"configurable": {"thread_id": f"e2e-plan-{uuid.uuid4().hex[:8]}"}, "recursion_limit": s.max_subagent_steps}
        result = planner_agent.invoke({"messages": [{"role": "user", "content": request}]}, config=cfg)
        structured = result.get("structured_response")
        if structured is not None:
            return structured.model_dump_json(indent=2, ensure_ascii=False)
        msgs = result.get("messages", [])
        return str(msgs[-1].content) if msgs else ""

    @tool
    def research(request: str) -> str:
        """Execute research based on a plan."""
        cfg = {"configurable": {"thread_id": f"e2e-res-{uuid.uuid4().hex[:8]}"}, "recursion_limit": s.max_subagent_steps}
        result = research_agent_inst.invoke({"messages": [{"role": "user", "content": request}]}, config=cfg)
        msgs = result.get("messages", [])
        return str(msgs[-1].content) if msgs else ""

    @tool
    def critique(findings: str) -> str:
        """Evaluate research quality."""
        cfg = {"configurable": {"thread_id": f"e2e-crit-{uuid.uuid4().hex[:8]}"}, "recursion_limit": s.max_subagent_steps}
        result = critic_agent.invoke({"messages": [{"role": "user", "content": findings}]}, config=cfg)
        structured = result.get("structured_response")
        if structured is not None:
            return structured.model_dump_json(indent=2, ensure_ascii=False)
        msgs = result.get("messages", [])
        return str(msgs[-1].content) if msgs else ""

    prompt = SUPERVISOR_SYSTEM_PROMPT.format(max_revision_rounds=s.max_revision_rounds)

    # No save_report and no HITL for automated testing
    agent = create_agent(llm, tools=[plan, research, critique], system_prompt=prompt)

    cfg = {
        "configurable": {"thread_id": f"e2e-{uuid.uuid4().hex[:8]}"},
        "recursion_limit": s.max_iterations * 6 + 24,
    }
    result = agent.invoke({"messages": [HumanMessage(content=user_text)]}, config=cfg)
    msgs = result.get("messages", [])
    if msgs:
        last = msgs[-1]
        if isinstance(last, AIMessage) and last.content:
            return last.content if isinstance(last.content, str) else str(last.content)
    return ""


# ── Tests ───────────────────────────────────────────────────────

def test_e2e_happy_path():
    """End-to-end test on happy path examples from golden dataset."""
    dataset = load_golden_dataset()
    happy = [ex for ex in dataset if ex["category"] == "happy_path"][:2]

    for example in happy:
        output = _run_supervisor(example["input"])

        test_case = LLMTestCase(
            input=example["input"],
            actual_output=output,
            expected_output=example["expected_output"],
        )
        assert_test(test_case, [answer_relevancy, correctness])


def test_e2e_with_citations():
    """End-to-end: research responses should include source citations."""
    query = "Compare BM25 and semantic search approaches for document retrieval"
    output = _run_supervisor(query)

    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [citation_presence])
