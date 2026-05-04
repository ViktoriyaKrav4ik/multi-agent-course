"""
LLM-as-a-Judge тести (потрібен OPENAI_API_KEY у .env).

Запуск з каталогу market-analyst:
  pytest tests/test_llm_judge.py -v --tb=short

Повний пайплайн (довго, мережа):
  set RUN_MARKET_ANALYST_E2E=1
  pytest tests/test_llm_judge.py -v -k e2e
"""

from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

from schemas import CriticFeedback, DraftReport, Finding


def _api_configured() -> bool:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return True
    try:
        from config import Settings

        return bool(Settings().api_key.get_secret_value())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _api_configured(), reason="Потрібен OPENAI_API_KEY у .env або середовищі")


def test_analyst_sources_and_specificity_judge():
    """Analyst: звіт з джерелами та конкретикою (Judge)."""
    from retriever import index_ready

    if not index_ready():
        pytest.skip("Спочатку python ingest.py для corpus/")

    from agents.analyst import build_analyst_agent
    from config import Settings

    s = Settings()
    agent = build_analyst_agent()
    msg = (
        "Тема: агроринок України (зерно та олійні)\n"
        "Скоуп: ціни, логістика експорту, інпути 2024–2026\n"
        "Фокусні напрями: експортні коридори, добрива, погодні ризики\n"
        "Зроби стислий але обґрунтований драфт."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config={
            "configurable": {"thread_id": "test-analyst-judge"},
            "recursion_limit": s.max_analyst_steps,
        },
    )
    structured = result.get("structured_response")
    assert structured is not None
    draft = (
        structured
        if isinstance(structured, DraftReport)
        else DraftReport.model_validate(structured)
    )
    blob = draft.model_dump_json(indent=2, ensure_ascii=False)

    from judge_utils import llm_judge

    verdict = llm_judge(
        criteria=(
            "1) Є хоча б одне явне джерело (URL або назва файлу корпусу) у полі sources. "
            "2) Findings не лише загальні фрази на кшталт «ринок зростає» без деталей — "
            "мають бути конкретні аспекти (культури, логістика, добрива, експорт, маржа тощо)."
        ),
        artifact=blob,
    )
    assert verdict.passed, verdict.reasoning


def test_critic_finds_bias_judge():
    """Critic: виявляє упередженість / відсутність джерел (Judge перевіряє відповідь Критика)."""
    from agents.critic import build_critic_agent
    from config import Settings

    bad = DraftReport(
        executive_summary="Ринок ідеально позитивний без жодних ризиків.",
        findings=[
            Finding(
                title="Тільки успіх",
                detail="Усі показники чудові, конкуренти не становлять загрози.",
            )
        ],
        sources=[],
        data_points=["Зростання 100% без джерела"],
    )

    s = Settings()
    agent = build_critic_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Рев'ю цього драфта:\n"
                    + bad.model_dump_json(indent=2, ensure_ascii=False),
                }
            ]
        },
        config={
            "configurable": {"thread_id": "test-critic-judge"},
            "recursion_limit": s.max_critic_steps,
        },
    )
    structured = result.get("structured_response")
    assert structured is not None
    fb = (
        structured
        if isinstance(structured, CriticFeedback)
        else CriticFeedback.model_validate(structured)
    )
    assert fb.verdict == "NEEDS_REVISION", (
        f"Очікувалось NEEDS_REVISION для упередженого драфта, отримано {fb.verdict}"
    )

    blob = fb.model_dump_json(indent=2, ensure_ascii=False)

    from judge_utils import llm_judge

    verdict = llm_judge(
        criteria=(
            "Підтверди: у feedback є конкретні issues або missing_perspectives, що стосуються "
            "відсутності джерел / однобокого оптимізму / необґрунтованих тверджень."
        ),
        artifact=blob,
    )
    assert verdict.passed, verdict.reasoning


def test_compiler_preserves_structure_judge():
    """Compiler: FinalReport містить секції; summary відповідає знахідкам (Judge)."""
    from graph import compiler_node

    draft = DraftReport(
        executive_summary="Урожай завжди рекордний без ризиків; ціни завжди на користь аграрія.",
        findings=[
            Finding(
                title="Лише позитив",
                detail="Жодних погодних або логістичних ризиків не існує.",
            ),
            Finding(
                title="Інпути",
                detail="Добрива завжди дешеві й доступні без обмежень.",
            ),
        ],
        sources=["corpus/agro_01_market_overview.md", "https://example.org/placeholder"],
        data_points=["Експорт зерна зріс на 200% без джерела"],
    )
    state = {
        "topic": "Агроринок України",
        "scope": "2025–2026",
        "focus_areas": ["зерно", "експорт"],
        "session_id": "test-compiler-judge",
        "analyst_attempts": 1,
        "draft": draft.model_dump(),
        "feedback": {
            "verdict": "APPROVED",
            "issues": [],
            "missing_perspectives": [],
            "fact_check_results": [],
            "score": 0.9,
        },
    }
    out = compiler_node(state)
    fr = out.get("final_report")
    assert fr and isinstance(fr, dict)
    blob = json.dumps(fr, ensure_ascii=False, indent=2)

    from judge_utils import llm_judge

    verdict = llm_judge(
        criteria=(
            "У JSON є непорожні key_findings, recommendations, sources, methodology. "
            "Executive summary не суперечить ключовим знахідкам з драфту (зерно, експорт, інпути)."
        ),
        artifact=blob,
    )
    assert verdict.passed, verdict.reasoning


@pytest.mark.skipif(
    os.environ.get("RUN_MARKET_ANALYST_E2E", "").strip().lower() not in ("1", "true", "yes"),
    reason="Дорогий тест: встановіть RUN_MARKET_ANALYST_E2E=1",
)
def test_end_to_end_relevance_judge():
    """E2E: повний граф + Judge перевіряє відповідність topic/scope."""
    from retriever import index_ready

    if not index_ready():
        pytest.skip("Спочатку python ingest.py")

    from graph import app, set_tracing_callbacks

    set_tracing_callbacks([])
    state = {
        "topic": "Агроринок України: зерно та олійні",
        "scope": "Короткі висновки для стейкхолдерів, без води",
        "focus_areas": ["експорт", "добрива", "логістика"],
        "session_id": str(uuid4()),
        "analyst_attempts": 0,
    }
    from config import Settings

    s = Settings()
    out = app.invoke(
        state,
        config={"recursion_limit": s.max_analyst_critic_iterations * 10 + 40},
    )
    fr = out.get("final_report") or {}
    text = json.dumps(fr, ensure_ascii=False, indent=2) if fr else ""
    assert text

    from judge_utils import llm_judge

    verdict = llm_judge(
        criteria=(
            "Звіт релевантний темі агроринку України (зерно/олійні) та скоупу стейкхолдерського огляду. "
            "Має бути баланс (хоча б згадка ризиків або обмежень), не лише промо одного боку без застережень."
        ),
        artifact=text[:12000],
    )
    assert verdict.passed, verdict.reasoning
