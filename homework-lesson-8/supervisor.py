"""
Supervisor: plan → research → critique → save_report (HITL на save_report).
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agents.critic import build_critic_agent
from agents.planner import build_planner_agent
from agents.research import build_research_agent
from config import SUPERVISOR_SYSTEM_PROMPT, Settings
from tools import save_report

settings = Settings()

_planner_agent: Any = None
_research_agent: Any = None
_critic_agent: Any = None


def _get_planner():
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = build_planner_agent()
    return _planner_agent


def _get_research():
    global _research_agent
    if _research_agent is None:
        _research_agent = build_research_agent()
    return _research_agent


def _get_critic():
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = build_critic_agent()
    return _critic_agent


def _sub_cfg(role: str) -> dict:
    return {
        "configurable": {"thread_id": f"sub-{role}-{uuid.uuid4().hex[:10]}"},
        "recursion_limit": settings.max_subagent_steps,
    }


def _last_assistant_text(result: dict) -> str:
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    if isinstance(last, AIMessage):
        if last.content:
            if isinstance(last.content, str):
                return last.content
            return str(last.content)
    text = getattr(last, "text", None)
    if text:
        return str(text)
    return str(getattr(last, "content", "") or "")


@tool
def plan(request: str) -> str:
    """Побудувати структурований план дослідження (goal, search_queries, sources, output_format)."""
    agent = _get_planner()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config=_sub_cfg("planner"),
    )
    structured = result.get("structured_response")
    if structured is not None:
        return structured.model_dump_json(indent=2, ensure_ascii=False)
    return _last_assistant_text(result)


@tool
def research(request: str) -> str:
    """Виконати дослідження за планом або інструкціями (knowledge_search, web_search, read_url)."""
    agent = _get_research()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config=_sub_cfg("research"),
    )
    return _last_assistant_text(result)


@tool
def critique(findings: str) -> str:
    """Оцінити якість дослідження; може викликати ті самі інструменти для перевірки фактів."""
    agent = _get_critic()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": findings}]},
        config=_sub_cfg("critic"),
    )
    structured = result.get("structured_response")
    if structured is not None:
        return structured.model_dump_json(indent=2, ensure_ascii=False)
    return _last_assistant_text(result)


def build_supervisor():
    s = Settings()
    llm = ChatOpenAI(
        model=s.model_name,
        api_key=s.api_key.get_secret_value(),
        temperature=0,
    )
    prompt = SUPERVISOR_SYSTEM_PROMPT.format(max_revision_rounds=s.max_revision_rounds)
    return create_agent(
        llm,
        tools=[plan, research, critique, save_report],
        system_prompt=prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"save_report": True},
                description_prefix="⏸️  Збереження звіту потребує підтвердження",
            ),
        ],
        checkpointer=InMemorySaver(),
    )


supervisor = build_supervisor()
