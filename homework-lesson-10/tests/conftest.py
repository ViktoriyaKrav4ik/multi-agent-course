"""Shared fixtures: імпорт hw8 агентів, хелпери для DeepEval."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

# hw8 модулі доступні через sys.path
HW8_DIR = str(Path(__file__).resolve().parent.parent.parent / "homework-lesson-8")
if HW8_DIR not in sys.path:
    sys.path.insert(0, HW8_DIR)

from agents.critic import build_critic_agent  # noqa: E402
from agents.planner import build_planner_agent  # noqa: E402
from agents.research import build_research_agent  # noqa: E402
from config import Settings  # noqa: E402

EVAL_MODEL = "gpt-4o-mini"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def get_settings() -> Settings:
    return Settings()


def sub_cfg(role: str) -> dict:
    s = get_settings()
    return {
        "configurable": {"thread_id": f"test-{role}-{uuid.uuid4().hex[:8]}"},
        "recursion_limit": s.max_subagent_steps,
    }


def invoke_agent(agent, user_text: str, role: str = "agent") -> dict:
    """Invoke a LangGraph agent and return the full result dict."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=sub_cfg(role),
    )


def last_assistant_text(result: dict) -> str:
    """Extract last assistant message text from agent result."""
    from langchain_core.messages import AIMessage

    msgs = result.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    if isinstance(last, AIMessage) and last.content:
        return last.content if isinstance(last.content, str) else str(last.content)
    return str(getattr(last, "content", "") or "")


def extract_tool_calls(result: dict) -> list[str]:
    """Extract tool names called during agent execution."""
    from langchain_core.messages import AIMessage

    names = []
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                names.append(tc.get("name") or tc.get("function", {}).get("name", "unknown"))
    return names


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def planner_agent():
    return build_planner_agent()


@pytest.fixture(scope="session")
def research_agent():
    return build_research_agent()


@pytest.fixture(scope="session")
def critic_agent():
    return build_critic_agent()


@pytest.fixture(scope="session")
def golden_dataset():
    return load_golden_dataset()
